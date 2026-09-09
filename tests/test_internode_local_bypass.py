"""Bytewise A/B regression for normal internode local-copy bypass.

Run with torchrun on 2 or 4 nodes, first --record using the original binary,
then --reference using the patched binary. Covers self/remote-rail/mixed visits,
queue wraparound, cached dispatch, FP8 scales, top-k metadata and biased combine.
No packages beyond the existing DeepEP test environment are required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import deep_ep


def digest(value):
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest())
    if isinstance(value, (tuple, list)):
        return [digest(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record', type=Path)
    mode.add_argument('--reference', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world in (16, 32)
    n, hidden, k = 4097, 2048, 8
    dc, cc = deep_ep.Buffer.get_dispatch_config(world), deep_ep.Buffer.get_combine_config(world)
    deep_ep.Buffer.set_num_sms(20)
    buffer = deep_ep.Buffer(dist.group.WORLD,
        max(c.get_nvl_buffer_size_hint(hidden * 2, world) for c in (dc, cc)),
        max(c.get_rdma_buffer_size_hint(hidden * 2, world) for c in (dc, cc)), explicitly_destroy=True)
    results = {}
    try:
        for scheduled in (False, True):
            deep_ep.Buffer.set_channel_schedule_enabled(scheduled)
            for route in ('self', 'rail', 'mixed', 'sparse', 'skewed', 'empty'):
                for fmt in ('bf16', 'fp8_aligned', 'fp8_unaligned'):
                    torch.manual_seed(1234 + rank)
                    x = torch.randn((n, hidden), dtype=torch.bfloat16, device='cuda')
                    # Identity digits remain exactly representable in both BF16
                    # and E4M3. Scheduler scatter uses atomics, so A/B row order
                    # may differ even when comparing the original binary twice.
                    tokens = torch.arange(n, device='cuda')
                    x[:, 0], x[:, 1] = rank // 16, rank % 16
                    x[:, 2], x[:, 3], x[:, 4] = tokens // 256, (tokens // 16) % 16, tokens % 16
                    weights = torch.randn((n, k), device='cuda')
                    if route == 'empty':
                        topk = torch.full((n, k), -1, dtype=torch.long, device='cuda')
                    elif route == 'skewed':
                        topk = torch.arange(k, device='cuda').expand(n, k).contiguous()
                    elif route in ('self', 'rail'):
                        dst = rank if route == 'self' else ((rank // 8 + 1) % (world // 8)) * 8 + local
                        topk = (dst * 8 + torch.arange(k, device='cuda')).expand(n, k).contiguous()
                    else:
                        topk = torch.rand((n, world * 8), device='cuda').topk(k, dim=-1).indices
                        if route == 'sparse':
                            topk[::7] = -1
                            topk[1::3, 3:] = -1
                    # SourceMeta uses 8-byte loads; two scales keep it aligned
                    # while still exercising the non-16-byte scale copy path.
                    scales = 16 if fmt == 'fp8_aligned' else 2
                    payload = x if fmt == 'bf16' else (x.to(torch.float8_e4m3fn), torch.rand((n, scales), device='cuda'))
                    nr, nn, ne, mask, _ = buffer.get_dispatch_layout(topk, world * 8)
                    recv, ri, rw, counts, handle, event = buffer.dispatch(payload, topk_idx=topk, topk_weights=weights,
                        num_tokens_per_rank=nr, num_tokens_per_rdma_rank=nn, num_tokens_per_expert=ne,
                        is_token_in_rank=mask, config=dc, async_finish=True)
                    event.current_stream_wait()
                    key = f'{scheduled}/{route}/{fmt}'
                    raw_recv_digest = digest(recv)
                    values = (recv[0] if isinstance(recv, tuple) else recv).to(torch.bfloat16)
                    digits = values[:, :5].to(torch.int64)
                    identities = (digits[:, 0] * 16 + digits[:, 1]) * n + digits[:, 2] * 256 + digits[:, 3] * 16 + digits[:, 4]
                    order = identities.argsort()
                    def ordered_digest(v):
                        if isinstance(v, tuple):
                            return [ordered_digest(t) for t in v]
                        return digest(v[order] if scheduled else v)
                    # Raw handles are comparable only with scheduler disabled.
                    # Scheduled tests compare per-token data and invariant counts;
                    # cached reuse below still checks the raw order exactly.
                    checked_handle = (handle[0], handle[4], handle[6], handle[7][order]) if scheduled else handle
                    outputs = dict(recv=ordered_digest(recv), indices=ordered_digest(ri), weights=ordered_digest(rw), counts=counts, handle=digest(checked_handle))
                    cached, _, _, _, _, event = buffer.dispatch(payload, handle=handle, config=dc, async_finish=True)
                    event.current_stream_wait()
                    assert digest(cached) == raw_recv_digest, key
                    # Distinct contributions exercise reduction order and local indexing.
                    compute = (values.float() + rank / 16).to(torch.bfloat16)
                    bias = (torch.randn_like(x), torch.randn_like(x))
                    combined, cw, event = buffer.combine(compute, handle, topk_weights=rw, bias=bias, config=cc, async_finish=True)
                    event.current_stream_wait()
                    outputs.update(combined=digest(combined), combined_weights=digest(cw))
                    # Repeat after cached_notify has rewritten negative head sentinels.
                    again, aw, _ = buffer.combine(compute, handle, topk_weights=rw, bias=bias, config=cc)
                    assert digest(again) == outputs['combined'] and digest(aw) == outputs['combined_weights'], key
                    after_combine, _, _, _, _, _ = buffer.dispatch(payload, handle=handle, config=dc)
                    assert digest(after_combine) == raw_recv_digest, key
                    plain, pw, _ = buffer.combine(compute, handle, config=cc)
                    outputs.update(plain_combined=digest(plain), plain_weights=digest(pw))
                    results[key] = outputs
                    if rank == 0:
                        print('PASS', key, flush=True)
        directory = args.record or args.reference
        path = directory / f'rank{rank:02d}.json'
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f'actual-rank{rank:02d}.json').write_text(json.dumps(results, indent=2) + '\n')
        if args.record:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results, indent=2) + '\n')
        else:
            reference = json.loads(path.read_text())
            mismatches = {key: [field for field in results[key] if results[key][field] != reference[key][field]]
                          for key in results if results[key] != reference[key]}
            assert not mismatches, f'rank{rank} bytewise mismatch: {mismatches}'
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f'rank{rank:02d}.json').write_text(json.dumps(dict(rank=rank, cases=len(results), bytewise_reference=bool(args.reference), passed=True)) + '\n')
        dist.barrier()
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
