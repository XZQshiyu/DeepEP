"""Single-node correctness smoke for destination-aware channel scheduling."""

import argparse

import torch
import torch.distributed as dist

import deep_ep
from utils import init_dist


def run_rank(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks <= 32
    assert args.num_experts % num_ranks == 0

    config = deep_ep.Config(20, 8, 256)
    hidden_bytes = args.hidden * torch.tensor([], dtype=torch.bfloat16).element_size()
    num_nvl_bytes = max(
        config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks),
        deep_ep.Buffer.get_combine_config(num_ranks).get_nvl_buffer_size_hint(
            hidden_bytes, num_ranks
        ),
    )
    buffer = deep_ep.Buffer(
        group,
        num_nvl_bytes,
        explicitly_destroy=True,
    )

    x = torch.randn(
        (args.num_tokens, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    experts_per_rank = args.num_experts // num_ranks
    destination_ranks = min(args.num_topk, num_ranks)
    topk_idx = torch.arange(
        destination_ranks, dtype=torch.int64, device="cuda"
    ).mul_(experts_per_rank)
    topk_idx = topk_idx.unsqueeze(0).expand(args.num_tokens, -1).contiguous()
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, args.num_experts)

    def dispatch_combine(enabled: bool):
        deep_ep.Buffer.set_channel_schedule_enabled(enabled)
        recv_x, _, _, _, handle, _ = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            config=config,
        )
        combined_x, _, _ = buffer.combine(recv_x, handle, config=config)
        cached_recv_x, _, _, _, _, _ = buffer.dispatch(
            x,
            handle=handle,
            config=config,
        )
        cached_combined_x, _, _ = buffer.combine(
            cached_recv_x,
            handle,
            config=config,
        )
        assert torch.equal(cached_combined_x, combined_x)
        return combined_x, handle

    baseline_x, baseline_handle = dispatch_combine(False)
    scheduled_x, scheduled_handle = dispatch_combine(True)

    assert torch.equal(scheduled_x, baseline_x)
    assert baseline_handle[-2] is None and baseline_handle[-1] is None

    channel_offsets, channel_token_indices = scheduled_handle[-2:]
    assert channel_offsets is not None and channel_token_indices is not None
    assert torch.equal(
        torch.sort(channel_token_indices).values,
        torch.arange(args.num_tokens, dtype=torch.int32, device="cuda"),
    )
    channel_counts = channel_offsets[1:] - channel_offsets[:-1]
    assert int(channel_counts.max() - channel_counts.min()) <= 1

    channel_prefix = scheduled_handle[1]
    destination_counts = torch.diff(
        channel_prefix,
        dim=1,
        prepend=torch.zeros(
            (num_ranks, 1), dtype=torch.int32, device="cuda"
        ),
    )
    active = destination_counts[:destination_ranks]
    assert int(active.max() - active.min()) <= 1

    if rank == 0:
        print(
            "channel schedule smoke passed: "
            f"ranks={num_ranks}, tokens={args.num_tokens}, "
            f"channels={channel_counts.numel()}",
            flush=True,
        )

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=64)
    parsed = parser.parse_args()
    torch.multiprocessing.spawn(
        run_rank,
        args=(parsed.num_processes, parsed),
        nprocs=parsed.num_processes,
    )
