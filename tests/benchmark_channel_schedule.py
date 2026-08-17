"""Paired intranode DeepEP benchmark for destination-aware channel scheduling.

The ``shuffled`` and ``grouped`` cases contain the same per-rank top-k routes.
Only token order differs: ``grouped`` sorts tokens by destination-rank mask to
create channel-level tail load.  Each measured iteration alternates scheduler
off/on order and reports the slowest rank, which determines EP completion time.
"""

import argparse
import json

import torch
import torch.distributed as dist

import deep_ep
from utils import init_dist


def destination_masks(topk_idx: torch.Tensor, num_experts: int, num_ranks: int):
    experts_per_rank = num_experts // num_ranks
    destination_ranks = topk_idx // experts_per_rank
    masks = torch.zeros(topk_idx.size(0), dtype=torch.int64, device=topk_idx.device)
    for topk_column in range(topk_idx.size(1)):
        masks.bitwise_or_(1 << destination_ranks[:, topk_column])
    return masks


def make_routes(
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    rank: int,
):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260813 + rank)
    scores = torch.rand(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    shuffled = torch.topk(scores, num_topk, dim=-1, sorted=False).indices
    masks = destination_masks(shuffled, num_experts, num_ranks)
    grouped = shuffled.index_select(0, torch.argsort(masks, stable=True))
    return {"shuffled": shuffled, "grouped": grouped}


def timed_call(function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    end.synchronize()
    return output, start.elapsed_time(end) * 1000.0


def summarize(values: torch.Tensor):
    critical = values.max(dim=0).values
    spread = values.max(dim=0).values - values.min(dim=0).values
    return {
        "critical_p50_us": float(torch.quantile(critical, 0.50)),
        "critical_p95_us": float(torch.quantile(critical, 0.95)),
        "critical_max_us": float(critical.max()),
        "rank_spread_p50_us": float(torch.quantile(spread, 0.50)),
        "rank_spread_p95_us": float(torch.quantile(spread, 0.95)),
        "rank_mean_us": [float(value) for value in values.mean(dim=1)],
    }


def benchmark_pattern(
    *,
    buffer,
    x,
    topk_idx,
    num_experts,
    dispatch_config,
    combine_config,
    group,
    rank,
    warmups,
    iterations,
    pattern,
):
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    def dispatch_combine(enabled: bool, timed: bool):
        deep_ep.Buffer.set_channel_schedule_enabled(enabled)
        dispatch_args = dict(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            config=dispatch_config,
        )
        if timed:
            torch.cuda.nvtx.range_push(
                f"channel_schedule/{pattern}/{'on' if enabled else 'off'}/dispatch"
            )
            dispatch_output, dispatch_us = timed_call(
                lambda: buffer.dispatch(**dispatch_args)
            )
            torch.cuda.nvtx.range_pop()
        else:
            dispatch_output = buffer.dispatch(**dispatch_args)
            dispatch_us = 0.0
        recv_x, _, _, _, handle, _ = dispatch_output

        if timed:
            torch.cuda.nvtx.range_push(
                f"channel_schedule/{pattern}/{'on' if enabled else 'off'}/combine"
            )
            combine_output, combine_us = timed_call(
                lambda: buffer.combine(recv_x, handle, config=combine_config)
            )
            torch.cuda.nvtx.range_pop()
        else:
            combine_output = buffer.combine(recv_x, handle, config=combine_config)
            combine_us = 0.0
        return combine_output[0], dispatch_us, combine_us

    for enabled in (False, True):
        for _ in range(warmups):
            dispatch_combine(enabled, timed=False)
        torch.cuda.synchronize()
        dist.barrier(group=group)

    samples = torch.empty(
        (2, 2, iterations), dtype=torch.float64, device="cpu"
    )
    final_outputs = {}
    for iteration in range(iterations):
        mode_order = (False, True) if iteration % 2 == 0 else (True, False)
        for enabled in mode_order:
            dist.barrier(group=group)
            torch.cuda.synchronize()
            output, dispatch_us, combine_us = dispatch_combine(enabled, timed=True)
            mode = int(enabled)
            samples[mode, 0, iteration] = dispatch_us
            samples[mode, 1, iteration] = combine_us
            final_outputs[enabled] = output

    assert torch.equal(final_outputs[False], final_outputs[True])

    local_samples = samples.to(device="cuda")
    gathered = [torch.empty_like(local_samples) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_samples, group=group)
    if rank != 0:
        return None

    all_samples = torch.stack(gathered).cpu()
    result = {"pattern": pattern, "iterations": iterations}
    for mode, mode_name in enumerate(("off", "on")):
        result[mode_name] = {
            "dispatch": summarize(all_samples[:, mode, 0]),
            "combine": summarize(all_samples[:, mode, 1]),
        }
    for operation in ("dispatch", "combine"):
        off = result["off"][operation]["critical_p50_us"]
        on = result["on"][operation]["critical_p50_us"]
        result[f"{operation}_critical_p50_improvement_pct"] = (
            100.0 * (off - on) / off
        )
    return result


def run_rank(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks <= 32
    assert args.num_experts % num_ranks == 0

    deep_ep.Buffer.set_num_sms(args.num_sms)
    dispatch_config = deep_ep.Buffer.get_dispatch_config(num_ranks)
    combine_config = deep_ep.Buffer.get_combine_config(num_ranks)
    hidden_bytes = args.hidden * torch.tensor([], dtype=torch.bfloat16).element_size()
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks),
    )
    buffer = deep_ep.Buffer(group, num_nvl_bytes, explicitly_destroy=True)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260813 + rank)
    x = torch.randn(
        (args.num_tokens, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    routes = make_routes(
        args.num_tokens,
        args.num_experts,
        args.num_topk,
        num_ranks,
        rank,
    )
    for pattern in args.patterns:
        result = benchmark_pattern(
            buffer=buffer,
            x=x,
            topk_idx=routes[pattern],
            num_experts=args.num_experts,
            dispatch_config=dispatch_config,
            combine_config=combine_config,
            group=group,
            rank=rank,
            warmups=args.warmups,
            iterations=args.iterations,
            pattern=pattern,
        )
        if result is not None:
            result["config"] = {
                "num_ranks": num_ranks,
                "num_tokens_per_rank": args.num_tokens,
                "hidden": args.hidden,
                "num_topk": args.num_topk,
                "num_experts": args.num_experts,
                "num_sms": args.num_sms,
            }
            print("CHANNEL_SCHEDULE_BENCH " + json.dumps(result), flush=True)

    buffer.destroy()
    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-sms", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--patterns", nargs="+", choices=("shuffled", "grouped"),
        default=("shuffled", "grouped"),
    )
    arguments = parser.parse_args()
    torch.multiprocessing.spawn(
        run_rank,
        args=(arguments.num_processes, arguments),
        nprocs=arguments.num_processes,
    )
