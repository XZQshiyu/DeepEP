"""Regression test for exact-local intranode dispatch/combine bypass.

Run on one 8-GPU NVLink node:

    python tests/test_intranode_local_bypass.py

The off-diagonal traffic is fixed while the exact-local diagonal grows from
8,500 to 16,000 destination copies per source rank.  A real local bypass must
keep the cached dispatch+combine critical-path latency nearly unchanged.
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.distributed as dist

import deep_ep
from utils import init_dist


EP_SIZE = 8
NUM_EXPERTS = 128
NUM_TOKENS = 16_384
HIDDEN = 2_048
REMOTE_COPIES = 8_500
WARMUPS = 10
ITERATIONS = 50


def _route(diagonal: int, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_token_in_rank = torch.zeros(
        (NUM_TOKENS, EP_SIZE), dtype=torch.bool, device="cuda"
    )
    for destination in range(EP_SIZE):
        count = diagonal if destination == rank else REMOTE_COPIES
        is_token_in_rank[:count, destination] = True
    num_tokens_per_rank = is_token_in_rank.sum(dim=0, dtype=torch.int32)

    # Top-k metadata is intentionally absent in this payload-only regression.
    # Dispatch still requires the expert-count tensor to define expert width.
    num_tokens_per_expert = torch.zeros(
        (NUM_EXPERTS,), dtype=torch.int32, device="cuda"
    )
    return num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert


def _measure_case(
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
    rank: int,
    diagonal: int,
) -> float:
    x = torch.full((NUM_TOKENS, HIDDEN), rank, dtype=torch.bfloat16, device="cuda")
    num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert = _route(
        diagonal, rank
    )
    config = deep_ep.Config(20, 8, 256)

    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
    )

    # Validate source-major dispatch order and combine semantics once.
    rank_prefix = handle[0]
    begin = 0
    for source in range(EP_SIZE):
        end = int(rank_prefix[source, rank])
        assert torch.all(recv_x[begin:end] == source)
        begin = end
    combined_x, _, _ = buffer.combine(x=recv_x, handle=handle, config=config)
    expected_copies = is_token_in_rank.sum(dim=1, dtype=torch.int32)
    assert torch.equal(
        combined_x[:, 0].to(torch.int32), expected_copies * rank
    )

    samples_us: list[float] = []
    for iteration in range(WARMUPS + ITERATIONS):
        dist.barrier(group=group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cached_x, _, _, _, _, _ = buffer.dispatch(
            x=x, handle=handle, config=config
        )
        buffer.combine(x=cached_x, handle=handle, config=config)
        end.record()
        end.synchronize()
        elapsed_us = start.elapsed_time(end) * 1_000.0
        critical_us = torch.tensor(elapsed_us, dtype=torch.float64, device="cuda")
        dist.all_reduce(critical_us, op=dist.ReduceOp.MAX, group=group)
        if iteration >= WARMUPS:
            samples_us.append(float(critical_us))
    return statistics.median(samples_us)


def _worker(local_rank: int, num_processes: int, max_ratio: float) -> None:
    rank, world_size, group = init_dist(local_rank, num_processes)
    assert world_size == EP_SIZE
    deep_ep.Buffer.set_channel_schedule_enabled(True)
    buffer = deep_ep.Buffer(
        group,
        int(2e9),
        0,
        explicitly_destroy=True,
    )
    low_us = _measure_case(buffer, group, rank, REMOTE_COPIES)
    high_us = _measure_case(buffer, group, rank, 16_000)
    ratio = high_us / low_us
    if rank == 0:
        print(
            f"local-bypass diagonal scaling: 8500={low_us:.2f} us, "
            f"16000={high_us:.2f} us, ratio={ratio:.4f}, limit={max_ratio:.4f}",
            flush=True,
        )
    assert ratio <= max_ratio, (
        f"exact-local traffic remains on the staged communication path: "
        f"ratio {ratio:.4f} > {max_ratio:.4f}"
    )
    buffer.destroy()
    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=EP_SIZE)
    parser.add_argument("--max-ratio", type=float, default=1.03)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args.max_ratio),
        nprocs=args.num_processes,
    )
