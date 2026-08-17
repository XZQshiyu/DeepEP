#include "configs.cuh"
#include "exception.cuh"

namespace deep_ep {

namespace channel_schedule {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxScheduledRanks = 32;
constexpr int kMaxScheduledChannels = 32;

struct WorkspaceLayout {
    unsigned long long* keys;
    int* counts;
    int* cursors;
    int* unique_slots;
    int* token_slots;
    int* unique_count;
    int* channel_write;
    size_t num_bytes;
};

size_t align_workspace(size_t offset) {
    constexpr size_t alignment = 128;
    return (offset + alignment - 1) / alignment * alignment;
}

WorkspaceLayout get_workspace_layout(void* workspace, int table_size,
                                     int num_tokens, int num_channels) {
    auto* base = static_cast<uint8_t*>(workspace);
    size_t offset = 0;
    auto take = [&](size_t bytes) {
        offset = align_workspace(offset);
        auto* ptr = base + offset;
        offset += bytes;
        return ptr;
    };

    auto* keys = reinterpret_cast<unsigned long long*>(
        take(static_cast<size_t>(table_size) * sizeof(unsigned long long)));
    auto* counts = reinterpret_cast<int*>(
        take(static_cast<size_t>(table_size) * sizeof(int)));
    auto* cursors = reinterpret_cast<int*>(
        take(static_cast<size_t>(table_size) * sizeof(int)));
    auto* unique_slots = reinterpret_cast<int*>(
        take(static_cast<size_t>(num_tokens) * sizeof(int)));
    auto* token_slots = reinterpret_cast<int*>(
        take(static_cast<size_t>(num_tokens) * sizeof(int)));
    auto* unique_count = reinterpret_cast<int*>(take(sizeof(int)));
    auto* channel_write = reinterpret_cast<int*>(
        take(static_cast<size_t>(num_channels) * sizeof(int)));
    return {keys, counts, cursors, unique_slots, token_slots,
            unique_count, channel_write, align_workspace(offset)};
}

__device__ __forceinline__ uint32_t mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__device__ __forceinline__ int gcd_int(int lhs, int rhs) {
    while (rhs != 0) {
        int next = lhs % rhs;
        lhs = rhs;
        rhs = next;
    }
    return lhs;
}

__device__ __forceinline__ int channel_start(uint32_t mask, int source_rank,
                                              int num_channels) {
    return static_cast<int>(mix32(mask ^ (0x9e3779b9u * (source_rank + 1))) %
                            static_cast<uint32_t>(num_channels));
}

__device__ __forceinline__ int channel_stride(uint32_t mask, int source_rank,
                                               int num_channels) {
    if (num_channels == 1)
        return 1;
    uint32_t hash = mix32(mask + 0x85ebca6bu * (source_rank + 1));
    int stride = 1 + static_cast<int>(hash % static_cast<uint32_t>(num_channels - 1));
    while (gcd_int(stride, num_channels) != 1) {
        ++stride;
        if (stride >= num_channels)
            stride = 1;
    }
    return stride;
}

__global__ void collect_destination_masks(const bool* is_token_in_rank,
                                          unsigned long long* keys,
                                          int* counts,
                                          int* unique_slots,
                                          int* token_slots,
                                          int* unique_count,
                                          int table_size,
                                          int num_tokens,
                                          int num_ranks) {
    int token_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (token_idx >= num_tokens)
        return;

    uint32_t mask = 0;
    const bool* row = is_token_in_rank + token_idx * num_ranks;
    #pragma unroll
    for (int rank_idx = 0; rank_idx < kMaxScheduledRanks; ++rank_idx) {
        if (rank_idx < num_ranks and row[rank_idx])
            mask |= 1u << rank_idx;
    }

    // Zero is the empty-table sentinel, so store mask + 1 in 64 bits.  This
    // also represents the all-32-ranks mask without overflow.
    unsigned long long stored_key = static_cast<unsigned long long>(mask) + 1ull;
    int slot = static_cast<int>(mix32(mask) & static_cast<uint32_t>(table_size - 1));
    while (true) {
        auto old_key = atomicCAS(keys + slot, 0ull, stored_key);
        if (old_key == 0ull) {
            int unique_idx = atomicAdd(unique_count, 1);
            unique_slots[unique_idx] = slot;
            break;
        }
        if (old_key == stored_key)
            break;
        slot = (slot + 1) & (table_size - 1);
    }

    atomicAdd(counts + slot, 1);
    token_slots[token_idx] = slot;
}

__global__ void build_channel_metadata(const unsigned long long* keys,
                                       const int* counts,
                                       int* cursors,
                                       const int* unique_slots,
                                       const int* unique_count,
                                       int* channel_write,
                                       int* channel_offsets,
                                       int* rank_channel_prefix,
                                       int* rdma_channel_prefix,
                                       int num_ranks,
                                       int num_channels,
                                       int source_rank) {
    __shared__ int channel_counts[kMaxScheduledChannels];
    int thread_idx = static_cast<int>(threadIdx.x);
    int num_rdma_ranks = (num_ranks + NUM_MAX_NVL_PEERS - 1) / NUM_MAX_NVL_PEERS;

    for (int i = thread_idx; i < num_channels; i += blockDim.x) {
        channel_counts[i] = 0;
        channel_write[i] = 0;
    }
    for (int i = thread_idx; i < num_ranks * num_channels; i += blockDim.x)
        rank_channel_prefix[i] = 0;
    if (rdma_channel_prefix != nullptr) {
        for (int i = thread_idx; i < num_rdma_ranks * num_channels; i += blockDim.x)
            rdma_channel_prefix[i] = 0;
    }
    __syncthreads();

    int num_groups = *unique_count;
    for (int group_idx = thread_idx; group_idx < num_groups; group_idx += blockDim.x) {
        int slot = unique_slots[group_idx];
        uint32_t mask = static_cast<uint32_t>(keys[slot] - 1ull);
        int count = counts[slot];
        int base = count / num_channels;
        int remainder = count - base * num_channels;
        int start = channel_start(mask, source_rank, num_channels);
        int stride = channel_stride(mask, source_rank, num_channels);

        cursors[slot] = 0;
        if (base > 0) {
            for (int channel = 0; channel < num_channels; ++channel) {
                atomicAdd(channel_counts + channel, base);
                uint32_t pending = mask;
                while (pending != 0) {
                    int dst_rank = __ffs(pending) - 1;
                    atomicAdd(rank_channel_prefix + dst_rank * num_channels + channel,
                              base);
                    pending &= pending - 1;
                }
                if (rdma_channel_prefix != nullptr) {
                    for (int node = 0; node < num_rdma_ranks; ++node) {
                        uint32_t local_mask =
                            (mask >> (node * NUM_MAX_NVL_PEERS)) & 0xffu;
                        if (local_mask != 0)
                            atomicAdd(rdma_channel_prefix + node * num_channels + channel,
                                      base);
                    }
                }
            }
        }

        for (int extra = 0; extra < remainder; ++extra) {
            int channel = (start + extra * stride) % num_channels;
            atomicAdd(channel_counts + channel, 1);
            uint32_t pending = mask;
            while (pending != 0) {
                int dst_rank = __ffs(pending) - 1;
                atomicAdd(rank_channel_prefix + dst_rank * num_channels + channel, 1);
                pending &= pending - 1;
            }
            if (rdma_channel_prefix != nullptr) {
                for (int node = 0; node < num_rdma_ranks; ++node) {
                    uint32_t local_mask =
                        (mask >> (node * NUM_MAX_NVL_PEERS)) & 0xffu;
                    if (local_mask != 0)
                        atomicAdd(rdma_channel_prefix + node * num_channels + channel, 1);
                }
            }
        }
    }
    __syncthreads();

    if (thread_idx == 0) {
        channel_offsets[0] = 0;
        for (int channel = 0; channel < num_channels; ++channel)
            channel_offsets[channel + 1] =
                channel_offsets[channel] + channel_counts[channel];
    }
    for (int rank_idx = thread_idx; rank_idx < num_ranks; rank_idx += blockDim.x) {
        int* row = rank_channel_prefix + rank_idx * num_channels;
        for (int channel = 1; channel < num_channels; ++channel)
            row[channel] += row[channel - 1];
    }
    if (rdma_channel_prefix != nullptr) {
        for (int node = thread_idx; node < num_rdma_ranks; node += blockDim.x) {
            int* row = rdma_channel_prefix + node * num_channels;
            for (int channel = 1; channel < num_channels; ++channel)
                row[channel] += row[channel - 1];
        }
    }
}

__global__ void scatter_channel_tokens(const unsigned long long* keys,
                                       int* cursors,
                                       const int* token_slots,
                                       int* channel_write,
                                       const int* channel_offsets,
                                       int* channel_token_indices,
                                       int num_tokens,
                                       int num_channels,
                                       int source_rank) {
    int token_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (token_idx >= num_tokens)
        return;

    int slot = token_slots[token_idx];
    uint32_t mask = static_cast<uint32_t>(keys[slot] - 1ull);
    int ordinal = atomicAdd(cursors + slot, 1);
    int start = channel_start(mask, source_rank, num_channels);
    int stride = channel_stride(mask, source_rank, num_channels);
    int channel = (start + (ordinal % num_channels) * stride) % num_channels;
    int local_offset = atomicAdd(channel_write + channel, 1);
    channel_token_indices[channel_offsets[channel] + local_offset] = token_idx;
}

}  // namespace

void build(const bool* is_token_in_rank,
           int* channel_offsets,
           int* channel_token_indices,
           int* rank_channel_prefix,
           int* rdma_channel_prefix,
           void* workspace,
           int num_tokens,
           int num_ranks,
           int num_channels,
           int source_rank,
           cudaStream_t stream) {
    EP_HOST_ASSERT(num_ranks > 0 and num_ranks <= kMaxScheduledRanks);
    EP_HOST_ASSERT(num_channels > 0 and num_channels <= kMaxScheduledChannels);

    if (num_tokens == 0) {
        CUDA_CHECK(cudaMemsetAsync(channel_offsets, 0,
                                   static_cast<size_t>(num_channels + 1) * sizeof(int),
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(rank_channel_prefix, 0,
                                   static_cast<size_t>(num_ranks) * num_channels * sizeof(int),
                                   stream));
        if (rdma_channel_prefix != nullptr) {
            int num_rdma_ranks =
                (num_ranks + NUM_MAX_NVL_PEERS - 1) / NUM_MAX_NVL_PEERS;
            CUDA_CHECK(cudaMemsetAsync(rdma_channel_prefix, 0,
                                       static_cast<size_t>(num_rdma_ranks) *
                                           num_channels * sizeof(int),
                                       stream));
        }
        return;
    }

    int table_size = 2;
    while (table_size < num_tokens * 2)
        table_size <<= 1;
    auto layout = get_workspace_layout(workspace, table_size, num_tokens,
                                       num_channels);
    EP_HOST_ASSERT(layout.num_bytes <= NUM_WORKSPACE_BYTES);
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, layout.num_bytes, stream));

    int num_blocks = (num_tokens + kThreads - 1) / kThreads;
    collect_destination_masks<<<num_blocks, kThreads, 0, stream>>>(
        is_token_in_rank, layout.keys, layout.counts, layout.unique_slots,
        layout.token_slots, layout.unique_count, table_size, num_tokens, num_ranks);
    CUDA_CHECK(cudaGetLastError());

    build_channel_metadata<<<1, kThreads, 0, stream>>>(
        layout.keys, layout.counts, layout.cursors, layout.unique_slots,
        layout.unique_count, layout.channel_write, channel_offsets,
        rank_channel_prefix, rdma_channel_prefix, num_ranks, num_channels,
        source_rank);
    CUDA_CHECK(cudaGetLastError());

    scatter_channel_tokens<<<num_blocks, kThreads, 0, stream>>>(
        layout.keys, layout.cursors, layout.token_slots, layout.channel_write,
        channel_offsets, channel_token_indices, num_tokens, num_channels,
        source_rank);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace channel_schedule

}  // namespace deep_ep
