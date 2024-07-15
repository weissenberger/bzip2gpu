// MIT License

// Copyright (c) 2024 André Weißenberger

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "ibwt_cuda.h"

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>
#include <thrust/device_vector.h>

#include "cuda_helpers.h"
#include "helpers.h"

#define SEGMENT_SIZE 800
#define LIST_DISTANCE 32
#define BLOCK_SIZE 128

struct TimesN {

    __host__ __device__ __forceinline__ TimesN(int n) {N = n;};

    __host__ __device__ __forceinline__
    int operator()(const int &a) const {

        return a * N;
    }

    int N;
};

struct ModN {

    __host__ __device__ __forceinline__ ModN(int n) {N = n;};

    __host__ __device__ __forceinline__
    int operator()(const int &a) const {

        return a % N;
    }

    int N;
};

// mask off unused values at end of block

__global__ void mask_end(
    std::uint8_t* out,
    const std::uint32_t* __restrict__ output_sizes,
    const size_t block_size,
    const size_t num_blocks) {

    for(size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        gid < num_blocks * block_size; gid += blockDim.x * gridDim.x) {

        const size_t block_id = gid / block_size;
        const size_t local_idx = gid % block_size;

        const std::uint32_t bls = output_sizes[block_id];

        if(local_idx >= bls) out[gid] = 0xFF;
    }
}

// kernels required for parallel list ranking

__global__ void helman_jaja_local(
    const std::uint32_t* __restrict__ perm,
    const std::uint32_t* __restrict__ idxs,
    std::uint32_t* ranks,
    std::uint32_t* sublist_ids,
    std::uint32_t* reduced_lists,
    std::uint32_t* reduced_ranks,
    const std::uint32_t* __restrict__ output_sizes,
    const std::uint32_t threads_per_block,
    const std::uint32_t list_distance,
    const size_t block_size,
    const size_t num_blocks) {

    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tid = gid % threads_per_block;
    const size_t block_id = gid / threads_per_block;
    const size_t block_offset = block_size * block_id;

    const std::uint32_t output_size = output_sizes[block_id];

    if(block_id < num_blocks) {

        const size_t block_idx = idxs[block_id];
        const size_t idx_tid = block_idx / list_distance;
        const size_t valid_idx = idx_tid * list_distance;
        size_t head = tid * list_distance;

        if(tid == idx_tid) head = block_idx;

        if(head < output_size) {

            std::uint32_t rank = 0;

            std::uint32_t next = perm[block_offset + head];
            sublist_ids[block_offset + next] = tid;
            ranks[block_offset + next] = rank;

            while(((next & (list_distance - 1)) || (next == valid_idx))
                && (next != block_idx)) {

                ++rank;

                next = perm[block_offset + next];

                sublist_ids[block_offset + next] = tid;
                ranks[block_offset + next] = rank;
            }

            reduced_lists[block_id * threads_per_block + tid] = next;
            reduced_ranks[block_id * threads_per_block + tid] = rank;
        }
    }
}

__global__ void helman_jaja_reduce(
    const std::uint32_t* __restrict__ reduced_lists,
    std::uint32_t* reduced_ranks,
    const std::uint32_t* __restrict__ idxs,
    const std::uint32_t* __restrict__ output_sizes,
    const std::uint32_t threads_per_block,
    const std::uint32_t list_distance,
    const size_t num_blocks) {

    for(size_t gid = blockIdx.x * blockDim.x + threadIdx.x; gid < num_blocks;
        gid += blockDim.x * gridDim.x) {

        const size_t block_offset = threads_per_block * gid;
        const size_t block_idx = idxs[gid];
        const std::uint32_t output_size = output_sizes[gid];

        std::uint32_t next = block_idx / list_distance;

        if(next < output_size) {

            std::uint32_t rank = reduced_ranks[block_offset + next] + 1;

            reduced_ranks[block_offset + next] = 0;

            for(size_t i = 1; i < SDIV(output_size, list_distance); ++i) {

                next = reduced_lists[block_offset + next] / list_distance;

                const std::uint32_t next_rank
                    = reduced_ranks[block_offset + next] + 1;

                reduced_ranks[block_offset + next] = rank;
                rank += next_rank;
            }
        }
    }
}

__global__ void add_permute(
    const std::uint8_t* __restrict__ in,
    std::uint8_t* out,
    const std::uint32_t* __restrict__ sublist_ids,
    const std::uint32_t* __restrict__ ranks,
    const std::uint32_t* __restrict__ reduced_ranks,
    const std::uint32_t* __restrict__ output_sizes,
    const size_t block_size,
    const std::uint32_t threads_per_block,
    const size_t n) {
    
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {
        
        const size_t block_id = i / block_size;

        const size_t offset = block_id * block_size;
        const size_t sublist = block_id * threads_per_block + sublist_ids[i];
        const std::uint32_t output_size = output_sizes[block_id];
        const size_t out_idx = ranks[i] + reduced_ranks[sublist];

        if(out_idx < output_size)
            out[offset + out_idx] = in[i];
    }
}

void ibwt_cuda::batch_iBWT(
    std::uint8_t* dev_bwt,
    std::uint32_t* dev_idxs,
    const size_t num_blocks,
    const size_t block_size,
    std::uint32_t* output_sizes,
    cudaStream_t &stream) {

    const size_t segment_size = SEGMENT_SIZE;
    const size_t segment_size_bytes = segment_size * block_size;

    const std::uint32_t list_distance = LIST_DISTANCE;
    const std::uint32_t threads_per_block = SDIV(block_size, list_distance);

    std::uint8_t* bwt_copy;
    cudaMallocAsync(
        &bwt_copy, segment_size_bytes * sizeof(std::uint8_t), stream); CUERR

    std::uint8_t* bwt_copy_alt;
    cudaMallocAsync(
        &bwt_copy_alt, segment_size_bytes * sizeof(std::uint8_t), stream); CUERR

    std::uint32_t* perm;
    cudaMallocAsync(
        &perm, segment_size_bytes * sizeof(std::uint32_t), stream); CUERR

    std::uint32_t* perm_alt;
    cudaMallocAsync(
        &perm_alt, segment_size_bytes * sizeof(std::uint32_t), stream); CUERR

    std::uint32_t* sublist_ids;
    cudaMallocAsync(
        &sublist_ids, segment_size_bytes * sizeof(std::uint32_t), stream); CUERR

    std::uint32_t* reduced_lists;
    cudaMallocAsync(
        &reduced_lists,
        threads_per_block * segment_size * sizeof(std::uint32_t), stream); CUERR

    std::uint32_t* reduced_ranks;
    cudaMallocAsync(
        &reduced_ranks,
        threads_per_block * segment_size * sizeof(std::uint32_t), stream); CUERR

    cub::DoubleBuffer<std::uint8_t> keys(bwt_copy, bwt_copy_alt);
    cub::DoubleBuffer<std::uint32_t> values(perm, perm_alt);

    ModN mod(block_size);

    TimesN times(block_size);
    cub::CountingInputIterator<int> count(0);
    cub::TransformInputIterator<int, TimesN, cub::CountingInputIterator<int>>
        block_size_stride(count, times);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceSegmentedSort::StableSortPairs(
        d_temp_storage,
        temp_storage_bytes,
        keys,
        values,
        segment_size_bytes,
        segment_size,
        block_size_stride,
        block_size_stride + 1,
        stream); CUERR

    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream); CUERR

    auto decode_segment = [&](
        std::uint32_t offset,
        std::uint32_t segment_size) {

        const std::uint32_t segment_size_bytes = segment_size * block_size;
        const std::uint32_t segment_begin = offset * block_size;
        const std::uint32_t segment_end = segment_begin + segment_size_bytes;

        thrust::copy(thrust::cuda::par_nosync.on(stream),
            dev_bwt + segment_begin,
            dev_bwt + segment_end, bwt_copy);
        CUERR
        thrust::tabulate(thrust::cuda::par_nosync.on(stream),
            perm, perm + segment_size_bytes, mod);
        CUERR

        mask_end<<<((segment_size * block_size) / 128) + 1, 128, 0, stream>>>(
        bwt_copy,
        output_sizes + offset,
        block_size,
        segment_size); CUERR

        cub::DeviceSegmentedSort::StableSortPairs(
            d_temp_storage,
            temp_storage_bytes,
            keys,
            values,
            segment_size_bytes,
            segment_size,
            block_size_stride,
            block_size_stride + 1,
            stream); CUERR

        helman_jaja_local<<<((segment_size * threads_per_block) / BLOCK_SIZE) + 1,
            BLOCK_SIZE, 0, stream>>>(
            perm,
            dev_idxs + offset,
            perm_alt,
            sublist_ids,
            reduced_lists,
            reduced_ranks,
            output_sizes + offset,
            threads_per_block,
            list_distance,
            block_size,
            segment_size); CUERR

        helman_jaja_reduce<<<(segment_size / BLOCK_SIZE) + 1,
            BLOCK_SIZE, 0, stream>>>(
            reduced_lists,
            reduced_ranks,
            dev_idxs + offset,
            output_sizes + offset,
            threads_per_block,
            list_distance,
            segment_size); CUERR

        add_permute<<<(segment_size_bytes / BLOCK_SIZE) + 1,
            BLOCK_SIZE, 0, stream>>>(
            dev_bwt + segment_begin,
            bwt_copy,
            sublist_ids,
            perm_alt,
            reduced_ranks,
            output_sizes + offset,
            block_size,
            threads_per_block,
            segment_size_bytes); CUERR

        thrust::copy(thrust::cuda::par_nosync.on(stream),
            bwt_copy, bwt_copy + segment_size_bytes,
            dev_bwt + segment_begin);
        CUERR
    };

    if(segment_size <= num_blocks) {

        std::uint32_t offset = 0;
        
        for(; offset < num_blocks - segment_size;
            offset += segment_size)
            decode_segment(offset, segment_size);

        if(num_blocks - offset > 0)
            decode_segment(offset, num_blocks - offset);
    }

    else decode_segment(0, num_blocks);

    cudaFreeAsync(d_temp_storage, stream); CUERR
    cudaFreeAsync(perm, stream); CUERR
    cudaFreeAsync(perm_alt, stream); CUERR
    cudaFreeAsync(sublist_ids, stream); CUERR
    cudaFreeAsync(bwt_copy, stream); CUERR
    cudaFreeAsync(bwt_copy_alt, stream); CUERR
    cudaFreeAsync(reduced_lists, stream); CUERR
    cudaFreeAsync(reduced_ranks, stream); CUERR
}
