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

#include "rle_cuda.h"

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/tuple.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include "cuda_helpers.h"
#include "thrust_custom_alloc.cuh"

#define TILE (1 << 26)
#define TEMP_BLOCKS_COMPACT 1500
#define GROW_FACTOR 1.25f

struct is_valid {

    __host__ is_valid(std::uint32_t block_size, std::uint32_t* block_sizes)
        : block_size_(block_size), block_sizes_(block_sizes) {}

    __host__ __device__ bool operator()(
        thrust::tuple<size_t, std::uint8_t> t) {

        const size_t block = thrust::get<0>(t) / block_size_;
        const size_t idx = thrust::get<0>(t) % block_size_;
        
        return !(idx < block_sizes_[block]);
    }

    std::uint32_t block_size_;
    const std::uint32_t* __restrict__ block_sizes_;
};

struct is_run {

    __host__ is_run(size_t total_size, std::uint8_t* in) 
        : total_size_(total_size), in_(in) {}

    __host__ __device__ bool operator()(
        thrust::tuple<size_t, std::uint8_t> t) {

        const size_t idx = thrust::get<0>(t);
        const std::uint8_t val = thrust::get<1>(t);

        auto is_run_candidate = [&](const size_t &i) -> bool {
            
            if((in_[i - 5] != in_[i - 4])
                && (in_[i - 4] == in_[i - 3])
                && (in_[i - 3] == in_[i - 2])
                && (in_[i - 2] == in_[i - 1])) return true;

            return false;
        };

        auto preceded_by_run = [&](const size_t &i) -> bool {

            if((in_[i - 9] == in_[i - 8])
                && (in_[i - 8] == in_[i - 7])
                && (in_[i - 7] == in_[i - 6])
                && (in_[i - 5] == in_[i - 4])
                && (in_[i - 4] == in_[i - 3])
                && (in_[i - 3] == in_[i - 2])
                && (in_[i - 2] == in_[i - 1])) return true;

            return false;
        };

        if(idx < 4) return false;
        
        if(idx == 4) return (in_[idx - 4] == in_[idx - 3])
                && (in_[idx - 3] == in_[idx - 2])
                && (in_[idx - 2] == in_[idx - 1]);
        
        if((4 < idx) && (idx < 8)) {

            if(!is_run_candidate(idx)) return false;

            return true;
        }

        if(idx == 8) {

            if((in_[idx - 8] == in_[idx - 7])
                && (in_[idx - 7] == in_[idx - 6])
                && (in_[idx - 6] == in_[idx - 5])) return false;
            
            return (in_[idx - 4] == in_[idx - 3])
                && (in_[idx - 3] == in_[idx - 2])
                && (in_[idx - 2] == in_[idx - 1]);
        }

        if(preceded_by_run(idx)) return true;

        size_t j = 0;

        while(is_run_candidate(idx - j)) j += 4;

        if((j >> 2) % 2 == 0) return false;

        return true;
    }

    const size_t total_size_;
    const std::uint8_t* __restrict__ in_;
};

struct value_only : public thrust::unary_function<
    thrust::tuple<size_t, std::uint8_t>, int> {
    
    __host__ __device__ int operator()(
        thrust::tuple<size_t, std::uint8_t> t) const {
        
        return (int) (thrust::get<1>(t)) - 1;
    }
};

struct strip_fst : public thrust::unary_function<
    thrust::tuple<size_t, std::uint8_t>, std::uint8_t> {
    
    __host__ __device__
    std::uint8_t operator()(thrust::tuple<size_t, std::uint8_t> t) const {
        
        return thrust::get<1>(t);
    }
};

using int32Allocator = ThrustCudaMallocAsyncAllocator<std::uint32_t>;

// modified Thrust example from
// https://github.com/NVIDIA/cccl/blob/main/thrust/examples/expand.cu

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator>
size_t expand(InputIterator1 first1,
        InputIterator1 last1,
        InputIterator2 first2,
        OutputIterator output,
        thrust::device_vector<std::uint32_t, int32Allocator> &output_offsets,
        thrust::device_vector<std::uint32_t, int32Allocator> &output_indices,
        const size_t output_size,
        cudaStream_t &stream) {
    
    typedef typename thrust::iterator_difference<InputIterator1>::type
        difference_type;

    difference_type input_size  = thrust::distance(first1, last1);

    // scan the counts to obtain output offsets for each input element
    thrust::exclusive_scan(
        thrust::cuda::par_nosync(int32Allocator(stream)).on(stream),
        first1, last1, output_offsets.begin());

    // scatter the nonzero counts into their corresponding output positions
    while(output_size > output_indices.size())
        output_indices.resize(
            (std::uint32_t) (((float) output_indices.size()) * GROW_FACTOR));

    thrust::fill(thrust::cuda::par_nosync(int32Allocator(stream)).on(stream),
        output_indices.begin(), output_indices.end(), 0);

    thrust::scatter_if(thrust::cuda::par_nosync.on(stream),
    thrust::counting_iterator<difference_type>(0),
    thrust::counting_iterator<difference_type>(input_size),
        output_offsets.begin(),
        first1,
        output_indices.begin());

    // compute max-scan over the output indices, filling in the holes
    thrust::inclusive_scan(thrust::cuda::par_nosync.on(stream),
        output_indices.begin(),
        output_indices.begin() + output_size,
        output_indices.begin(),
        thrust::maximum<difference_type>());

    // gather input values according to index array
    //  (output = first2[output_indices])
    thrust::gather(thrust::cuda::par_nosync.on(stream),
        output_indices.begin(),
        output_indices.begin() + output_size,
        first2,
        output);

    // return output + output_size
    thrust::advance(output, output_size);
    return output_size;
}

__global__ void insert_runs(
    const thrust::tuple<size_t, std::uint8_t>* __restrict__ runs,
    std::uint32_t* counts,
    const size_t offset,
    const size_t num_runs,
    const size_t tile_offset) {
    
    const size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(gid < num_runs)
        counts[thrust::get<0>(runs[offset + gid]) - tile_offset]
            = thrust::get<1>(runs[offset + gid]);
}

__global__ void fill_gaps(
    const thrust::tuple<size_t, std::uint8_t>* __restrict__ runs,
    std::uint8_t* data,
    const size_t num_runs) {
    
    const size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint32_t run_pos = thrust::get<0>(runs[gid]);
    
    if(gid < num_runs)
        data[run_pos] = data[run_pos - 1];
}

struct geq {
    
    __host__ geq(size_t val) : val_(val)  {}
    
    __host__ __device__ bool operator()(
        const thrust::tuple<size_t, std::uint8_t> &t) {
        
        return thrust::get<0>(t) >= val_;
    }
    
    const size_t val_;
};

void rle_cuda::decode(std::uint8_t* in,
    std::uint8_t* &out,
    const bz2stream &bz2s,
    device_metadata &md,
    size_t &out_size,
    cudaStream_t &stream) {

    const size_t num_blocks = bz2s.compressed_data.size();
    const size_t block_size = bz2s.block_size;

    thrust::device_vector<std::uint32_t, int32Allocator> output_offsets(
        TILE, 0, int32Allocator(stream));
    thrust::device_vector<std::uint32_t, int32Allocator> output_indices(
        TILE, 0,
        int32Allocator(stream));

    thrust::device_ptr<std::uint8_t> in_ptr = thrust::device_pointer_cast(in);

    // remove block tails
    thrust::counting_iterator<size_t> index_itr(0);
    
    auto iter = thrust::make_zip_iterator(index_itr, in_ptr);
    
    using uint8_allocator = ThrustCudaMallocAsyncAllocator<std::uint8_t>;
    thrust::device_vector<std::uint8_t, uint8_allocator> temp_compact(
        block_size * TEMP_BLOCKS_COMPACT, uint8_allocator(stream));
        
    auto out_iter = make_transform_output_iterator(temp_compact.begin(),
        strip_fst());

    size_t buffer_offset = 0;

    auto compact_segment = [&](std::uint32_t offset,
        std::uint32_t segment_size) {
        
        auto end_buffer = thrust::remove_copy_if(
            thrust::cuda::par_nosync(uint8_allocator(stream)).on(stream),
            iter + (offset * block_size),
            iter + ((offset + segment_size) * block_size),
            out_iter,
            is_valid(block_size, md.output_size));
            
        thrust::copy(thrust::cuda::par_nosync.on(stream),
            temp_compact.begin(),
            temp_compact.begin() + thrust::distance(out_iter, end_buffer),
            in_ptr + buffer_offset);
        
        buffer_offset += thrust::distance(out_iter, end_buffer);
    };

    if(TEMP_BLOCKS_COMPACT <= num_blocks) {

        size_t offset = 0;
        
        for(; offset < num_blocks - TEMP_BLOCKS_COMPACT;
            offset += TEMP_BLOCKS_COMPACT)
            compact_segment(offset, TEMP_BLOCKS_COMPACT);

        if(num_blocks - offset > 0)
            compact_segment(offset, num_blocks - offset);
    }

    else compact_segment(0, num_blocks);

    const size_t total_size_compacted = buffer_offset;

    // locate runs
    thrust::counting_iterator<size_t> index_itr_clean(0);
    
    auto iter_clean = thrust::make_zip_iterator(index_itr_clean, in_ptr);

    const size_t num_runs = thrust::count_if(
        thrust::cuda::par_nosync.on(stream),
        iter_clean,
        iter_clean + total_size_compacted, is_run(total_size_compacted, in));

    using tuple_allocator = ThrustCudaMallocAsyncAllocator<
        thrust::tuple<size_t, std::uint8_t>>;
    thrust::device_vector<
        thrust::tuple<size_t, std::uint8_t>, tuple_allocator> runs(
            num_runs, tuple_allocator(stream));

    thrust::copy_if(thrust::cuda::par_nosync(
        tuple_allocator(stream)).on(stream),
        iter_clean,
        iter_clean + total_size_compacted, runs.begin(),
        is_run(total_size_compacted, in));

    std::int64_t total_run_size = thrust::reduce(
        thrust::cuda::par_nosync.on(stream),
        thrust::make_transform_iterator(runs.begin(), value_only()),
        thrust::make_transform_iterator(runs.end(), value_only()));
        
    const size_t output_size = total_size_compacted + total_run_size;
    
    // alloc buffer for final output
    cudaMallocAsync(&out, output_size, stream);
    CUERR
    
    // replace run-lengths with respective symbols
    fill_gaps<<<SDIV(num_runs, 128), 128, 0, stream>>>(
        thrust::raw_pointer_cast(runs.data()),
        in,
        num_runs);
    CUERR

    // alloc counts
    using int32Allocator = ThrustCudaMallocAsyncAllocator<std::uint32_t>;
    thrust::device_vector<std::uint32_t, int32Allocator> counts(
        TILE, int32Allocator(stream));
    
    auto lower_bound = runs.begin();
    auto max_value = runs.begin();

    size_t out_idx = 0;
    
    auto decode_segment = [&](std::uint32_t offset,
        std::uint32_t segment_size) {

        thrust::fill(thrust::cuda::par_nosync(
            int32Allocator(stream)).on(stream),
            counts.begin(), counts.end(), 1);

        // update bounds
        lower_bound = max_value;
        
        max_value = thrust::find_if(thrust::cuda::par_nosync.on(stream),
            lower_bound, runs.end(), geq(offset + segment_size));

        const size_t run_range = max_value - lower_bound;

        // update counts with runs
        insert_runs<<<SDIV(run_range, 128), 128, 0, stream>>>(
            thrust::raw_pointer_cast(runs.data()),
            thrust::raw_pointer_cast(counts.data()),
            lower_bound - runs.begin(),
            run_range,
            offset);
        CUERR

        const size_t output_size = thrust::reduce(
            thrust::cuda::par_nosync.on(stream),
            counts.begin(), counts.begin() + segment_size);

        out_idx += expand(counts.begin(), counts.begin() + segment_size,
            thrust::device_pointer_cast(in) + offset,
            thrust::device_pointer_cast(out) + out_idx,
            output_offsets, output_indices, output_size,
            stream);
    };

    if(TILE <= total_size_compacted) {

        size_t offset = 0;
        
        for(; offset < total_size_compacted - TILE; offset += TILE)
            decode_segment(offset, TILE);

        if(total_size_compacted - offset > 0)
            decode_segment(offset, total_size_compacted - offset);
    }

    else decode_segment(0, total_size_compacted);

    out_size = out_idx;
}
