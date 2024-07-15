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

#include "imtf_cuda.h"

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include "helpers.h"
#include "cuda_helpers.h"

#define BLOCK_SIZE 64
#define PRIVATE_PERM_SIZE 10000

__global__ void write_neutral(
    std::uint8_t* perms,
    const size_t block_size,
    const size_t shift,
    const size_t n) {

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {

        const size_t block_offset = i * block_size;
        const size_t last_perm = block_offset + shift - 1;
        const size_t out = last_perm * 256;

        for(size_t j = 0; j < 256; ++j) perms[out + j] = j;
    }
}

__device__ __forceinline__ void concat(
    std::uint8_t* perms,
    std::uint8_t* aux,
    const size_t a,
    const size_t b) {

    const size_t block_a = a * 256;
    const size_t block_b = b * 256;

    for(size_t i = 0; i < 256; ++i)
        aux[block_b + i] = perms[block_a + perms[block_b + i]];

    for(size_t i = 0; i < 256; ++i)
        perms[block_b + i] = aux[block_b + i];
}

__device__ __forceinline__ void swap_concat(
    std::uint8_t* perms,
    std::uint8_t* aux,
    const size_t a,
    const size_t b) {

    const size_t block_a = a * 256;
    const size_t block_b = b * 256;

    for(size_t i = 0; i < 256; ++i)
        aux[block_b + i] = perms[block_b + perms[block_a + i]];

    for(size_t i = 0; i < 256; ++i)
        perms[block_a + i] = perms[block_b + i];

    for(size_t i = 0; i < 256; ++i)
        perms[block_b + i] = aux[block_b + i];
}

__global__ void link_tiles(
    std::uint8_t* perms,
    std::uint8_t* aux,
    std::uint8_t* carry,
    const size_t block_size,
    const size_t shift,
    const size_t n) {

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {

        const size_t block_offset = i * block_size;
        const size_t last_perm = block_offset + shift - 1;
        const size_t in = last_perm * 256;

        for(size_t j = 0; j < 256; ++j) carry[i * 256 + j] = perms[in + j];

        concat(perms, aux, last_perm, last_perm + 1);
    }
}

__global__ void paste_perm(
    std::uint8_t* perms,
    std::uint8_t* carry,
    const size_t block_size,
    const size_t shift,
    const size_t n) {

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {

        const size_t block_offset = i * block_size;
        const size_t first_perm = block_offset + shift;
        const size_t in = first_perm * 256;

        for(size_t j = 0; j < 256; ++j) perms[in + j] = carry[i * 256 + j];
    }
}

__global__ void concat_perms_upsweep(
    std::uint8_t* perms,
    std::uint8_t* aux,
    const size_t step,
    const size_t offset,
    const size_t block_size,
    const size_t shift,
    const size_t n) {

    const size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t thid = gid % step;
    const size_t block_id = gid / step;
    const size_t block_offset = block_id * block_size;

    if(block_id < n) {

        const size_t a = block_offset + shift + offset * (2 * thid + 1) - 1;
        const size_t b = block_offset + shift + offset * (2 * thid + 2) - 1;

        concat(perms, aux, a, b);
    }
}

__global__ void concat_perms_downsweep(
    std::uint8_t* perms,
    std::uint8_t* aux,
    const size_t step,
    const size_t offset,
    const size_t block_size,
    const size_t shift,
    const size_t n) {

    const size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t thid = gid % step;
    const size_t block_id = gid / step;
    const size_t block_offset = block_id * block_size;

    if(block_id < n) {

        const size_t a = block_offset + shift + offset * (2 * thid + 1) - 1;
        const size_t b = block_offset + shift + offset * (2 * thid + 2) - 1;

        swap_concat(perms, aux, a, b);
    }
}

__global__ void concat_stacks(
    const std::uint8_t* __restrict__ perms,
    const std::uint8_t* __restrict__ stacks,
    std::uint8_t* aux,
    const size_t perms_per_block,
    const size_t n) {

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {

        const size_t block = i / perms_per_block;
        const size_t block_perm = i * 256;
        const size_t block_stack = block * 256;

        for(size_t j = 0; j < 256; ++j)
            aux[block_perm + j] = stacks[block_stack + perms[block_perm + j]];
    }
}

// referred to as Algorithm 4 in article
__global__ void private_perm(
    const std::uint8_t* __restrict__ in,
    std::uint8_t* perms,
    const size_t block_size,
    const size_t n) {

    __shared__ std::uint8_t private_block[BLOCK_SIZE][256];

    auto popp_shift = [&](std::uint8_t pos) {

        const std::uint8_t sym = private_block[threadIdx.x][pos];

        for(size_t i = pos; i > 0; --i)
            private_block[threadIdx.x][i] = private_block[threadIdx.x][i - 1];

        private_block[threadIdx.x][0] = sym;
    };

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {

        const size_t block = i * block_size;

        for(size_t j = 0; j < 256; ++j)
            private_block[threadIdx.x][j] = j;

        for(size_t j = 0; j < block_size; ++j)
            popp_shift(in[block + j]);

        __syncthreads();

        cub::StoreDirectBlockedVectorized(
            i, perms, private_block[threadIdx.x]);
    }
}

// referred to as Algorithm 5 in article
__global__ void private_perm_2(
    const std::uint8_t* __restrict__ in,
    std::uint8_t* __restrict__ perms,
    const size_t block_size,
    const size_t n) {

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {

        const size_t perm = blockIdx.x * 256;
        const size_t block = blockIdx.x * block_size;
        std::uint8_t idx = threadIdx.x;

        for(size_t j = 0; j < block_size; ++j) {

            const std::uint8_t next = in[block + j];

            if(idx < next) ++idx;
            else if(idx == next) idx = 0;
        }

        perms[perm + idx] = threadIdx.x;
    }
}

__global__ void private_perm_write(
    std::uint8_t* in,
    std::uint8_t* perms,
    const size_t block_size,
    const size_t n) {

    __shared__ std::uint8_t private_block[BLOCK_SIZE][256];

    auto popp_shift = [&](std::uint8_t pos, std::uint8_t &sym) {

        sym = private_block[threadIdx.x][pos];

        for(size_t i = pos; i > 0; --i)
            private_block[threadIdx.x][i] = private_block[threadIdx.x][i - 1];

        private_block[threadIdx.x][0] = sym;
    };

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {

        cub::LoadDirectBlockedVectorized(
            i, perms, private_block[threadIdx.x]);

        __syncthreads();

        const size_t block = i * block_size;

        for(size_t j = 0; j < block_size; ++j) {

            std::uint8_t sym;

            popp_shift(in[block + j], sym);
            in[block + j] = sym;
        }
    }
}

__global__ void private_perm_write_2(
    std::uint8_t* in,
    const std::uint8_t* __restrict__ perms,
    const size_t block_size,
    const size_t n) {

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {

        const size_t perm = blockIdx.x * 256;
        const size_t block = blockIdx.x * block_size;
        std::uint8_t idx = threadIdx.x;
        const std::uint8_t sym = perms[perm + idx];

        for(size_t j = 0; j < block_size; ++j) {

            const std::uint8_t next = in[block + j];
            __syncthreads();

            if(idx < next) ++idx;
            
            else if(idx == next) {
                
                idx = 0;
                in[block + j] = sym;
            }
        }
    }
}

void scan_perms(
    std::uint8_t* perms,
    std::uint8_t* perms_aux,
    const size_t private_perms_per_block,
    const size_t num_blocks,
    cudaStream_t &stream) {
    
    // concatenate all permutations using parallel scan (Blelloch)
    
    std::uint8_t* carry;
    cudaMallocAsync(&carry, num_blocks * 256, stream);
    
    std::uint8_t* carry_alt;
    cudaMallocAsync(&carry_alt, num_blocks * 256, stream);

    std::uint8_t* carry_ptr_a = carry;
    std::uint8_t* carry_ptr_b = carry_alt;

    size_t remainder = private_perms_per_block;
    
    // integer log_2
    size_t log = 0;
    size_t size_cpy = private_perms_per_block;
    
    while(size_cpy > 0) {

        ++log;
        size_cpy >>= 1;
    }

    size_t tile_size = 1 << log;
    size_t shift = 0;

    bool paste = false;

    while(remainder > 0) {

        while(tile_size > remainder) tile_size >>= 1;
        remainder -= tile_size;

        size_t offset = 1;

        // upsweep
        for(size_t step = tile_size >> 1; step > 0; step >>= 1) {

            concat_perms_upsweep<<<(step * num_blocks / BLOCK_SIZE) + 1,
                BLOCK_SIZE, 0, stream>>>(
                perms,
                perms_aux,
                step,
                offset,
                private_perms_per_block,
                shift,
                num_blocks); CUERR

            offset *= 2;
        }

        // save last permutation and concatenate with first permutation
        // of the next tile
        if(remainder > 0) {

            link_tiles<<<(num_blocks / BLOCK_SIZE) + 1,
            BLOCK_SIZE, 0, stream>>>(
                perms,
                perms_aux,
                carry_ptr_a,
                private_perms_per_block,
                shift + tile_size,
                num_blocks); CUERR
        }

        // set last perm in each block to neutral (0, 1, 2,..., 255)
        write_neutral<<<(num_blocks / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
            perms,
            private_perms_per_block,
            shift + tile_size,
            num_blocks); CUERR

        // downsweep
        for(size_t step = 1; step < tile_size; step *= 2) {

            offset >>= 1;

            concat_perms_downsweep<<<(step * num_blocks / BLOCK_SIZE) + 1,
                BLOCK_SIZE, 0, stream>>>(
                perms,
                perms_aux,
                step,
                offset,
                private_perms_per_block,
                shift,
                num_blocks); CUERR
        }

        // paste copied perm to next tile
        if(paste) {

            paste_perm<<<(num_blocks / BLOCK_SIZE) + 1,
            BLOCK_SIZE, 0, stream>>>(
                perms,
                carry_ptr_b,
                private_perms_per_block,
                shift,
                num_blocks); CUERR
        }

        shift += tile_size;
        paste = true;
        
        // swap pointers for temporary storage
        if(carry_ptr_a == carry) {
            
            carry_ptr_a = carry_alt;
            carry_ptr_b = carry;
        }

        else {
            
            carry_ptr_a = carry;
            carry_ptr_b = carry_alt;
        }
    }

    cudaFreeAsync(carry, stream);
    cudaFreeAsync(carry_alt, stream);
}

void imtf_cuda::batch_iMTF(
    std::uint8_t* dev_mtf,
    std::uint8_t* dev_mtf_stacks,
    const size_t num_blocks,
    const size_t block_size,
    cudaStream_t &stream) {
    
    size_t total_size = num_blocks * block_size;
    size_t private_blocks = total_size / PRIVATE_PERM_SIZE;
    size_t private_perms_per_block = block_size / PRIVATE_PERM_SIZE;
    
    std::uint8_t* perms;
    cudaMallocAsync(&perms, private_blocks * 256, stream);

    std::uint8_t* perms_aux;
    cudaMallocAsync(&perms_aux, private_blocks * 256, stream);

    // use Algorithm 5
    /* private_perm_2<<<private_blocks, 256,
        0, stream>>>(
        dev_mtf, perms, private_perm_size, private_blocks * 256); CUERR */

    // use Algorithm 4
    private_perm<<<(private_blocks / BLOCK_SIZE) + 1, BLOCK_SIZE,
        0, stream>>>(
        dev_mtf, perms, PRIVATE_PERM_SIZE, private_blocks); CUERR

    scan_perms(perms, perms_aux, private_perms_per_block, num_blocks, stream);

    concat_stacks<<<(private_blocks / BLOCK_SIZE) + 1, BLOCK_SIZE,
        0, stream>>>(
        perms,
        dev_mtf_stacks,
        perms_aux,
        private_perms_per_block,
        private_blocks); CUERR

    // use Algorithm 4
    private_perm_write<<<(private_blocks / BLOCK_SIZE) + 1, BLOCK_SIZE,
        0, stream>>>(
        dev_mtf, perms_aux, PRIVATE_PERM_SIZE, private_blocks); CUERR

    // use Algorithm 5
    /* private_perm_write_2<<<private_blocks, 256,
        0, stream>>>(
        dev_mtf, perms_aux, PRIVATE_PERM_SIZE, 
        private_blocks * 256); CUERR */

    cudaFreeAsync(perms, stream);
    cudaFreeAsync(perms_aux, stream);
}
