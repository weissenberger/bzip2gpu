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

#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda.h>

#include "stream.cuh"

#include "cuda_decoder.h"

#include "cuda_helpers.h"

#include "helpers.h"

#define SECOND_STAGE_SIZE 1320

void decoder_cuda::alloc_metadata(
    device_metadata &md,
    const bz2stream &bz2s,
    cudaStream_t &stream) {

    const size_t num_blocks = bz2s.compressed_data.size();
    
    cudaMallocAsync(&md.max_lones,
        6 * num_blocks, stream);
    CUERR

    cudaMallocAsync(&md.table_size,
        num_blocks * sizeof(std::uint32_t), stream);
    CUERR

    cudaMallocAsync(&md.table_position,
        num_blocks * sizeof(std::uint32_t), stream);
    CUERR

    cudaMallocAsync(&md.subtables,
        num_blocks * 6 * 21 * sizeof(bz2subtable_entry), stream);
    CUERR

    cudaMallocAsync(&md.table,
        bz2s.table.size() * sizeof(bz2symbol), stream);
    CUERR

    cudaMallocAsync(&md.output_size,
        num_blocks * sizeof(std::uint32_t), stream);
    CUERR

    cudaMallocAsync(&md.num_selectors,
        num_blocks * sizeof(std::uint32_t), stream);
    CUERR

    cudaMallocAsync(&md.selectors, num_blocks * 32768, stream);
    CUERR

    cudaMallocAsync(&md.stack_size,
        num_blocks * sizeof(std::uint32_t), stream);
    CUERR

    cudaMallocAsync(&md.compressed_data,
        num_blocks * sizeof(location), stream);
    CUERR
    
    cudaMallocAsync(&md.mtf_stacks,
        num_blocks * 256, stream);
    CUERR
    
    cudaMallocAsync(&md.bwt_indices,
        num_blocks * sizeof(std::uint32_t), stream);
    CUERR
}

void decoder_cuda::copy_metadata(
    device_metadata &md,
    bz2stream &bz2s,
    cudaStream_t &stream) {

    const size_t num_blocks = bz2s.compressed_data.size();
    
    cudaMemcpyAsync(md.max_lones, bz2s.max_lones.data(),
        6 * num_blocks, cudaMemcpyHostToDevice, stream);
    CUERR

    cudaMemcpyAsync(md.num_selectors, bz2s.num_selectors.data(),
        num_blocks * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream);
    CUERR

    cudaMemcpyAsync(md.selectors, bz2s.selectors.data(),
        num_blocks * 32768 * sizeof(std::uint8_t),
        cudaMemcpyHostToDevice, stream);
    CUERR

    cudaMemcpyAsync(md.stack_size, bz2s.stack_size.data(),
        num_blocks * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream);
    CUERR

    cudaMemcpyAsync(md.compressed_data, bz2s.compressed_data.data(),
        num_blocks * sizeof(location), cudaMemcpyHostToDevice, stream);
    CUERR

    cudaMemcpyAsync(md.table_size, bz2s.table_size.data(),
        num_blocks * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream);
    CUERR

    cudaMemcpyAsync(md.table_position, bz2s.table_position.data(),
        num_blocks * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream);
    CUERR

    cudaMemcpyAsync(md.subtables, bz2s.subtables.data(),
        num_blocks * 6 * 21 * sizeof(bz2subtable_entry),
        cudaMemcpyHostToDevice, stream);
    CUERR

    cudaMemcpyAsync(md.table, bz2s.table.data(),
        bz2s.table.size() * sizeof(bz2symbol), cudaMemcpyHostToDevice, stream);
    CUERR
    
    cudaMemcpyAsync(md.mtf_stacks, bz2s.mtf_stacks.data(),
        num_blocks * 256, cudaMemcpyHostToDevice, stream);
    CUERR
    
    cudaMemcpyAsync(md.bwt_indices, bz2s.bwt_indices.data(),
        num_blocks * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream);
    CUERR
}

void decoder_cuda::free_metadata(
    const device_metadata &md,
    cudaStream_t &stream) {

    cudaFreeAsync(md.max_lones, stream);
    CUERR

    cudaFreeAsync(md.table_size, stream);
    CUERR

    cudaFreeAsync(md.table_position, stream);
    CUERR

    cudaFreeAsync(md.subtables, stream);
    CUERR

    cudaFreeAsync(md.table, stream);
    CUERR

    cudaFreeAsync(md.output_size, stream);
    CUERR

    cudaFreeAsync(md.num_selectors, stream);
    CUERR

    cudaFreeAsync(md.selectors, stream);
    CUERR

    cudaFreeAsync(md.stack_size, stream);
    CUERR

    cudaFreeAsync(md.compressed_data, stream);
    CUERR
    
    cudaFreeAsync(md.mtf_stacks, stream);
    CUERR
    
    cudaFreeAsync(md.bwt_indices, stream);
    CUERR
}

__global__ void decode_block(
    const std::uint64_t* __restrict__ in, 
    bz2uchar16* __restrict__ out,
    device_metadata md,
    std::uint32_t block_size) {
    
    const std::uint32_t block = blockIdx.x;
    std::uint8_t* out_ = reinterpret_cast<std::uint8_t*>(out);

    // load first-stage table
    __shared__ bz2subtable_entry fst_tab[6 * 21];
    std::uint8_t max_lones[6];
    
    for(std::uint32_t i = 0; i < 6 * 21; ++i)
        fst_tab[i] = md.subtables[(block * 6 * 21) + i];
        
    for(std::uint32_t i = 0; i < 6; ++i)
        max_lones[i] = md.max_lones[(block * 6) + i];

    // load second-stage table
    const std::uint32_t table_size = md.table_size[block];
    const std::uint32_t table_position = md.table_position[block];
    
    extern __shared__ bz2symbol snd_tab[];
    
    for(size_t i = 0; i < table_size; ++i)
        snd_tab[i] = md.table[table_position + i];
        
    // selectors
    const std::uint32_t num_selectors = md.num_selectors[block];
    const std::uint8_t* __restrict__ selectors
        = md.selectors + (block * 32768);
    
    // EOB symbol
    const std::uint16_t eob_sym = md.stack_size[block] + 1;

    // init stream
    cu_stream<std::uint64_t> s(in, md.compressed_data[block], {0,0});
    
    std::uint32_t out_idx = block * block_size;
        
    std::uint32_t run = 0;
    std::uint32_t run_len = 0;
    bool in_run = false;
    const std::uint32_t one = 1;
    
    for(size_t sel_idx = 0; sel_idx < num_selectors; ++sel_idx) {
        
        const std::uint8_t sel = selectors[sel_idx];
        const std::uint8_t ml = max_lones[sel];
        const std::uint32_t tab_offset = sel * 21;
        
        for(size_t i = 0; i < 50; ++i) {
        
            std::uint64_t buffer;
            bz2symbol sym;
            
            auto look_up = [&](bz2symbol &sym,
                std::uint32_t const &tab_offset) {
                
                const std::uint8_t lz = __clzll(~buffer);
                const bz2subtable_entry fst = fst_tab[tab_offset + lz];
                const std::uint32_t rem = fst.remainder ?
                    (buffer << lz) >> (64 - fst.remainder) : 0;
                
                sym = snd_tab[fst.position + rem];
            };
            
            s.read(ml, buffer);
            look_up(sym, tab_offset);
            s.skip(sym.length);
            
            if(sym.symbol < 2) {

                in_run = true;
                run |= (sym.symbol << run_len);
                ++run_len;
            }

            if(sym.symbol > 1 && in_run) {
                
                out_idx += (run | (one << run_len)) - 1;
                run = 0;
                run_len = 0;
                in_run = false;
            }
            
            if(sym.symbol == eob_sym) break;
            
            if(!in_run) out_[out_idx++] = sym.symbol - 1;
        }
    }
    
    md.output_size[block] = out_idx - block * block_size;
}

void decoder_cuda::decode(
    std::uint64_t* in,
    bz2uchar16* out,
    bz2stream &bz2stream,
    device_metadata &md,
    cudaStream_t &stream) {

    const size_t num_blocks = bz2stream.compressed_data.size();
    
    cudaMemsetAsync(out, 0, num_blocks * bz2stream.block_size, stream);
    CUERR
    
    // compute maximum size of second-stage tables
    const std::uint32_t max_table_size
        = *std::max_element(bz2stream.table_size.begin(),
            bz2stream.table_size.end());

    decode_block<<<num_blocks, 1, max_table_size * sizeof(bz2symbol), stream>>>(
        in, out, md, bz2stream.block_size);
    CUERR
}
