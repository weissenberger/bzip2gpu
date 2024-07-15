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

#include "bz2gpu.h"

void bz2gpu::decode_stream(
    std::uint64_t* in,
    std::uint8_t* &out,
    size_t &output_size,
    const size_t size,
    const size_t num_threads,
    cudaStream_t &stream) {

    bz2stream bz2s;
    device_metadata md;

    // allocate input buffer
    std::uint64_t* d_input;
    cudaMallocAsync(&d_input, size * sizeof(std::uint64_t), stream);
    CUERR

    // transfer bz2 stream to device
    cudaMemcpyAsync(d_input, in, size * sizeof(std::uint64_t), 
        cudaMemcpyHostToDevice, stream);
    CUERR

    // parse the stream
    parser::parse(in, bz2s, size, num_threads);

    // build Huffman / metadata tables
    parser::build_tables(bz2s, num_threads);

    const size_t num_blocks = bz2s.compressed_data.size();
    const size_t block_size = bz2s.block_size;

    // allocate output buffer
    bz2uchar16* d_decoder_out;
    
    cudaMallocAsync(&d_decoder_out, num_blocks * block_size, stream);
    CUERR

    // allocate and copy metadata to device
    decoder_cuda::alloc_metadata(md, bz2s, stream);
    decoder_cuda::copy_metadata(md, bz2s, stream);

    // decode
    decoder_cuda::decode(d_input, d_decoder_out, bz2s, md, stream);

    // iMTF
    imtf_cuda::batch_iMTF(reinterpret_cast<std::uint8_t*>(d_decoder_out),
        md.mtf_stacks, num_blocks, block_size, stream);
    
    // iBWT
    ibwt_cuda::batch_iBWT(reinterpret_cast<std::uint8_t*>(d_decoder_out),
        md.bwt_indices, num_blocks, block_size, md.output_size, stream);

    // RLE1
    rle_cuda::decode(reinterpret_cast<std::uint8_t*>(d_decoder_out),
        out, bz2s, md, output_size, stream);

    // free output buffer
    cudaFreeAsync(d_decoder_out, stream);
    CUERR

    // free input buffer
    cudaFreeAsync(d_input, stream);
    CUERR

    // free metadata
    decoder_cuda::free_metadata(md, stream);
}
