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

#ifndef DECODER_CUDA_
#define DECODER_CUDA_

#include "parser.h"
#include "decoder.h"

struct device_metadata {

    uint8_t* max_lones;
    
    std::uint32_t* table_size;
    std::uint32_t* table_position;

    bz2subtable_entry* subtables;
    bz2symbol* table;
  
    std::uint32_t* output_size;

    std::uint32_t* num_selectors;
    std::uint8_t* selectors;
    std::uint32_t* stack_size;

    location* compressed_data;
  
    std::uint8_t* mtf_stacks;
    std::uint32_t* bwt_indices;
};

class decoder_cuda {

    public:
        
        static void alloc_metadata(
            device_metadata &md,
            const bz2stream &bz2s,
            cudaStream_t &stream);

        static void copy_metadata(
            device_metadata &md,
            bz2stream &bz2s,
            cudaStream_t &stream);

        static void free_metadata(
            const device_metadata &md,
            cudaStream_t &stream);

        static void decode(
            std::uint64_t* in,
            bz2uchar16* out,
            bz2stream &bz2stream,
            device_metadata &md,
            cudaStream_t &stream);
};

#endif /* DECODER_CUDA_H_ */
