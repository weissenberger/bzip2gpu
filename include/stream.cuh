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

#ifndef STREAM_CUH_
#define STREAM_CUH_

#include "parser.h"

template <typename T>
struct cu_stream {

    __device__ __forceinline__ cu_stream(const T* __restrict__ base,
        location begin,
        location end) : base(base), begin(begin), end(end) {
            
            loc = {0,0};
            win = bswap64(base[begin.unit]);
            next = bswap64(base[begin.unit + 1]);
            
            if(begin.bit > 0) forward(begin.bit);
            loc = begin;
    }

    __device__ __forceinline__ void skip(size_t len) {

        // if request exceeds unit boundary
        if(loc.bit + len >= s) {

            win = bswap64(base[++loc.unit]);
            next = bswap64(base[loc.unit + 1]);
            size_t prev_bit = loc.bit + len - s;
            loc.bit = 0;

            if(prev_bit > 0) forward(prev_bit);
        }

        // if request does not exceed unit boundary
        else forward(len);
    }

    __device__ __forceinline__ void read(size_t len, T &buffer) {
        
        const T mask = (~0ul) << (s - len);
        buffer = win & mask;
    }

    const std::uint32_t s = sizeof(T) * 8;
    const T* base;

    const location begin;
    const location end;

    T win;
    T next;
    location loc;

    private:

        // if request does not exceed unit boundary
        __device__ __forceinline__ void forward(size_t len) {

            win <<= len;
            win |= ((next << loc.bit) >> (s - len));
            loc.bit += len;
        }
        
        __device__ __forceinline__ std::uint64_t bswap64(std::uint64_t u) {
            const std::uint32_t hi = __byte_perm(u >> 32, 0, 0x0123);
            const std::uint32_t lo = __byte_perm(u, 0, 0x0123);
            
            return (((std::uint64_t) lo) << 32) | hi;
        }
};

#endif /* STREAM_CUH_ */
