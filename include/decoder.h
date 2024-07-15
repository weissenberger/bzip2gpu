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

#ifndef DECODER_
#define DECODER_

#include <memory>
#include <tuple>

#include "parser.h"

struct bz2uchar4 {

    std::uint8_t s0; std::uint8_t s1; std::uint8_t s2; std::uint8_t s3;
};

struct bz2uchar16 {
    
    std::uint8_t s0; std::uint8_t s1; std::uint8_t s2; std::uint8_t s3;
    std::uint8_t s4; std::uint8_t s5; std::uint8_t s6; std::uint8_t s7;
    std::uint8_t s8; std::uint8_t s9; std::uint8_t s10; std::uint8_t s11;
    std::uint8_t s12; std::uint8_t s13; std::uint8_t s14; std::uint8_t s15;
};

#endif /* DECODER_H_ */
