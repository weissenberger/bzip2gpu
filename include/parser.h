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

#ifndef PARSER_H_
#define PARSER_H_

#include <memory>
#include <vector>
#include <cstdint>

struct location {
    size_t unit;
    size_t bit;
    
    friend inline bool operator<=(location const &a, location const &b) {

        if(a.unit < b.unit) return true;
        if((a.unit == b.unit) && (a.bit <= b.bit)) return true;

        return false;
    }

    friend inline location operator+(location const &a, location const &b) {

        return location{a.unit + b.unit, a.bit + b.bit};
    }    
};

template <typename T>
struct stream {

    inline stream(T* base,
        location begin,
        location end) : base(base), begin(begin), end(end) {
            
            loc = {0,0};
            win = __builtin_bswap64(base[begin.unit]);
            next = __builtin_bswap64(base[begin.unit + 1]);
            
            if(begin.bit > 0) forward(begin.bit);
            loc = begin;
    }

    inline void skip(size_t len) {

        // if request exceeds unit boundary
        if(loc.bit + len >= s) {

            win = __builtin_bswap64(base[++loc.unit]);
            next = __builtin_bswap64(base[loc.unit + 1]);
            size_t prev_bit = loc.bit + len - s;
            loc.bit = 0;

            if(prev_bit > 0) forward(prev_bit);
        }

        // if request does not exceed unit boundary
        else forward(len);
    }

    inline void read(size_t len, T &buffer) {
        
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
        inline void forward(size_t len) {

            win <<= len;
            win |= ((next << loc.bit) >> (s - len));
            loc.bit += len;
        }
};

struct bz2symbol {
    
    std::uint16_t symbol;
    std::uint16_t length;
};

struct bz2subtable_entry {

    std::uint16_t position;
    std::uint16_t remainder;
};

struct bz2stream {

    std::uint32_t block_size;
    std::uint32_t checksum;
    std::vector<location> compressed_data;
    std::vector<std::uint32_t> bwt_indices;
    std::vector<std::uint32_t> block_checksums;
    std::vector<std::uint8_t> mtf_stacks;
    std::vector<std::uint32_t> num_trees;
    std::vector<std::uint32_t> num_selectors;
    std::vector<std::uint8_t> selectors;
    std::vector<std::uint32_t> stack_size;
    std::vector<std::uint8_t> code_lengths;
    std::vector<std::uint8_t> max_lones;
    
    std::vector<std::uint32_t> table_size;
    std::vector<std::uint32_t> table_position;

    std::vector<bz2subtable_entry> subtables;
    std::vector<bz2symbol> table;
  
    std::vector<std::uint32_t> output_size;
};

class parser {

    public:
        
        static void parse(std::uint64_t* in, bz2stream &bz2stream,
        size_t length, size_t num_threads);

        static void build_tables(bz2stream &bz2stream, size_t num_threads);
};

#endif /* PARSER_H_ */
