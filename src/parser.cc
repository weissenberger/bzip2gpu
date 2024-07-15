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

#include "parser.h"

#include <iostream>
#include <vector>
#include <bitset>
#include <numeric>
#include <algorithm>

#include <omp.h>

void parse_symmap(stream<std::uint64_t> &s, bz2stream &bz2s, size_t block) {

    std::uint64_t buffer;
    size_t idx = 0;

    s.read(16, buffer);
    s.skip(16);
    const std::uint64_t map1 = buffer >> 48;

    for(size_t i = 0; i < 16; ++i) {

        if(map1 & (0x8000 >> i)) {

            s.read(16, buffer);
            s.skip(16);
            const std::uint64_t map2 = buffer >> 48;

            for(size_t j = 0; j < 16; ++j) {

                if(map2 & (0x8000 >> j)) {

                    bz2s.mtf_stacks[(256 * block) + idx] = (16 * i) + j;
                    ++idx;
                }
            }
        }
    }

    bz2s.stack_size[block] = static_cast<std::uint32_t>(idx);
}

void parse_selectors(stream<std::uint64_t> &s, bz2stream &bz2s, size_t block) {
    
    std::uint64_t buffer;
    const size_t base = 32768 * block;
    std::vector<std::uint8_t> stack(bz2s.num_trees[block]);
    for(size_t i = 0; i < bz2s.num_trees[block]; ++i) stack[i] = i;

    for(size_t i = 0; i < bz2s.num_selectors[block]; ++i) {

        std::uint8_t idx = 0;
        s.read(1, buffer);

        while((buffer >> 63)) {

            ++idx;
            s.skip(1);
            s.read(1, buffer);
        }

        s.skip(1);
        
        bz2s.selectors[base + i] = idx;
    }

    std::uint8_t sym;

    auto popp_shift = [&](std::uint8_t pos) {

        sym = stack[pos];

        for(size_t i = pos; i > 0; --i) {

            stack[i] = stack[i - 1];
        }

        stack[0] = sym;
    };

    for(size_t i = 0; i < bz2s.num_selectors[block]; ++i) {

        popp_shift(bz2s.selectors[base + i]);
        bz2s.selectors[base + i] = sym;
    }
}

void parse_trees(stream<std::uint64_t> &s, bz2stream &bz2s, size_t block) {

    std::uint64_t buffer;
    
    const std::uint32_t num_symbols = bz2s.stack_size[block] + 2;

    for(size_t t = 0; t < bz2s.num_trees[block]; ++t) {

        std::uint8_t length;
        s.read(5, buffer);
        length = static_cast<std::uint8_t>(buffer >> 59);
        s.skip(5);

        for(size_t i = 0; i < num_symbols; ++i) {

            while(true) {

                s.read(1, buffer);
                s.skip(1);

                if(!(buffer >> 63)) break;
                
                s.read(1, buffer);
                s.skip(1);

                if((buffer >> 63)) --length; else ++length;
            }

            bz2s.code_lengths[(512 * 6 * block) + (512 * t) + i] = length;
        }
    }
}

void parser::parse(std::uint64_t* in, bz2stream &bz2s,
    size_t length, size_t num_threads) {

    std::vector<std::vector<location>> local_blocks(num_threads);

    const size_t piece_size = length / num_threads;
    const size_t last_piece_size = length % num_threads;
    
    omp_set_num_threads(num_threads);
    
    {
        std::uint64_t buffer;
        stream<std::uint64_t> stream(in, {0,0}, {0,0});
        stream.skip(24);
        stream.read(8, buffer);
        bz2s.block_size = ((buffer >> 56) - 48) * 100000;
    }

    #pragma omp parallel for
    for(size_t i = 0; i < num_threads; ++i) {

        auto process_piece = [&](size_t idx) {

            std::uint64_t curr = __builtin_bswap64(in[idx]);
            std::uint64_t next = __builtin_bswap64(in[idx + 1]);

            const std::uint64_t magic = 0x3141592653590000;
            const std::uint64_t mask = ~0ul;

            #pragma GCC unroll 17
            for(std::uint32_t j = 0; j <= 16; ++j) {

                if(((curr << j) & (mask << 16)) == magic)
                    local_blocks[i].push_back({idx, j});
            }

            #pragma GCC unroll 47
            for(std::uint32_t j = 17; j < 64; ++j) {

                const std::uint64_t masked_magic = magic & (mask << j);

                if((curr << j) == masked_magic) {
                    const std::uint64_t shifted_magic = magic << (64 - j);

                    if((next & (mask << (80 - j))) == shifted_magic)
                        local_blocks[i].push_back({idx, j});
                }
            }
        };

        for(size_t k = i * piece_size; k < ((i + 1) * piece_size); ++k) {

            process_piece(k);
        }

        if((i == num_threads - 1) && (last_piece_size > 0)) {

            for(size_t k = (i + 1) * piece_size;
                k < (i + 1) * piece_size + last_piece_size; ++k) {

                process_piece(k);
            }
        }
    }
    
    // compute total number of blocks in stream
    size_t blocks = 0;
    for(auto i : local_blocks) blocks += i.size();
    
    // flatten local vectors
    std::vector<location> global_blocks = std::accumulate(
        local_blocks.begin(), local_blocks.end(),
        decltype(local_blocks)::value_type{},
        [](auto &x, auto &y) {x.insert(x.end(), y.begin(), y.end());
        return x;
    });

    bz2s.mtf_stacks.resize(256 * blocks);
    bz2s.block_checksums.resize(blocks);
    bz2s.bwt_indices.resize(blocks);
    bz2s.num_trees.resize(blocks);
    bz2s.num_selectors.resize(blocks);
    bz2s.selectors.resize(32768 * blocks);
    bz2s.stack_size.resize(blocks);
    bz2s.code_lengths.resize(512 * 6 * blocks);
    bz2s.compressed_data.resize(blocks);
    bz2s.max_lones.resize(6 * blocks);
    bz2s.output_size.resize(blocks);

    std::uint64_t buffer;

    #pragma omp parallel for
    for(size_t i = 0; i < blocks; ++i) {
        
        stream<std::uint64_t> stream(in, global_blocks[i], global_blocks[i]);

        // block magic
        stream.skip(48);

        // CRC32
        stream.read(32, buffer);
        bz2s.block_checksums[i]= static_cast<std::uint32_t>(buffer >> 32);
        stream.skip(32);
        
        // randomization
        stream.skip(1);

        // BWT index
        stream.read(24, buffer);
        bz2s.bwt_indices[i] = static_cast<std::uint32_t>(buffer >> 40);
        stream.skip(24);
        
        // SymMap (MTF stacks)
        parse_symmap(stream, bz2s, i);

        // number of Huffman trees
        stream.read(3, buffer);
        bz2s.num_trees[i] = static_cast<std::uint32_t>(
            buffer >> 61);
        stream.skip(3);

        // number of selectors
        stream.read(15, buffer);
        bz2s.num_selectors[i] = static_cast<std::uint32_t>(
            buffer >> 49);
        stream.skip(15);

        // selectors
        parse_selectors(stream, bz2s, i);

        // trees
        parse_trees(stream, bz2s, i);

        // store begin of compressed data
        bz2s.compressed_data[i] = stream.loc;
    }
}

void compute_table_size(bz2stream &bz2stream, const size_t block) {

    const size_t num_tables = bz2stream.num_trees[block];
    const size_t num_symbols = bz2stream.stack_size[block] + 2;

    // number of entries in current table
    size_t num_entries = 0;

    std::vector<bz2subtable_entry> subtable(21 * 6, {0,0});

    for(size_t k = 0; k < num_tables; ++k) {

        // build table
        struct length_symbol{std::uint8_t length; std::uint16_t symbol;};

        std::vector<length_symbol> ls_pairs(num_symbols);

        // load code lengths
        for(size_t i = 0; i < num_symbols; ++i) {

            ls_pairs[i] = {bz2stream.code_lengths[
                512 * 6 * block + (512 * k) + i],
                static_cast<std::uint16_t>(i)};
        }
    
        // sort code lengths
        std::stable_sort(ls_pairs.begin(), ls_pairs.end(),
            [](const length_symbol &a, const length_symbol &b) {
                return a.length < b.length;});

        // create table entries
        std::uint32_t curr_code = 0;
        std::uint32_t curr_lones = 0;
        std::uint32_t curr_remainder = 0;
        std::uint32_t curr_len = ls_pairs[0].length;
        
        std::uint8_t max_lones = 0;

        for(size_t i = 0; i < num_symbols; ++i) {
            
            // process single symbol
            while(curr_len < ls_pairs[i].length) {

                curr_code <<= 1;
                ++curr_len;
            }

            curr_lones = __builtin_clz(~(curr_code << (32 - curr_len)));
            curr_remainder = curr_len - curr_lones;

            if(curr_remainder > subtable[(k * 21) + curr_lones].position)
                subtable[(k * 21) + curr_lones].position = curr_remainder;
                
            max_lones = curr_lones;
            ++curr_code;
        }

        // number of entries in current table
        for(size_t i = 0; i < 21; ++i) {
            
            std::uint16_t curr_table = (1 << subtable[(k * 21) + i].position);
            std::uint16_t remainder = subtable[(k * 21) + i].position;
            subtable[(k * 21) + i].position = num_entries;
            subtable[(k * 21) + i].remainder = remainder;
            num_entries += curr_table;
        }
        
        bz2stream.max_lones[(block * 6) + k] = max_lones;
    }

    std::copy(subtable.begin(), subtable.end(),
        bz2stream.subtables.begin() + (block * 21 * 6));

    bz2stream.table_size[block] = num_entries;
}

void compute_table(bz2stream &bz2stream, const size_t block) {

    const size_t num_tables = bz2stream.num_trees[block];
    const size_t num_symbols = bz2stream.stack_size[block] + 2;
    const std::uint32_t table_pos = bz2stream.table_position[block];

    for(size_t k = 0; k < num_tables; ++k) {

        const std::uint32_t table_base = block * 21 * 6 + (21 * k);

        // build table
        struct length_symbol{std::uint8_t length; std::uint16_t symbol;};

        std::vector<length_symbol> ls_pairs(num_symbols);

        // load code lengths
        for(size_t i = 0; i < num_symbols; ++i) {

            ls_pairs[i] = {bz2stream.code_lengths[
                512 * 6 * block + (512 * k) + i],
                static_cast<std::uint16_t>(i)};
        }
    
        // sort code lengths
        std::stable_sort(ls_pairs.begin(), ls_pairs.end(),
            [](const length_symbol &a, const length_symbol &b) {
                return a.length < b.length;});

        // create table entries
        std::uint32_t curr_code = 0;
        std::uint32_t curr_lones = 0;
        std::uint32_t curr_remainder = 0;
        std::uint16_t curr_len = ls_pairs[0].length;

        for(size_t i = 0; i < num_symbols; ++i) {
            
            // process single symbol
            while(curr_len < ls_pairs[i].length) {

                curr_code <<= 1;
                ++curr_len;
            }

            curr_lones = __builtin_clz(~(curr_code << (32 - curr_len)));
            curr_remainder = curr_len - curr_lones;
            
            const std::uint32_t sub_position
                = bz2stream.subtables[table_base + curr_lones].position
                    + table_pos;
            const std::uint32_t sub_remainder 
                = bz2stream.subtables[table_base + curr_lones].remainder;
            
            const std::uint32_t remainder = sub_remainder - curr_remainder;
            const std::uint32_t mask = (1 << curr_remainder) - 1;
            
            const size_t base = (curr_code & mask)  << remainder;
            const size_t end = (((curr_code + 1) & mask) << remainder)
                + (curr_len == curr_lones);

            for(size_t j = base; j < end; ++j)
                bz2stream.table[sub_position + j]
                    = {ls_pairs[i].symbol, curr_len};

            ++curr_code;
        }
    }
}

void parser::build_tables(bz2stream &bz2stream, size_t num_threads) {

    const size_t num_blocks = bz2stream.compressed_data.size();
    
    bz2stream.table_size.resize(num_blocks);
    bz2stream.table_position.resize(num_blocks);

    bz2stream.subtables.resize(num_blocks * 6 * 21, {0,0});

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for(size_t i = 0; i < num_blocks; ++i)
        compute_table_size(bz2stream, i);

    std::exclusive_scan(bz2stream.table_size.begin(),
        bz2stream.table_size.end(),
        bz2stream.table_position.begin(), 0);
    
    bz2stream.table.resize(std::accumulate(bz2stream.table_size.begin(),
        bz2stream.table_size.end(), 0));
        
    #pragma omp parallel for
    for(size_t i = 0; i < num_blocks; ++i)
        compute_table(bz2stream, i);
}
