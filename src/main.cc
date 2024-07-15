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
#include <filesystem>
#include <fstream>

#include "bz2gpu.h"

void load_file_with_padding(
    std::string path,
    std::shared_ptr<std::uint64_t[]> &buffer,
    size_t &size) {
    
        std::ifstream file(path, std::ifstream::binary);

        // get file size
        file.seekg(0, file.end);
        size = file.tellg();

        // add padding to complete final block, if needed
        const size_t units = SDIV(size, 8);
        
        // reset cursor to start of file
        file.seekg(0, file.beg);

        buffer = std::shared_ptr<std::uint64_t[]>(new std::uint64_t[units]);

        file.read(reinterpret_cast<char*>(buffer.get()), size);
        size = units;

        file.close();
}

void store_file(std::uint8_t* data, size_t size,
    std::string path) {

    std::ofstream file(path, std::ofstream::binary);

    file.write(reinterpret_cast<char*> (data), size);

    file.close();
}

int main(int argc, char** argv) {

    std::shared_ptr<std::uint64_t[]> compressed_data;
    std::size_t compressed_data_size;

    std::uint8_t* d_output;
    std::size_t output_size;

    // select device
    if(argc < 4) cudaSetDevice(0); else cudaSetDevice(atoi(argv[3]));
    CUERR

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CUERR

    // load bz2 stream from disk
    load_file_with_padding(
        std::filesystem::absolute(
            std::filesystem::path(argv[1])),
        compressed_data,
        compressed_data_size);

    // decompression using GPU
    bz2gpu::decode_stream(
        compressed_data.get(),
        d_output,
        output_size,
        compressed_data_size,
        omp_get_max_threads(),
        stream);

    // allocate host buffer for output
    std::shared_ptr<std::uint8_t[]> h_output(new std::uint8_t[output_size]);
    
    // transfer decompressed stream from device to host
    cudaMemcpyAsync(h_output.get(), d_output, output_size,
        cudaMemcpyDeviceToHost, stream);
    CUERR

    // free device output buffer
    cudaFreeAsync(d_output, stream);
    CUERR

    // store uncompressed data
    store_file(
        h_output.get(),
        output_size,
        std::filesystem::absolute(
            std::filesystem::path(argv[2])));

    return 0;
}
