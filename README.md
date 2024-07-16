# bzip2gpu

Fast, GPU-based decompressor for standard bzip2 streams. Except for the initial parser, all stages of the bzip2 pipeline (Huffman decoding, RLE2, inv. Move-to-Front, inv. Burrows-Wheeler Transform, RLE1) are executed on the GPU. Uncompressed output is kept in device memory.

This current implementation supports bzip2 streams of up to 2^31 - 1 bytes (uncompressed).

For further information, please refer to our [conference paper](https://doi.org/10.1145/3673038.3673067).

> Note: This program serves as auxiliary material for our publication and
> is not suitable for production use.

## Requirements

* CUDA-enabled GPU (compute capability 7.0 or higher)
* GNU/Linux
* GNU compiler version 11.4.1 or higher
* CUDA SDK 12.4.1 or higher
* latest proprietary graphics drivers

## Demo program

### Compiling the demo program

To compile the demo program, run:

`make`

### Running the demo program

`./bin/bz2gpu <input_path> <output_path> <GPU to be used> (default 0)`

## TODO

- [ ] batched execution for supporting larger streams
- [ ] port parser to CUDA
- [ ] stream error detection (parsing, checksums,...)
- [ ] recursive variant of Helman-Jájá for iBWT
- [ ] random selection of splitters for list ranking
- [ ] automatic parameter tuning
- [ ] unit tests

## Licensing

This project is distributed under the MIT License (see LICENSE).

The files

`src/rle.cu
include/thrust_custom_alloc.cuh`

contain modified sourcecode from the [Thrust library](https://github.com/NVIDIA/cccl/tree/main/thrust). Thrust is distributed under the [Apache License v2.0 with LLVM Exceptions](https://llvm.org/LICENSE.txt) (see thrust_license).

## Test file

`test/lichess_puzzles_10M.bz2`

contains the first 10 million bytes from the [lichess.org puzzle dataset](https://database.lichess.org/#puzzles) and can be used for quick tests.
