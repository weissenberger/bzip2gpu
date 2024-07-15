#ifndef THRUSTCUDAMALLOCASYNCALLOCATOR_
#define THRUSTCUDAMALLOCASYNCALLOCATOR_

// modified Thrust allocator from
// https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>

template<class T>
struct ThrustCudaMallocAsyncAllocator : thrust::device_malloc_allocator<T> {

    using value_type = T;
    using super_t = thrust::device_malloc_allocator<T>;

    using pointer = typename super_t::pointer;
    using size_type = typename super_t::size_type;
    using reference = typename super_t::reference;
    using const_reference = typename super_t::const_reference;

    cudaStream_t stream{};

    ThrustCudaMallocAsyncAllocator(cudaStream_t stream_)
        : stream(stream_) {
        
    }

    pointer allocate(size_type n) {

        T* ptr = nullptr;
        cudaError_t status = cudaMallocAsync(&ptr, n * sizeof(T), stream);

        if(status != cudaSuccess) {
            
            cudaGetLastError(); //reset error state
            throw std::bad_alloc();
        }

        return thrust::device_pointer_cast(ptr);
    }

    void deallocate(pointer ptr, size_type /* n */) {

        cudaError_t status = cudaFreeAsync(ptr.get(), stream);
        
        if(status != cudaSuccess) {

            cudaGetLastError(); //reset error state
            throw std::bad_alloc();
        }
    }
};

#endif/* THRUSTCUDAMALLOCASYNCALLOCATOR_H_ */