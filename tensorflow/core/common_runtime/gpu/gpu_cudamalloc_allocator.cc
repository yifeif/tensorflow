/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/port.h"
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
#ifdef GOOGLE_CUDA
namespace dyload {

#ifdef PLATFORM_GOOGLE
#define STREAM_EXECUTOR_LIBCUDA_WRAP(__name) \
  struct WrapperShim__##__name {             \
    template <typename... Args>              \
    CUresult operator()(Args... args) {      \
      return ::__name(args...);              \
    }                                        \
  } __name;

#else
#define STREAM_EXECUTOR_LIBCUDA_WRAP(__name)                                 \
  struct DynLoadShim__##__name {                                             \
    static const char *kName;                                                \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;             \
    static void *GetDsoHandle() {                                            \
      auto s =                                                               \
          stream_executor::internal::CachedDsoLoader::GetLibcudaDsoHandle(); \
      return s.ValueOrDie();                                                 \
    }                                                                        \
    static FuncPtrT LoadOrDie() {                                            \
      void *f;                                                               \
      auto s = stream_executor::port::Env::Default()->GetSymbolFromLibrary(  \
          GetDsoHandle(), kName, &f);                                        \
      DCHECK(s.ok()) << "could not find " << kName                           \
                     << " in libcuda DSO; dlerror: " << s.error_message();   \
      return reinterpret_cast<FuncPtrT>(f);                                  \
    }                                                                        \
    static FuncPtrT DynLoad() {                                              \
      static FuncPtrT f = LoadOrDie();                                       \
      return f;                                                              \
    }                                                                        \
    template <typename... Args>                                              \
    CUresult operator()(Args... args) {                                      \
      return DynLoad()(args...);                                             \
    }                                                                        \
  } __name;                                                                  \
  const char *DynLoadShim__##__name::kName = #__name;

#endif
// clang-format off

#define LIBCUDA_ROUTINE_EACH(__macro)                   \
  __macro(cuMemAlloc)                                   \
  __macro(cuMemFree)

// clang-format on

LIBCUDA_ROUTINE_EACH(STREAM_EXECUTOR_LIBCUDA_WRAP)
#undef LIBCUDA_ROUTINE_EACH
}  // namespace dyload
#endif  // GOOGLE_CUDA

GPUcudaMallocAllocator::GPUcudaMallocAllocator(Allocator* allocator,
                                               PlatformGpuId platform_gpu_id)
    : base_allocator_(allocator) {
  stream_exec_ =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
}

GPUcudaMallocAllocator::~GPUcudaMallocAllocator() { delete base_allocator_; }

void* GPUcudaMallocAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
#ifdef GOOGLE_CUDA
  // allocate with cudaMalloc
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  CUdeviceptr rv = 0;
  CUresult res = dyload::cuMemAlloc(&rv, num_bytes);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "cuMemAlloc failed to allocate " << num_bytes;
    return nullptr;
  }
  return reinterpret_cast<void*>(rv);
#else
  return nullptr;
#endif  // GOOGLE_CUDA
}
void GPUcudaMallocAllocator::DeallocateRaw(void* ptr) {
#ifdef GOOGLE_CUDA
  // free with cudaFree
  CUresult res = dyload::cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "cuMemFree failed to free " << ptr;
  }
#endif  // GOOGLE_CUDA
}

bool GPUcudaMallocAllocator::TracksAllocationSizes() { return false; }

}  // namespace tensorflow
