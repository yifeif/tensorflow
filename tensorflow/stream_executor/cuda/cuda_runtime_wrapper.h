/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// This file wraps cuda runtime calls with dso loader so that we don't need to
// have explicit linking to libcuda. All TF cuda core runtime usage should route
// through this wrapper.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_WRAPPER_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_WRAPPER_H_

#include "cuda/include/cuda_runtime.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace tensorflow {
namespace wrap {
#ifdef PLATFORM_GOOGLE
// Use static linked library
#define STREAM_EXECUTOR_LIBCUDART_WRAP(cudaSymbolName)                     \
  template <typename... Args>                                              \
  auto cudaSymbolName(Args... args)->decltype(::cudaSymbolName(args...)) { \
    return ::cudaSymbolName(args...);                                      \
  }

// This macro wraps a global identifier, given by cudaSymbolName, in a callable
// structure that loads the DLL symbol out of the DSO handle in a thread-safe
// manner on first use. This dynamic loading technique is used to avoid DSO
// dependencies on vendor libraries which may or may not be available in the
// deployed binary environment.
#else
#define TO_STR_(x) #x
#define TO_STR(x) TO_STR_(x)

#define STREAM_EXECUTOR_LIBCUDART_WRAP(cudaSymbolName)                        \
  template <typename... Args>                                                 \
  auto cudaSymbolName(Args... args)->decltype(::cudaSymbolName(args...)) {    \
    using FuncPtrT = std::add_pointer<decltype(::cudaSymbolName)>::type;      \
    static FuncPtrT loaded = []() -> FuncPtrT {                               \
      static const char *kName = TO_STR(cudaSymbolName);                      \
      void *f;                                                                \
      auto s = stream_executor::port::Env::Default()->GetSymbolFromLibrary(   \
          stream_executor::internal::CachedDsoLoader::GetLibcudartDsoHandle() \
              .ValueOrDie(),                                                  \
          kName, &f);                                                         \
      CHECK(s.ok()) << "could not find " << kName                             \
                    << " in libcudart DSO; dlerror: " << s.error_message();   \
      return reinterpret_cast<FuncPtrT>(f);                                   \
    }();                                                                      \
    return loaded(args...);                                                   \
  }
#endif

// clang-format off
#define LIBCUDART_ROUTINE_EACH(__macro)                   \
  __macro(cudaFree)                                       \
  __macro(cudaGetDevice)                                  \
  __macro(cudaGetDeviceProperties)                        \
  __macro(cudaGetErrorString)                             \
  __macro(cudaGetLastError)                               \
  __macro(cudaSetDevice)                                  \
  __macro(cudaStreamAddCallback)                          \
  __macro(cudaStreamCreate)                               \
  __macro(cudaStreamDestroy)                              \
  __macro(cudaGetDeviceCount)                             \
  __macro(cudaStreamSynchronize)

// clang-format on
LIBCUDART_ROUTINE_EACH(STREAM_EXECUTOR_LIBCUDART_WRAP)
#undef LIBCUDART_ROUTINE_EACH
#undef STREAM_EXECUTOR_LIBCUDART_WRAP
#undef TO_STR
#undef TO_STR_
}  // namespace wrap
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_WRAPPER_H_
