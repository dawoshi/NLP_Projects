/* Copyright 2023, The GenC Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#ifndef GENC_CC_RUNTIME_EXECUTOR_STACKS_H_
#define GENC_CC_RUNTIME_EXECUTOR_STACKS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "third_party/llama.cpp/runtime/concurrency.h"
#include "third_party/llama.cpp/runtime/executor.h"
#include "third_party/llama.cpp/runtime/intrinsic_handler.h"

namespace genc {

// Constructs a local executor for a given set of intrinsic handlers.
// Use this when you have a set of custom handlers to add that is non default.
absl::StatusOr<std::shared_ptr<Executor>> CreateLocalExecutor(
    std::shared_ptr<IntrinsicHandlerSet> handler_set,
    std::shared_ptr<ConcurrencyInterface> concurrency_interface = nullptr);

// Returns an executor that performs local processing in-process.
absl::StatusOr<std::shared_ptr<Executor>> CreateDefaultLocalExecutor();

}  // namespace genc

#endif  // GENC_CC_RUNTIME_EXECUTOR_STACKS_H_
