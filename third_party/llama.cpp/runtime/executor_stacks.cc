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

#include "third_party/llama.cpp/runtime/executor_stacks.h"

#include <memory>

#include "absl/status/statusor.h"
#include "third_party/llama.cpp/intrinsics/handler_sets.h"
#include "third_party/llama.cpp/runtime/concurrency.h"
#include "third_party/llama.cpp/runtime/control_flow_executor.h"
#include "third_party/llama.cpp/runtime/executor.h"
#include "third_party/llama.cpp/runtime/inline_executor.h"
#include "third_party/llama.cpp/runtime/intrinsic_handler.h"
#include "third_party/llama.cpp/runtime/status_macros.h"
#include "third_party/llama.cpp/runtime/threading.h"

namespace genc {

absl::StatusOr<std::shared_ptr<Executor>> CreateLocalExecutor(
    std::shared_ptr<IntrinsicHandlerSet> handler_set,
    std::shared_ptr<ConcurrencyInterface> concurrency_interface) {
  if (concurrency_interface == nullptr) {
    concurrency_interface = CreateThreadBasedConcurrencyManager();
  }
  return CreateControlFlowExecutor(
      handler_set,
      GENC_TRY(CreateInlineExecutor(handler_set, concurrency_interface)),
      concurrency_interface);
}

absl::StatusOr<std::shared_ptr<Executor>> CreateDefaultLocalExecutor() {
  return CreateLocalExecutor(intrinsics::CreateCompleteHandlerSet({}));
}

}  // namespace genc
