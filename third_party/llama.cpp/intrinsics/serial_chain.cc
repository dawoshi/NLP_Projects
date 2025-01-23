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

#include "third_party/llama.cpp/intrinsics/serial_chain.h"

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/llama.cpp/runtime/intrinsic_handler.h"
#include "third_party/llama.cpp/runtime/status_macros.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

namespace genc {
namespace intrinsics {

absl::Status SerialChain::CheckWellFormed(
    const v0::Intrinsic& intrinsic_pb) const {
  if (!intrinsic_pb.static_parameter().has_struct_() ||
      intrinsic_pb.static_parameter().struct_().element_size() < 1) {
    return absl::InvalidArgumentError(
        "Expect at least 1 function in chain, but got none.");
  }
  return absl::OkStatus();
}

absl::StatusOr<ControlFlowIntrinsicHandlerInterface::ValueRef>
SerialChain::ExecuteCall(const v0::Intrinsic& intrinsic_pb,
                        std::optional<ValueRef> arg, Context* context) const {
  auto params = intrinsic_pb.static_parameter().struct_().element();

  ValueRef state = arg.value();
  for (const auto& fn : params) {
    ValueRef fn_ref = GENC_TRY(context->CreateValue(fn));
    state = GENC_TRY(context->CreateCall(fn_ref, state));
  }

  return state;
}

}  // namespace intrinsics
}  // namespace genc
