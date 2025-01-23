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

#include "third_party/llama.cpp/base/computation.h"

#include <vector>

#include "absl/status/statusor.h"
#include "third_party/llama.cpp/base/context.h"
#include "third_party/llama.cpp/runtime/status_macros.h"

namespace genc {

const v0::Value& Computation::portable_ir() { return portable_ir_; }

absl::StatusOr<Computation> Computation::operator()(
    const std::vector<v0::Value>& args) {
  v0::Value result =
      GENC_TRY(GetContextStack()->CurrentContext()->Call(portable_ir_, args));
  return Computation(result);
}
}  // namespace genc
