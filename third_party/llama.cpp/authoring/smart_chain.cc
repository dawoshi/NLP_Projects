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

#include "third_party/llama.cpp/authoring/smart_chain.h"

#include <vector>

#include "absl/status/statusor.h"
#include "third_party/llama.cpp/authoring/constructor.h"
#include "third_party/llama.cpp/runtime/status_macros.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

namespace genc {

absl::StatusOr<v0::Value> SmartChain::Build() {
  v0::Value resulting_chain;
  if (num_iteration_ > 1) {
    resulting_chain =
        GENC_TRY(CreateRepeatedConditionalChain(num_iteration_, chained_ops_));
  } else {
    resulting_chain = GENC_TRY(CreateSerialChain(chained_ops_));
  }

  if (is_parallel_) {
    resulting_chain = GENC_TRY(CreateParallelMap(resulting_chain));
  }
  return resulting_chain;
}

}  // namespace genc
