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

#ifndef GENC_CC_TYPES_CHECKING_H_
#define GENC_CC_TYPES_CHECKING_H_

#include "absl/status/status.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

namespace genc {

absl::Status CheckEqual(v0::Type x, v0::Type y);

}  // namespace genc

#endif  // GENC_CC_TYPES_CHECKING_H_
