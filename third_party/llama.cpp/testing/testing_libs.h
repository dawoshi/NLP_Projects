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

#ifndef GENC_CC_TESTING_TESTING_LIBS_H_
#define GENC_CC_TESTING_TESTING_LIBS_H_
// Common libs used for simplify testing code or running mock examples.
#include <string>
#include "absl/status/statusor.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

namespace genc {
namespace testing {

// Returns a Value contains a str fn_name(formatted_value).
absl::StatusOr<genc::v0::Value> WrapFnNameAroundValue(
    std::string fn_name, const genc::v0::Value& value);

// Returns the content of value, if value contains struct, returns the content
// of the struct separated by ','. used for testing purpose.
absl::StatusOr<std::string> FormatValueAsString(
    const genc::v0::Value& value);

}  // namespace testing
}  // namespace genc
#endif  // GENC_CC_TESTING_TESTING_LIBS_H_
