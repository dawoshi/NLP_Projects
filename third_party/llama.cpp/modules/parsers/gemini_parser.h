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

#ifndef GENC_CC_MODULES_PARSERS_GEMINI_PARSER_H_
#define GENC_CC_MODULES_PARSERS_GEMINI_PARSER_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/llama.cpp/intrinsics/custom_function.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

namespace genc {

// Parsers for Gemini model.
class GeminiParser final {
 public:
  ~GeminiParser() = default;

  // Extract Top Candidate as Text.
  static absl::StatusOr<v0::Value> GetTopCandidateAsText(v0::Value input);

  // Wraps a text as Gemini request JSON.
  static absl::StatusOr<v0::Value> WrapTextAsInputJson(v0::Value input);

  // Make Parser functions visible to the runtime.
  static absl::Status SetCustomFunctions(
      intrinsics::CustomFunction::FunctionMap& fn_map);

  // Not copyable or movable.
  GeminiParser(const GeminiParser&) = delete;
  GeminiParser& operator=(const GeminiParser&) = delete;

 private:
  // Do not hold states in this class.
  GeminiParser() = default;
};
}  // namespace genc

#endif  // GENC_CC_MODULES_PARSERS_GEMINI_PARSER_H_
