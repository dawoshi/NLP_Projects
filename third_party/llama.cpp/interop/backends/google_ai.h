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

#ifndef GENC_CC_INTEROP_BACKENDS_GOOGLE_AI_H_
#define GENC_CC_INTEROP_BACKENDS_GOOGLE_AI_H_

#include "absl/status/status.h"
#include "third_party/llama.cpp/intrinsics/model_inference_with_config.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"
namespace genc {

// Backend for calling Google Model backends such as
// https://platform.openai.com/docs/api-reference
// https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini

class GoogleAI final {
 public:
  ~GoogleAI() = default;

  // Not copyable or movable.
  GoogleAI(const GoogleAI&) = delete;
  GoogleAI& operator=(const GoogleAI&) = delete;

  // Sets the inference map to process model calls.
  static absl::Status SetInferenceMap(
      intrinsics::ModelInferenceWithConfig::InferenceMap& inference_map);

 private:
  // Do not hold states in this class.
  GoogleAI() = default;
};
}  // namespace genc

#endif  // GENC_CC_INTEROP_BACKENDS_GOOGLE_AI_H_
