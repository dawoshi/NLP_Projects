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

#include "third_party/llama.cpp/interop/backends/java/google_ai.h"

#include <jni.h>

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/llama.cpp/interop/backends/java/google_ai_client_jni.h"
#include "third_party/llama.cpp/intrinsics/intrinsic_uris.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"
#include <nlohmann/json.hpp>  // IWYU pragma: keep

using json = nlohmann::json;

namespace genc {

namespace {
constexpr absl::string_view kTestModelUri = "/cloud/testing";
absl::StatusOr<std::string> AsString(const json& value) {
  if (!value.is_string()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "expected string, got: " + value.dump());
  }
  return value.get<std::string>();
}
}  // namespace

// Schedules a model call on Google AI backends, using the endpoint and model
// config provided in the 'func' value, and text provided in the 'arg'
// value. Processes the response, and returns the text received.
absl::StatusOr<v0::Value> GoogleAiCall(JavaVM* jvm, jobject google_ai_client,
                                       const v0::Value& func,
                                       const v0::Value& arg) {
  if (func.has_intrinsic() &&
      (func.intrinsic().uri() == intrinsics::kModelInferenceWithConfig) &&
      (func.intrinsic().static_parameter().struct_().element(0).str() ==
       kTestModelUri)) {
    v0::Value resp_pb;
    resp_pb.set_str(absl::StrCat("Testing model with prompt: ", arg.str()));
    return resp_pb;
  }

  std::string model_inference_with_config;
  func.SerializeToString(&model_inference_with_config);
  const std::string& request = arg.str();

  std::string response_str = genc::CallGoogleAiClient(
      jvm, google_ai_client, model_inference_with_config, request);
  if (response_str.empty()) {
    LOG(ERROR) << "Error encountered in fetching response from Google AI.";
    return absl::Status(absl::StatusCode::kInternal,
                        "Internal error in calling Google AI client.");
  }

  json response =
      json::parse(response_str, nullptr, /*allow_exceptions=*/false);
  if (!response.is_object()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Failed to parse Google AI response: " + response_str);
  }

  // Return the text reply.
  json parts = response["candidates"][0]["content"]["parts"];
  std::string full_text = "";
  for (const auto& part : parts) {
    absl::StatusOr<std::string> text = AsString(part["text"]);
    if (!text.ok()) {
      return text.status();
    }
    full_text += text.value();
  }
  v0::Value response_pb;
  response_pb.set_str(full_text);
  return response_pb;
}

}  // namespace genc
