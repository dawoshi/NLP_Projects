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

#include "third_party/llama.cpp/interop/backends/java/open_ai.h"

#include <jni.h>

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/llama.cpp/interop/backends/java/open_ai_client_jni.h"
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

// Creates an OpenAI chat completion request using chat completions API, sends
// it to on-device OpenAI client which sends the request to OpenAI server.
// Processes the response, and returns the next round of chat reply.
absl::StatusOr<v0::Value> OpenAiCall(JavaVM* jvm, jobject open_ai_client,
                                     const v0::Value& func,
                                     const v0::Value& arg) {
  if (func.has_intrinsic() &&
      (func.intrinsic().uri() == intrinsics::kModelInference) &&
      (func.intrinsic().static_parameter().str() == kTestModelUri)) {
    v0::Value resp_pb;
    resp_pb.set_str(absl::StrCat("Testing model with prompt: ", arg.str()));
    return resp_pb;
  }

  std::string model_inference_with_config;
  func.SerializeToString(&model_inference_with_config);
  const std::string& request = arg.str();

  std::string response_str = genc::CallOpenAiClient(
      jvm, open_ai_client, model_inference_with_config, request);
  if (response_str.empty()) {
    LOG(ERROR) << "Error encountered in fetching response from OpenAI.";
    return absl::Status(absl::StatusCode::kInternal,
                        "Internal error in calling OpenAI client");
  }

  json response =
      json::parse(response_str, nullptr, /*allow_exceptions=*/false);
  if (!response.is_object()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Failed to parse OpenAI response: " + response_str);
  }

  // Return the chat reply
  absl::StatusOr<std::string> message_reply =
      AsString(response["choices"][0]["message"]["content"]);
  if (!message_reply.ok()) {
    return message_reply.status();
  }
  v0::Value response_pb;
  response_pb.set_str(message_reply.value());
  return response_pb;
}

}  // namespace genc
