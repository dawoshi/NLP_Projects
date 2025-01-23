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

#include <stdlib.h>

#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "third_party/llama.cpp/authoring/constructor.h"
#include "third_party/llama.cpp/authoring/smart_chain.h"
#include "third_party/llama.cpp/examples/executors/executor_stacks.h"
#include "third_party/llama.cpp/runtime/executor.h"
#include "third_party/llama.cpp/runtime/runner.h"
#include "third_party/llama.cpp/runtime/status_macros.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

// Demo calling Google AI Studio Gemini API as backend.
// To get API key follow https://ai.google.dev/
// bazle run genc/cc/examples:run_gemini_on_ai_studio --
// --api_key=

ABSL_FLAG(std::string, api_key, "", "Gemini API Auth Token.");

namespace genc {

absl::StatusOr<v0::Value> RunGeminiOnAIStudio(std::string query) {
  std::string api_key = absl::GetFlag(FLAGS_api_key);
  std::shared_ptr<Executor> executor = GENC_TRY(CreateDefaultExecutor());
  std::string end_point =
      "https://generativelanguage.googleapis.com/v1beta/models/"
      "gemini-pro:generateContent";
  std::string json_request_template =
      R"pb(
      {
        "contents":[{"role":"user","parts":[{"text":"PROMPT REPLACES THIS"}]}],
        "safety_settings":[{
          "category":"HARM_CATEGORY_SEXUALLY_EXPLICIT",
          "threshold":"BLOCK_LOW_AND_ABOVE"}],
          "generation_config":{
          "temperature":0.9, "topP":1.0, "maxOutputTokens":1024}
      }
    )pb";

  v0::Value model_config = GENC_TRY(
      CreateRestModelConfigWithJsonRequestTemplate(
          end_point, api_key, json_request_template));
  v0::Value model_call =
      GENC_TRY(CreateModelInferenceWithConfig("/cloud/gemini", model_config));
  Runner runner = GENC_TRY(Runner::Create(executor));
  v0::Value arg;
  arg.set_str(query);
  return runner.Run(model_call, arg);
}
}  // namespace genc

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  std::string query = "Write a story about a magic backpack.";
  genc::v0::Value output =
      genc::RunGeminiOnAIStudio(query).value();
  std::cout << output.DebugString() << "\n";
  return 0;
}
