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

#ifndef GENC_CC_AUTHORING_CONSTRUCTOR_H_
#define GENC_CC_AUTHORING_CONSTRUCTOR_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/llama.cpp/base/computation.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

namespace genc {

// Given a list of functions [f, g, ...] create a chain g(f(...)). Compared to
// CreateBasicChain, this chain can contain break point as part of the chain.
absl::StatusOr<v0::Value> CreateBreakableChain(std::vector<v0::Value> fns_list);

// Constructs a function call.
absl::StatusOr<v0::Value> CreateCall(v0::Value fn, v0::Value arg);

// Creates a conditional expression. condition evaluates to a boolean, if true,
// positive_branch will be executed, else negative_branch.
absl::StatusOr<v0::Value> CreateConditional(v0::Value condition,
                                            v0::Value positive_branch,
                                            v0::Value negative_branch);

// Creates a conditional expression with parameterized input.
absl::StatusOr<v0::Value> CreateLambdaForConditional(v0::Value condition,
                                                     v0::Value positive_branch,
                                                     v0::Value negative_branch);

// Returns a custom function proto with the given fn URI.
absl::StatusOr<v0::Value> CreateCustomFunction(absl::string_view fn_uri);

// Creates a fallback expression from a given list of functions. The first
// successful one is the result; if failed, keep going down the list.
absl::StatusOr<v0::Value> CreateFallback(std::vector<v0::Value> function_list);

// Creates an InjaTemplate.
absl::StatusOr<v0::Value> CreateInjaTemplate(absl::string_view template_str);

// Given arg_name & computation body create a Lambda that applies a computation
// to the provided argument.
absl::StatusOr<v0::Value> CreateLambda(absl::string_view arg_name,
                                       v0::Value body);

// Creates a Logger, it takes an input logs it and returns the original input.
absl::StatusOr<v0::Value> CreateLogger();

// Creates a logical negation computation.
absl::StatusOr<v0::Value> CreateLogicalNot();

// Returns a model inference proto with the given model URI.
absl::StatusOr<v0::Value> CreateModelInference(absl::string_view model_uri);

// Returns a model config for models behind REST endpoint.
// api_key is optional.
absl::StatusOr<v0::Value> CreateRestModelConfig(std::string endpoint,
                                                std::string api_key = "");

// Returns a model config for models behind REST endpoint,
// including a JSON request template.
absl::StatusOr<v0::Value> CreateRestModelConfigWithJsonRequestTemplate(
    std::string endpoint, std::string api_key,
    std::string json_request_template);

absl::StatusOr<v0::Value> CreateLlamaCppConfig(std::string model_path,
                                               int num_threads = 1,
                                               int max_tokens = 32);
// Returns a model inference proto with the given model URI and model config.
absl::StatusOr<v0::Value> CreateModelInferenceWithConfig(
    absl::string_view model_uri, v0::Value model_config);

// Creates a parallel map that applies map_fn to a all input values.
absl::StatusOr<v0::Value> CreateParallelMap(v0::Value map_fn);

// Creates a prompt template computation with the given template string.
// NOTE: Please use `CreatePromptTemplateWithParameters` instead for building
/// multivariate prompt templates. The use of this function with multivariate
// prompt templates is deprecated, and will soon be removed.
absl::StatusOr<v0::Value> CreatePromptTemplate(absl::string_view template_str);

// Creates a prompt template computation with the given template string.
absl::StatusOr<v0::Value> CreatePromptTemplateWithParameters(
    absl::string_view template_str,
    std::vector<absl::string_view> parameter_list);

// Given arg_name returns a Reference argument. Useful for parameterizing the
// computation.
absl::StatusOr<v0::Value> CreateReference(absl::string_view arg_name);

// Creates a partial regex matcher with the given template string.
absl::StatusOr<v0::Value> CreateRegexPartialMatch(
    absl::string_view pattern_str);

// Returns a repeat proto which will repeat body_fn for num_steps, sequentially,
// the output of the current step is the input to next iteration.
absl::StatusOr<v0::Value> CreateRepeat(int num_steps, v0::Value body_fn);

// Creates a for loop with the given num_steps, and a sequence of body_fns,
// which will be executed sequentially each iteration, if any the function
// inside body_fns is a conditional and evaluates to be true, loop will break
// and return the state before the conditional; if it evaluates to be false,
// chain will continue execution.
absl::StatusOr<v0::Value> CreateRepeatedConditionalChain(
    int num_steps, std::vector<v0::Value> body_fns);

// Constructs a selection to pick the i-th element from a source.struct_.
absl::StatusOr<v0::Value> CreateSelection(v0::Value source, int index);

// Given a list of functions [f, g, ...] create a chain g(f(...)).
absl::StatusOr<v0::Value> CreateSerialChain(
    std::vector<v0::Value> function_list);

// Constructs a struct from named values
absl::StatusOr<v0::Value> CreateStruct(std::vector<v0::Value> value_list);

// Adds a label (name) to a value.
absl::StatusOr<v0::Value> CreateNamedValue(absl::string_view label,
                                           v0::Value unlabeled_value);

// Creates a while loop with the given condition_fn, body_fn.
absl::StatusOr<v0::Value> CreateWhile(v0::Value condition_fn,
                                      v0::Value body_fn);

// Creates an operator that calls a rest endpoint using curl.
absl::StatusOr<v0::Value> CreateRestCall(absl::string_view uri,
                                         absl::string_view api_key = "",
                                         absl::string_view method = "POST");

// Creates an operator that calls WolframAlpha.
absl::StatusOr<v0::Value> CreateWolframAlpha(absl::string_view appid);

// Populate the computation.proto in `intrinsics` to represent a model
// inference call to a model `model_uri`.
void SetModelInference(v0::Value& computation, absl::string_view model_uri);

// Convenient methods to convert various input types to Value Proto.
// Converts int to Value Proto.
v0::Value ToValue(int arg);

// Converts float to Value Proto.
v0::Value ToValue(float arg);

// Converts string to Value Proto.
v0::Value ToValue(std::string arg);

// Converts bool to Value Proto.
v0::Value ToValue(bool arg);

// Converts binary to Value Proto.
v0::Value ToValue(absl::string_view arg);

// Converts Computation to Value Proto.
v0::Value ToValue(Computation arg);

// Converts vector of Value to Value Proto.
v0::Value ToValue(std::vector<v0::Value>& arg);

template <typename... Args>
v0::Value ToValue(Args&&... args) {
  std::vector<v0::Value> values = {ToValue(std::forward<Args>(args))...};
  return ToValue(values);
}

}  // namespace genc

#endif  // GENC_CC_AUTHORING_CONSTRUCTOR_H_
