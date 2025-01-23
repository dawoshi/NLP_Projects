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

#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/statusor.h"
#include "third_party/llama.cpp/authoring/constructor.h"
#include "third_party/llama.cpp/runtime/executor_stacks.h"
#include "third_party/llama.cpp/runtime/runner.h"
#include "third_party/llama.cpp/runtime/status_macros.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

namespace genc {
namespace {

absl::StatusOr<v0::Value> MakeTestArg() {
  v0::Value foo_val, bar_val;
  foo_val.set_str("XXX");
  bar_val.set_str("YYY");
  std::vector<v0::Value> elements;
  elements.push_back(GENC_TRY(CreateNamedValue("foo", foo_val)));
  elements.push_back(GENC_TRY(CreateNamedValue("bar", bar_val)));
  return CreateStruct(elements);
}

absl::StatusOr<v0::Value> MakeTestArgWithoutLabels() {
  v0::Value foo_val, bar_val;
  foo_val.set_str("XXX");
  bar_val.set_str("YYY");
  std::vector<v0::Value> elements;
  elements.push_back(foo_val);
  elements.push_back(bar_val);
  return CreateStruct(elements);
}

absl::StatusOr<v0::Value> RunFuncOnArg(v0::Value func, v0::Value arg) {
  return GENC_TRY(Runner::Create(
      func, GENC_TRY(CreateDefaultLocalExecutor()))).Run(arg);
}

TEST(MultivariatePromptTemplateTest, Simple) {
  absl::StatusOr<v0::Value> func = CreatePromptTemplate(
      "A template in which a foo is {foo} and a bar is {bar}.");
  ASSERT_TRUE(func.ok());

  absl::StatusOr<v0::Value> arg = MakeTestArg();
  ASSERT_TRUE(arg.ok());
  absl::StatusOr<v0::Value> result = RunFuncOnArg(func.value(), arg.value());
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.value().str(),
            "A template in which a foo is XXX and a bar is YYY.");
}

absl::StatusOr<v0::Value> WrapInLambda(v0::Value func) {
  const std::string arg_name = "arg";
  v0::Value arg_ref = GENC_TRY(CreateReference(arg_name));
  std::vector<v0::Value> elements;
  elements.push_back(GENC_TRY(CreateNamedValue(
      "foo", GENC_TRY(CreateSelection(arg_ref, 0)))));
  elements.push_back(GENC_TRY(CreateNamedValue(
      "bar", GENC_TRY(CreateSelection(arg_ref, 1)))));
  return CreateLambda(arg_name, GENC_TRY(CreateCall(
      func, GENC_TRY(CreateStruct(elements)))));
}

TEST(MultivariatePromptTemplateTest, WrappedInLambda) {
  absl::StatusOr<v0::Value> func = CreatePromptTemplate(
      "A template in which a foo is {foo} and a bar is {bar}.");
  ASSERT_TRUE(func.ok());
  func = WrapInLambda(func.value());
  ASSERT_TRUE(func.ok());

  absl::StatusOr<v0::Value> arg = MakeTestArg();
  ASSERT_TRUE(arg.ok());
  absl::StatusOr<v0::Value> result = RunFuncOnArg(func.value(), arg.value());
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.value().str(),
            "A template in which a foo is XXX and a bar is YYY.");
}

TEST(MultivariatePromptTemplateTest, WithParameters) {
  absl::StatusOr<v0::Value> func = CreatePromptTemplateWithParameters(
      "A template in which a foo is {foo} and a bar is {bar}.",
      {"foo", "bar"});
  ASSERT_TRUE(func.ok());
  func = WrapInLambda(func.value());
  ASSERT_TRUE(func.ok());

  absl::StatusOr<v0::Value> arg = MakeTestArgWithoutLabels();
  ASSERT_TRUE(arg.ok());
  absl::StatusOr<v0::Value> result = RunFuncOnArg(func.value(), arg.value());
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.value().str(),
            "A template in which a foo is XXX and a bar is YYY.");
}

TEST(MultivariatePromptTemplateTest, WithParametersInOppositeOrder) {
  absl::StatusOr<v0::Value> func = CreatePromptTemplateWithParameters(
      "A template in which a foo is {foo} and a bar is {bar}.",
      {"bar", "foo"});
  ASSERT_TRUE(func.ok());
  func = WrapInLambda(func.value());
  ASSERT_TRUE(func.ok());

  absl::StatusOr<v0::Value> arg = MakeTestArgWithoutLabels();
  ASSERT_TRUE(arg.ok());
  absl::StatusOr<v0::Value> result = RunFuncOnArg(func.value(), arg.value());
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.value().str(),
            "A template in which a foo is YYY and a bar is XXX.");
}

}  // namespace
}  // namespace genc
