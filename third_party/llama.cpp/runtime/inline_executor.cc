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

#include "third_party/llama.cpp/runtime/inline_executor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/llama.cpp/runtime/concurrency.h"
#include "third_party/llama.cpp/runtime/executor.h"
#include "third_party/llama.cpp/runtime/intrinsic_handler.h"
#include "third_party/llama.cpp/runtime/status_macros.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"

namespace genc {
namespace {

class ExecutorValue {
 public:
  explicit ExecutorValue(const std::shared_ptr<v0::Value>& value_pb)
      : value_(value_pb) {}

  ExecutorValue(const ExecutorValue& other) = default;
  ExecutorValue(ExecutorValue&& other) : value_(std::move(other.value_)) {}
  ExecutorValue& operator=(ExecutorValue&& other) {
    this->value_ = std::move(other.value_);
    return *this;
  }

  const v0::Value& value() const { return *value_; }

 private:
  ExecutorValue() = delete;

  std::shared_ptr<v0::Value> value_;
};

using ValueFuture =
    std::shared_ptr<FutureInterface<absl::StatusOr<ExecutorValue>>>;

absl::StatusOr<ExecutorValue> Wait(ValueFuture value_future) {
  return GENC_TRY(value_future->Get());
}

// Executor that specializes in handling inline intrinsics.
class InlineExecutor : public ExecutorBase<ValueFuture>,
                       public InlineIntrinsicHandlerInterface::Context {
 public:
  explicit InlineExecutor(
      std::shared_ptr<IntrinsicHandlerSet> handler_set,
      std::shared_ptr<ConcurrencyInterface> concurrency_interface)
      : concurrency_interface_(std::move(concurrency_interface)),
        intrinsic_handlers_(std::move(handler_set)) {}

  ~InlineExecutor() override { ClearTracked(); }

  std::shared_ptr<ConcurrencyInterface> concurrency_interface() const override {
    return concurrency_interface_;
  }

  absl::string_view ExecutorName() final {
    static constexpr absl::string_view kExecutorName = "InlineExecutor";
    return kExecutorName;
  }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& val_pb) final {
    return concurrency_interface_->RunAsync(
        [val_pb]() -> absl::StatusOr<ExecutorValue> {
          return ExecutorValue(std::make_shared<v0::Value>(val_pb));
        });
  }

  absl::Status Materialize(ValueFuture value_future, v0::Value* val_pb) final {
    ExecutorValue value = GENC_TRY(Wait(value_future));
    if (val_pb != nullptr) {
      val_pb->CopyFrom(value.value());
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture func_future, std::optional<ValueFuture> arg_future) final {
    if (!arg_future.has_value()) {
      return absl::InvalidArgumentError("An argument is always required.");
    }
    return concurrency_interface_->RunAsync(
        [function = std::move(func_future), argument = std::move(arg_future),
         this]() -> absl::StatusOr<ExecutorValue> {
          ExecutorValue fn = GENC_TRY(Wait(function));
          ExecutorValue arg = GENC_TRY(Wait(argument.value()));
          if (!fn.value().has_intrinsic()) {
            return absl::InvalidArgumentError(
                absl::StrCat("Unsupported function type: ",
                    fn.value().DebugString()));
          }
          const v0::Intrinsic& intr_pb = fn.value().intrinsic();
          const IntrinsicHandler* const handler =
              GENC_TRY(intrinsic_handlers_->GetHandler(intr_pb.uri()));
          GENC_TRY(handler->CheckWellFormed(intr_pb));
          const InlineIntrinsicHandlerInterface* const interface =
              GENC_TRY(IntrinsicHandler::GetInlineInterface(handler));
          std::shared_ptr<v0::Value> result = std::make_shared<v0::Value>();
          GENC_TRY(
              interface->ExecuteCall(intr_pb, arg.value(), result.get(), this));
          return ExecutorValue(result);
        });
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> member_futures) final {
    return concurrency_interface_->RunAsync(
        [member_futures]() -> absl::StatusOr<ExecutorValue> {
          std::shared_ptr<v0::Value> result_pb = std::make_shared<v0::Value>();
          auto elements = result_pb->mutable_struct_()->mutable_element();
          for (const auto& member_future : member_futures) {
            ExecutorValue val = GENC_TRY(Wait(member_future));
            elements->Add()->CopyFrom(val.value());
          }
          return ExecutorValue(result_pb);
        });
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value_future,
                                              const uint32_t index) final {
    return concurrency_interface_->RunAsync(
        [value_future, index]() -> absl::StatusOr<ExecutorValue> {
          ExecutorValue val = GENC_TRY(Wait(value_future));
          if (!val.value().has_struct_()) {
            return absl::InvalidArgumentError(
                absl::StrCat("Not a struct: ", val.value().DebugString()));
          }
          if (val.value().struct_().element_size() <= index) {
            return absl::OutOfRangeError("Selection index out of bounds.");
          }
          return ExecutorValue(std::make_shared<v0::Value>(
              val.value().struct_().element(index)));
        });
  }

 private:
  const std::shared_ptr<ConcurrencyInterface> concurrency_interface_;
  const std::shared_ptr<IntrinsicHandlerSet> intrinsic_handlers_;
};

}  // namespace

absl::StatusOr<std::shared_ptr<Executor>> CreateInlineExecutor(
    std::shared_ptr<IntrinsicHandlerSet> handler_set,
    std::shared_ptr<ConcurrencyInterface> concurrency_interface) {
  return std::make_shared<InlineExecutor>(std::move(handler_set),
                                          std::move(concurrency_interface));
}

}  // namespace genc
