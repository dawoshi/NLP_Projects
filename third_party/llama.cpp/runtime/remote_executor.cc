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

#include "third_party/llama.cpp/runtime/remote_executor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/llama.cpp/base/to_from_grpc_status.h"
#include "third_party/llama.cpp/runtime/concurrency.h"
#include "third_party/llama.cpp/runtime/executor.h"
#include "third_party/llama.cpp/runtime/status_macros.h"
#include "third_party/llama.cpp/proto/v0/computation.pb.h"
#include "third_party/llama.cpp/proto/v0/executor.grpc.pb.h"
#include "third_party/llama.cpp/proto/v0/executor.pb.h"
#include "include/grpcpp/client_context.h"
#include "include/grpcpp/support/status.h"

namespace genc {
namespace {

using ExecutorStub = v0::Executor::StubInterface;

class ExecutorValue {
 public:
  ExecutorValue(std::shared_ptr<ConcurrencyInterface> concurrency_interface,
                std::shared_ptr<ExecutorStub> executor_stub,
                v0::ValueRef value_ref)
      : concurrency_interface_(concurrency_interface),
        executor_stub_(executor_stub),
        value_ref_(std::move(value_ref)) {}

  ~ExecutorValue() {
    concurrency_interface_->RunAsync([executor_stub = executor_stub_,
                                      value_ref = value_ref_]() -> bool {
      v0::DisposeRequest request;
      v0::DisposeResponse response;
      grpc::ClientContext context;
      *request.add_value_ref() = value_ref;
      const grpc::Status status =
          executor_stub->Dispose(&context, request, &response);
      if (!status.ok()) {
        // Silently ignore for now...
      }
      return true;
    });
  }

  const v0::ValueRef& ref() const { return value_ref_; }

 private:
  std::shared_ptr<ConcurrencyInterface> concurrency_interface_;
  const std::shared_ptr<ExecutorStub> executor_stub_;
  const v0::ValueRef value_ref_;
};

using ValueFuture = std::shared_ptr<
    FutureInterface<absl::StatusOr<std::shared_ptr<ExecutorValue>>>>;

absl::StatusOr<std::shared_ptr<ExecutorValue>> Wait(ValueFuture value_future) {
  return GENC_TRY(value_future->Get());
}

class RemoteExecutor : public ExecutorBase<ValueFuture> {
 public:
  explicit RemoteExecutor(
      std::unique_ptr<ExecutorStub> stub,
      std::shared_ptr<ConcurrencyInterface> concurrency_interface)
      : executor_stub_(stub.release()),
        concurrency_interface_(concurrency_interface) {}

  ~RemoteExecutor() override = default;

  absl::string_view ExecutorName() final {
    static constexpr absl::string_view kExecutorName = "RemoteExecutor";
    return kExecutorName;
  }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& val_pb) final {
    return concurrency_interface_->RunAsync(
        [val_pb, this, this_keepalive = shared_from_this()]()
            -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
          grpc::ClientContext client_context;
          v0::CreateValueRequest request;
          v0::CreateValueResponse response;
          *request.mutable_value() = val_pb;
          grpc::Status status = executor_stub_->CreateValue(
              &client_context, request, &response);
          GENC_TRY(GrpcToAbslStatus(status));
          return std::make_shared<ExecutorValue>(
              concurrency_interface_, executor_stub_,
              std::move(response.value_ref()));
        });
  }

  absl::Status Materialize(ValueFuture value_future, v0::Value* val_pb) final {
    std::shared_ptr<ExecutorValue> value_ref = GENC_TRY(Wait(value_future));
    grpc::ClientContext client_context;
    v0::MaterializeRequest request;
    v0::MaterializeResponse response;
    *request.mutable_value_ref() = value_ref->ref();
    grpc::Status status = executor_stub_->Materialize(
        &client_context, request, &response);
    *val_pb = std::move(*response.mutable_value());
    return GrpcToAbslStatus(status);
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture func_future, std::optional<ValueFuture> arg_future) final {
    return concurrency_interface_->RunAsync(
        [func = std::move(func_future), arg = std::move(arg_future), this,
         this_keepalive = shared_from_this()]()
            -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
          std::shared_ptr<ExecutorValue> func_value = GENC_TRY(Wait(func));
          grpc::ClientContext context;
          v0::CreateCallRequest request;
          v0::CreateCallResponse response;
          *request.mutable_function_ref() = func_value->ref();
          if (arg.has_value()) {
            std::shared_ptr<ExecutorValue> arg_value =
                GENC_TRY(Wait(arg.value()));
            *request.mutable_argument_ref() = arg_value->ref();
          }
          grpc::Status status = executor_stub_->CreateCall(
              &context, request, &response);
          GENC_TRY(GrpcToAbslStatus(status));
          return std::make_shared<ExecutorValue>(
              concurrency_interface_, executor_stub_,
              std::move(response.result_ref()));
        });
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> member_futures) final {
    return concurrency_interface_->RunAsync(
        [elements = std::move(member_futures), this,
         this_keepalive = shared_from_this()]()
            -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
          grpc::ClientContext context;
          v0::CreateStructRequest request;
          v0::CreateStructResponse response;
          for (const ValueFuture& element : elements) {
            std::shared_ptr<ExecutorValue> element_value =
                GENC_TRY(Wait(element));
            *request.add_element_ref() = element_value->ref();
          }
          grpc::Status status = executor_stub_->CreateStruct(
              &context, request, &response);
          GENC_TRY(GrpcToAbslStatus(status));
          return std::make_shared<ExecutorValue>(
              concurrency_interface_, executor_stub_,
              std::move(response.struct_ref()));
        });
  }

  absl::StatusOr<ValueFuture> CreateSelection(
      ValueFuture value_future, const uint32_t index) final {
    return concurrency_interface_->RunAsync(
        [source = std::move(value_future), index = index, this,
         this_keepalive = shared_from_this()]()
            -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
          std::shared_ptr<ExecutorValue> source_value = GENC_TRY(Wait(source));
          grpc::ClientContext client_context;
          v0::CreateSelectionRequest request;
          v0::CreateSelectionResponse response;
          *request.mutable_source_ref() = source_value->ref();
          request.set_index(index);
          grpc::Status status = executor_stub_->CreateSelection(
              &client_context, request, &response);
          GENC_TRY(GrpcToAbslStatus(status));
          return std::make_shared<ExecutorValue>(
              concurrency_interface_, executor_stub_,
              std::move(response.selection_ref()));
        });
  }

 private:
  const std::shared_ptr<ExecutorStub> executor_stub_;
  const std::shared_ptr<ConcurrencyInterface> concurrency_interface_;
};

}  // namespace

absl::StatusOr<std::shared_ptr<Executor>> CreateRemoteExecutor(
    std::unique_ptr<v0::Executor::StubInterface> executor_stub,
    std::shared_ptr<ConcurrencyInterface> concurrency_interface) {
  return std::make_shared<RemoteExecutor>(std::move(executor_stub),
                                          std::move(concurrency_interface));
}

}  // namespace genc
