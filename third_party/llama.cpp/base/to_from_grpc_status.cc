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

#include "absl/status/status.h"
#include "third_party/llama.cpp/base/to_from_grpc_status.h"
#include "include/grpcpp/support/status.h"

namespace genc {

grpc::Status AbslToGrpcStatus(absl::Status status) {
  if (status.ok()) {
    return grpc::Status::OK;
  } else {
    return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                        std::string(status.message()));
  }
}

absl::Status GrpcToAbslStatus(grpc::Status status) {
  if (status.ok()) {
    return absl::OkStatus();
  } else {
    return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                        status.error_message());
  }
}

}  // namespace genc
