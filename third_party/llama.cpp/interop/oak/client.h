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

#ifndef GENC_CC_INTEROP_OAK_CLIENT_H_
#define GENC_CC_INTEROP_OAK_CLIENT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "third_party/llama.cpp/proto/v0/executor.grpc.pb.h"
#include "include/grpcpp/channel.h"
#include "cc/attestation/verification/attestation_verifier.h"

namespace genc {
namespace interop {
namespace oak {

// Creates an Oak client that implements the GenC `Executor` stub interface.
absl::StatusOr<std::unique_ptr<v0::Executor::StubInterface>> CreateClient(
    std::shared_ptr<grpc::Channel> channel,
    std::shared_ptr<::oak::attestation::verification::AttestationVerifier>
        attestation_verifier,
    bool debug);

}  // namespace oak
}  // namespace interop
}  // namespace genc

#endif  // GENC_CC_INTEROP_OAK_CLIENT_H_
