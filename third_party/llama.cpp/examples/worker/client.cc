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

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/llama.cpp/base/read_file.h"
#include "third_party/llama.cpp/interop/confidential_computing/attestation.h"
#include "third_party/llama.cpp/interop/oak/client.h"
#include "third_party/llama.cpp/runtime/executor.h"
#include "third_party/llama.cpp/runtime/remote_executor.h"
#include "third_party/llama.cpp/runtime/runner.h"
#include "third_party/llama.cpp/runtime/status_macros.h"
#include "third_party/llama.cpp/runtime/threading.h"
#include "third_party/llama.cpp/proto/v0/executor.grpc.pb.h"
#include "third_party/llama.cpp/proto/v0/executor.pb.h"
#include "include/grpcpp/channel.h"
#include "include/grpcpp/client_context.h"
#include "include/grpcpp/create_channel.h"
#include "include/grpcpp/security/credentials.h"
#include "include/grpcpp/support/channel_arguments.h"
#include "include/grpcpp/support/status.h"
#include "google/protobuf/text_format.h"

// An example worker client that interacts with a worker server over gRPC.
//
// NOTE: This client has not been fully implemented yet.
//
// Example usage:
//   bazel run genc/cc/examples/worker:client -- \
//     --server=<address> --ir=<ir_ascii_string> --prompt=<prompt_string>
//
// For a secure gRPC connection over SSL/TLS, you can additionally specify:
//   --ssl --cert=<path-to-cert> --target_override=<expected-target-name>

ABSL_FLAG(std::string, server, "", "The address of the worker server.");
ABSL_FLAG(std::string, ir, "", "The IR string in the ASCII form.");
ABSL_FLAG(std::string, prompt, "", "The prompt string.");
ABSL_FLAG(bool, oak, false, "Whether to use project Oak for communication.");
ABSL_FLAG(bool, ssl, false, "Whether to use SSL for communication.");
ABSL_FLAG(std::string, cert, "", "The path to the root cert.");
ABSL_FLAG(std::string, target_override, "", "The expected target name.");
ABSL_FLAG(bool, debug, false, "Whether to print debug output.");
ABSL_FLAG(std::string, image_reference, "", "The container image reference.");
ABSL_FLAG(std::string, image_digest, "", "The container image digest.");

namespace genc {

std::string CreateServerAddress() {
  return absl::GetFlag(FLAGS_server);
}

std::shared_ptr<grpc::ChannelCredentials> CreateChannelCredentials() {
  if (absl::GetFlag(FLAGS_ssl)) {
    grpc::SslCredentialsOptions options;
    std::string cert = absl::GetFlag(FLAGS_cert);
    if (!cert.empty()) {
      options.pem_root_certs = ReadFile(cert);
    }
    return grpc::SslCredentials(options);
  }
  return grpc::InsecureChannelCredentials();
}

std::shared_ptr<grpc::Channel> CreateChannel() {
  std::string server_address = CreateServerAddress();
  std::shared_ptr<grpc::ChannelCredentials> creds = CreateChannelCredentials();
  std::string target_override = absl::GetFlag(FLAGS_target_override);
  if (!target_override.empty()) {
    grpc::ChannelArguments args;
    args.SetSslTargetNameOverride(target_override);
    return grpc::CreateCustomChannel(server_address, creds, args);
  }
  return grpc::CreateChannel(server_address, creds);
}

absl::StatusOr<v0::Value> CreateFn() {
  std::string ir_ascii_string = absl::GetFlag(FLAGS_ir);
  if (ir_ascii_string.empty()) {
    return absl::InvalidArgumentError("IR is required.");
  }
  v0::Value value;
  if (!::google::protobuf::TextFormat::ParseFromString(ir_ascii_string, &value)) {
    return absl::InvalidArgumentError("Could not parse the IR.");
  }
  return value;
}

absl::StatusOr<v0::Value> CreateArg() {
  std::string prompt_string = absl::GetFlag(FLAGS_prompt);
  if (prompt_string.empty()) {
    return absl::InvalidArgumentError("Prompt is required.");
  }
  v0::Value value;
  value.set_str(prompt_string);
  return value;
}

absl::Status RunClient() {
  v0::Value func = GENC_TRY(CreateFn());
  v0::Value arg = GENC_TRY(CreateArg());
  std::shared_ptr<grpc::Channel> channel = CreateChannel();
  std::unique_ptr<v0::Executor::StubInterface> executor_stub;
  if (absl::GetFlag(FLAGS_oak)) {
    const bool debug = absl::GetFlag(FLAGS_debug);
    interop::confidential_computing::WorkloadProvenance provenance;
    provenance.container_image_reference = absl::GetFlag(FLAGS_image_reference);
    provenance.container_image_digest = absl::GetFlag(FLAGS_image_digest);
    auto verifier = GENC_TRY(
        interop::confidential_computing::CreateAttestationVerifier(
            provenance, debug));
    executor_stub = GENC_TRY(
        interop::oak::CreateClient(channel, verifier, debug));
  } else {
    executor_stub = v0::Executor::NewStub(channel);
  }
  std::shared_ptr<Executor> executor = GENC_TRY(CreateRemoteExecutor(
      std::move(executor_stub), CreateThreadBasedConcurrencyManager()));
  Runner runner = GENC_TRY(Runner::Create(func, executor));
  v0::Value result = GENC_TRY(runner.Run(arg));
  std::cout << "\n" << result.DebugString() << "\n\n";
  return absl::OkStatus();
}

}  // namespace genc

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = genc::RunClient();
  if (!status.ok()) {
    std::cout << "Client failed with status: " << status << "\n";
  }
  return 0;
}
