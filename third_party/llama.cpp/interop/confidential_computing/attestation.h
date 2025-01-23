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

#ifndef GENC_CC_INTEROP_CONFIDENTIAL_COMPUTING_ATTESTATION_H_
#define GENC_CC_INTEROP_CONFIDENTIAL_COMPUTING_ATTESTATION_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "third_party/llama.cpp/interop/networking/http_client_interface.h"
#include "third_party/llama.cpp/interop/oak/attestation_provider.h"
#include "cc/attestation/verification/attestation_verifier.h"
#include "tink/jwt/verified_jwt.h"


namespace genc {
namespace interop {
namespace confidential_computing {

absl::StatusOr<std::string> GetAttestationToken(
    std::shared_ptr<networking::HttpClientInterface> http_client = nullptr);

absl::StatusOr<std::string> GetAttestationToken(
    const std::string& audience,
    const std::string& nonce,
    std::shared_ptr<networking::HttpClientInterface> http_client = nullptr);

absl::StatusOr<crypto::tink::VerifiedJwt> DecodeAttestationToken(
    const std::string& token,
    const std::string& audience,
    const std::string& issuer,
    std::shared_ptr<networking::HttpClientInterface> http_client = nullptr);

class AttestationProvider : public oak::AttestationProvider {};

class AttestationVerifier
    : public ::oak::attestation::verification::AttestationVerifier {};

absl::StatusOr<std::shared_ptr<AttestationProvider>> CreateAttestationProvider(
    bool debug,
    std::shared_ptr<networking::HttpClientInterface> http_client = nullptr);

struct WorkloadProvenance {
  std::string container_image_reference;
  std::string container_image_digest;
};

absl::StatusOr<std::shared_ptr<AttestationVerifier>> CreateAttestationVerifier(
    WorkloadProvenance workload_provenance,
    bool debug,
    std::shared_ptr<networking::HttpClientInterface> http_client = nullptr);

}  // namespace confidential_computing
}  // namespace interop
}  // namespace genc

#endif  // GENC_CC_INTEROP_CONFIDENTIAL_COMPUTING_ATTESTATION_H_
