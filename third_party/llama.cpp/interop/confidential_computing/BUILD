package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "attestation",
    srcs = ["attestation.cc"],
    hdrs = ["attestation.h"],
    deps = [
        "//third_party/llama.cpp/interop/networking:curl_based_http_client",
        "//third_party/llama.cpp/interop/networking:http_client_interface",
        "//third_party/llama.cpp/interop/oak:attestation_provider",
        "//third_party/llama.cpp/runtime:status_macros",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "@oak//cc/attestation/verification:attestation_verifier",
        "@oak//proto/attestation:evidence_cc_proto",
        "@oak//proto/session:messages_cc_proto",
        "@tink_cc//tink:keyset_handle",
        "@tink_cc//tink/config:global_registry",
        "@tink_cc//tink/jwt:jwk_set_converter",
        "@tink_cc//tink/jwt:jwt_public_key_verify",
        "@tink_cc//tink/jwt:jwt_signature_config",
        "@tink_cc//tink/jwt:jwt_validator",
        "@tink_cc//tink/jwt:verified_jwt",
    ],
)
