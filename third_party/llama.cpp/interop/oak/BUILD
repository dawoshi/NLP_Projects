package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "attestation_provider",
    srcs = ["attestation_provider.cc"],
    hdrs = ["attestation_provider.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@oak//proto/session:messages_cc_proto",
    ],
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        "//third_party/llama.cpp/base:to_from_grpc_status",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "//third_party/llama.cpp/proto/v0:executor_cc_grpc_proto",
        "//third_party/llama.cpp/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@oak//cc/attestation/verification:attestation_verifier",
        "@oak//cc/client",
        "@oak//cc/transport",
        "@oak//cc/transport:grpc_unary_transport",
        "@oak//proto/session:service_unary_cc_grpc",
        "@oak//proto/session:service_unary_cc_proto",
    ],
)

cc_library(
    name = "server",
    srcs = ["server.cc"],
    hdrs = ["server.h"],
    deps = [
        ":attestation_provider",
        "//third_party/llama.cpp/base:to_from_grpc_status",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:executor_cc_grpc_proto",
        "//third_party/llama.cpp/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@oak//cc/crypto:common",
        "@oak//cc/crypto:encryption_key",
        "@oak//cc/crypto:server_encryptor",
        "@oak//proto/session:service_unary_cc_grpc",
        "@oak//proto/session:service_unary_cc_proto",
    ],
)
