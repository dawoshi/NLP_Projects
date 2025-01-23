package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "run_server",
    srcs = ["run_server.cc"],
    hdrs = ["run_server.h"],
    deps = [
        "//third_party/llama.cpp/base:read_file",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/interop/confidential_computing:attestation",
        "//third_party/llama.cpp/interop/oak:server",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:executor_service",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "//third_party/llama.cpp/proto/v0:executor_cc_grpc_proto",
        "//third_party/llama.cpp/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_reflection",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@oak//proto/session:service_unary_cc_grpc",
    ],
)
