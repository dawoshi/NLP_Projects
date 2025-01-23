package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_binary(
    name = "client",
    srcs = ["client.cc"],
    deps = [
        "//third_party/llama.cpp/base:read_file",
        "//third_party/llama.cpp/interop/confidential_computing:attestation",
        "//third_party/llama.cpp/interop/oak:client",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:remote_executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/runtime:threading",
        "//third_party/llama.cpp/proto/v0:executor_cc_grpc_proto",
        "//third_party/llama.cpp/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "server",
    srcs = ["server.cc"],
    deps = [
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/modules/worker:run_server",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
