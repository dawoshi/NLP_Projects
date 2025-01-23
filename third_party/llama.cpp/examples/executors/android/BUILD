package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "android_executor_stacks",
    srcs = ["android_executor_stacks.cc"],
    hdrs = ["android_executor_stacks.h"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/interop/backends:llamacpp",
        "//third_party/llama.cpp/interop/backends/java:google_ai",
        "//third_party/llama.cpp/interop/backends/java:mediapipe_llm_inference",
        "//third_party/llama.cpp/interop/backends/java:open_ai",
        "//third_party/llama.cpp/interop/backends/java:wolfram_alpha_handler",
        "//third_party/llama.cpp/interop/networking:cronet_based_android_http_client",
        "//third_party/llama.cpp/intrinsics:handler_sets",
        "//third_party/llama.cpp/intrinsics:model_inference",
        "//third_party/llama.cpp/modules/agents:react",
        "//third_party/llama.cpp/modules/retrieval:local_cache",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:executor_stacks",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/java/src/jni:copy_jni_hdr_lib",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@bazel_tools//tools/jdk:jni",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
