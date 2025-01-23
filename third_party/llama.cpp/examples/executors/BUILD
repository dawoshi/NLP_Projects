load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pybind_extension(
    name = "executor_bindings",
    srcs = ["executor_bindings.cc"],
    tags = ["generated_py_module=genc.examples.executors"],
    deps = [
        ":executor_stacks",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

cc_library(
    name = "executor_stacks",
    srcs = ["executor_stacks.cc"],
    hdrs = ["executor_stacks.h"],
    deps = [
        "//third_party/llama.cpp/interop/backends:google_ai",
        "//third_party/llama.cpp/interop/backends:llamacpp",
        "//third_party/llama.cpp/intrinsics:handler_sets",
        "//third_party/llama.cpp/modules/agents:react",
        "//third_party/llama.cpp/modules/parsers:gemini_parser",
        "//third_party/llama.cpp/modules/retrieval:local_cache",
        "//third_party/llama.cpp/modules/tools:wolfram_alpha",
        "//third_party/llama.cpp/runtime:concurrency",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:executor_stacks",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/runtime:threading",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "executor_stacks_test",
    srcs = ["executor_stacks_test.cc"],
    deps = [
        ":executor_stacks",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "java_executor_stacks",
    srcs = ["java_executor_stacks.cc"],
    hdrs = ["java_executor_stacks.h"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/interop/backends:llamacpp",
        "//third_party/llama.cpp/interop/backends/java:google_ai",
        "//third_party/llama.cpp/interop/backends/java:open_ai",
        "//third_party/llama.cpp/interop/backends/java:wolfram_alpha_handler",
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
