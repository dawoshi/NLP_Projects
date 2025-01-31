load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "conditional",
    srcs = ["conditional.cc"],
    hdrs = ["conditional.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "delegate",
    srcs = ["delegate.cc"],
    hdrs = ["delegate.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fallback",
    srcs = ["fallback.cc"],
    hdrs = ["fallback.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "inja_template",
    srcs = ["inja_template.cc"],
    hdrs = ["inja_template.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/modules/templates:inja_status_or",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "inja_template_test",
    srcs = ["inja_template_test.cc"],
    deps = [
        ":handler_sets",
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:inline_executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:threading",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "handler_sets",
    srcs = ["handler_sets.cc"],
    hdrs = ["handler_sets.h"],
    deps = [
        ":breakable_chain",
        ":conditional",
        ":confidential_computation",
        ":custom_function",
        ":delegate",
        ":fallback",
        ":inja_template",
        ":logger",
        ":logical_not",
        ":model_inference",
        ":model_inference_with_config",
        ":parallel_map",
        ":prompt_template",
        ":regex_partial_match",
        ":repeat",
        ":repeated_conditional_chain",
        ":rest_call",
        ":serial_chain",
        ":while",
        "//third_party/llama.cpp/interop/networking:http_client_interface",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
    ],
)

pybind_extension(
    name = "intrinsic_bindings",
    srcs = ["intrinsic_bindings.cc"],
    tags = ["generated_py_module=genc.intrinsics"],
    deps = [
        ":intrinsic_uris",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

cc_library(
    name = "intrinsic_uris",
    srcs = [],
    hdrs = ["intrinsic_uris.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "model_inference",
    srcs = ["model_inference.cc"],
    hdrs = ["model_inference.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "model_inference_with_config",
    srcs = ["model_inference_with_config.cc"],
    hdrs = ["model_inference_with_config.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "logical_not",
    srcs = ["logical_not.cc"],
    hdrs = ["logical_not.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "prompt_template",
    srcs = ["prompt_template.cc"],
    hdrs = ["prompt_template.h"],
    copts = ["-std=c++17"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "regex_partial_match",
    srcs = ["regex_partial_match.cc"],
    hdrs = ["regex_partial_match.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "repeat",
    srcs = ["repeat.cc"],
    hdrs = ["repeat.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "while",
    srcs = ["while.cc"],
    hdrs = ["while.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "repeated_conditional_chain",
    srcs = ["repeated_conditional_chain.cc"],
    hdrs = ["repeated_conditional_chain.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "breakable_chain",
    srcs = ["breakable_chain.cc"],
    hdrs = ["breakable_chain.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "custom_function",
    srcs = ["custom_function.cc"],
    hdrs = ["custom_function.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "parallel_map",
    srcs = ["parallel_map.cc"],
    hdrs = ["parallel_map.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "serial_chain",
    srcs = ["serial_chain.cc"],
    hdrs = ["serial_chain.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "rest_call",
    srcs = ["rest_call.cc"],
    hdrs = ["rest_call.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/modules/tools:curl_client",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "confidential_computation",
    srcs = ["confidential_computation.cc"],
    hdrs = ["confidential_computation.h"],
    deps = [
        ":intrinsic_uris",
        "//third_party/llama.cpp/interop/confidential_computing:attestation",
        "//third_party/llama.cpp/interop/networking:curl_based_http_client",
        "//third_party/llama.cpp/interop/networking:http_client_interface",
        "//third_party/llama.cpp/interop/oak:client",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:remote_executor",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
