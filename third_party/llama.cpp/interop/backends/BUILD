# Libraries to access various LLM Backends. So user can stay backend agnostic.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "llamacpp",
    srcs = ["llamacpp.cc"],
    hdrs = ["llamacpp.h"],
    linkopts = [
        "-lm",
        "-ldl",
    ],
    deps = [
        "//third_party/llama.cpp/intrinsics:model_inference",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "//third_party/llama.cpp/llama_cpp_b3201:llama_cpp",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_library(
    name = "google_ai",
    srcs = ["google_ai.cc"],
    hdrs = ["google_ai.h"],
    deps = [
        "//third_party/llama.cpp/intrinsics:model_inference_with_config",
        "//third_party/llama.cpp/modules/parsers:gemini_parser",
        "//third_party/llama.cpp/modules/tools:curl_client",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
    ],
)
