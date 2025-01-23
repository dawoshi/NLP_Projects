package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "wolfram_alpha",
    srcs = ["wolfram_alpha.cc"],
    hdrs = ["wolfram_alpha.h"],
    deps = [
        "//third_party/llama.cpp/intrinsics:intrinsic_uris",
        "//third_party/llama.cpp/runtime:intrinsic_handler",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@curl",
    ],
)

cc_library(
    name = "curl_client",
    srcs = ["curl_client.cc"],
    hdrs = ["curl_client.h"],
    deps = [
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@curl",
    ],
)
