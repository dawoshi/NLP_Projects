package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "checking",
    srcs = ["checking.cc"],
    hdrs = ["checking.h"],
    deps = [
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "checking_test",
    timeout = "short",
    srcs = ["checking_test.cc"],
    deps = [
        ":checking",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "inference",
    srcs = ["inference.cc"],
    hdrs = ["inference.h"],
    deps = [
        ":checking",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "inference_test",
    timeout = "short",
    srcs = ["inference_test.cc"],
    deps = [
        ":inference",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
