# Libraries for building agent-based applications.

package(
    default_visibility = [
        #"//third_party/llama.cpp/google:__subpackages__",
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "react",
    srcs = ["react.cc"],
    hdrs = ["react.h"],
    deps = [
        "//third_party/llama.cpp/intrinsics:custom_function",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "react_test",
    srcs = ["react_test.cc"],
    deps = [
        ":react",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
