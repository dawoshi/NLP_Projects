package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "local_cache",
    srcs = ["local_cache.cc"],
    hdrs = ["local_cache.h"],
    deps = [
        "//third_party/llama.cpp/intrinsics:custom_function",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "local_cache_test",
    srcs = ["local_cache_test.cc"],
    deps = [
        ":local_cache",
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/intrinsics:handler_sets",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:inline_executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:threading",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bi_map",
    hdrs = ["bi_map.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "bi_map_test",
    srcs = ["bi_map_test.cc"],
    deps = [
        ":bi_map",
        "@com_google_googletest//:gtest_main",
    ],
)
