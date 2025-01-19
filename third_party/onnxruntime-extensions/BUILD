cc_library(
    name = "includes",
    hdrs = glob([
        "include/*.h*",
        "include/custom_op/*.h*",
        "include/custom_op/tensor_tuple.inc",
    ]),
    strip_include_prefix = "include",
    deps = [
             "//third_party/onnxruntime:onnxruntime",
    ],
    alwayslink = True,
)

cc_library(
    name = "base",
    hdrs = glob([
        "base/*.h",
    ]),
    srcs = glob([
        "base/*.cc",
    ]),
    strip_include_prefix = "base",
    deps = [
        ":includes",
        "@gsl"
    ],
    alwayslink = True,
)


cc_library(
    name = "gpt2_tokenizer",
    hdrs = glob([
        "operators/tokenizer/bpe_*.h*",
        "operators/tokenizer/unicode*.h*",
        "operators/tokenizer/trietree.hpp",
        "operators/tokenizer/*",
    ]),
    srcs = glob([
        "operators/tokenizer/bpe_*.cc",
        "operators/tokenizer/unicode*.cc",
        "operators/tokenizer/tokenizers.cc",
    ]),
    strip_include_prefix = "operators/tokenizer",
    deps = [
        ":includes",
        ":base",
        "@nlohmann_json//:json",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cv2",
    hdrs = glob([
        "operators/vision/*.hpp",
        "operators/cv2/imgcodecs/*.hpp",
        "operators/cv2/imgproc/*.hpp",
    ]),
    srcs = glob([
        "operators/cv2/*.cc",
        "operators/vision/*.cc",
        "operators/*.cc",
    ]),
    strip_include_prefix = "operators",
    deps = [
        ":includes",
        ":base",
        "@opencv//:opencv",
        "@dlib//:libdlib",
    ],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "math",
    hdrs = glob([
        "operators/math/*.h*",
        "operators/math/dlib/*.h*",
    ]),
    srcs = glob([
        "operators/math/*.cc",
    ]),
    strip_include_prefix = "operators",
    deps = [
        ":includes",
        ":base",
    ],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "audio",
    hdrs = glob([
        "operators/audio/*.h*",
    ]),
    srcs = glob([
        "operators/audio/audio.cc",
    ]),
    strip_include_prefix = "operators",
    deps = [
        ":includes",
        ":base",
    ],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "onnxruntime-extensions-tokenizer",
    hdrs = glob([
        "shared/api/c_api_utils.hpp",
        "shared/api/runner.hpp",
        "shared/api/tokenizer_impl.h",
        "shared/api/c_only_test.h",
        "shared/api/tokenizer_impl.h",
    ]),
    srcs = glob([
        "shared/api/c_api_tokenizer.cc",
        "shared/api/c_api_utils.cc",
        "shared/api/tokenizer_impl.cc",
        "shared/lib/*.cc",
        "shared/extensions.cc",
        "shared/api/c_only_test.c",
    ]),
    strip_include_prefix = "shared/api",
    deps = [
        ":includes",
        ":base",
        ":gpt2_tokenizer",
        ":math",
        "@dlib//:libdlib",
    ],
    visibility = ["//visibility:public"],
)
cc_test(
    name = "tokenizer_test2",
    srcs = glob([
        "test/pp_api_test/test_tokenizer2.cc",
    ]),
    deps = [
        ":onnxruntime-extensions-tokenizer",
        "//third_party/googletest:gtest",
    ],

    visibility = ["//visibility:public"],
)
