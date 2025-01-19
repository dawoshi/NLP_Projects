package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
load("//build:buildflag_header.bzl", "buildflag_header")

cc_library(
    name = "t5_error_correction",
    srcs = ["t5_error_correction.cc",
           ],
    hdrs = [
            "t5_error_correction.h",
          ],
    deps = [
            "//base:base",
            "@nlohmann_json//:json",
	    "//third_party/gflags:gflags",
            "//third_party/onnxruntime:onnxruntime",
            "//third_party/onnxruntime-extensions:onnxruntime-extensions-tokenizer",
           ],
    copts = CXXOPTS + ["-fexceptions"],
    # linkstatic = True,
)
cc_test(
    name = "t5_error_correction_test",
    srcs = [
            "t5_error_correction_test.cc",
	   ],
    deps = [
	    ":t5_error_correction",
	    "//base:base",
	    "//third_party/gflags:gflags",
	   ],
)
