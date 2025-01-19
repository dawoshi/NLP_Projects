package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
load("//build:buildflag_header.bzl", "buildflag_header")

cc_library(
    name = "macbert_error_correction",
    srcs = ["mac_bert_error_correction.cc",
           ],
    hdrs = [
            "mac_bert_error_correction.h",
          ],
    deps = [
            "//base:base",
	    "//third_party/gflags:gflags",
            "//third_party/onnxruntime:onnxruntime",
           ],
    copts = CXXOPTS + ["-fexceptions"],
    # linkstatic = True,
)
cc_test(
    name = "mac_bert_error_correction_test",
    srcs = [
            "mac_bert_error_correction_test.cc",
	   ],
    deps = [
	    ":macbert_error_correction",
	    "//base:base",
	    "//third_party/gflags:gflags",
	   ],
)
