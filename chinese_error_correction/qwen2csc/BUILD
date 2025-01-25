package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
load("//build:buildflag_header.bzl", "buildflag_header")

cc_library(
    name = "llama_cpp_imp",
    srcs = ["llama_cpp.cc",
           ],
    hdrs = [
            "llama_cpp.h",
          ],
    deps = [
            "//base:base",
            "//third_party/gflags:gflags",
            "//third_party/llama.cpp/llama_cpp_b3201:llama_cpp",
           ],
    copts = CXXOPTS + ["-fexceptions"],
)

cc_test(
    name = "qwen_test",
    srcs = [
            "qwen_test.cc",
	   ],
    deps = [
	    ":llama_cpp_imp",
	    "//base:base",
	    "//third_party/gflags:gflags",
	   ],
)
