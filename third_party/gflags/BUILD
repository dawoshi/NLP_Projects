package(default_visibility = ["//visibility:public"])
load("//build:compiler.bzl", "CXXOPTS")
cc_library (
    name = "gflags",
    hdrs = glob(["src/*.h"]),
    srcs = glob(["src/*.cc"], exclude=["src/windows_*"]),
    includes = ["src"],
    linkstatic = True,
    strip_include_prefix = "src",
    linkopts = ["-pthread"],
)
