cc_library(
    name = "llama_cpp",
    visibility = ["//visibility:public"],
    srcs = [
        "ggml/src/ggml.c",
        "ggml/src/ggml-alloc.c",
        "ggml/src/ggml-backend.cpp",
        "ggml/src/ggml-quants.c",
        "llama.cpp",
    ],
    hdrs = [
        "ggml/include/ggml.h",
        "ggml/include/ggml-alloc.h",
        "ggml/include/ggml-backend.h",
        "ggml/src/ggml-backend-impl.h",
        "ggml/src/ggml-impl.h",
        "ggml/src/ggml-quants.h",
        "include/llama-cpp.h",
        "src/unicode.h",
    ],
)
# module(
#     name = "llama_cpp",
#     repo_name = "org_llama_cpp",
# )
