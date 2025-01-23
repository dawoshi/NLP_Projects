package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_binary(
    name = "run_openai",
    srcs = ["run_openai.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "run_llama_on_vertex",
    srcs = ["run_llama_on_vertex.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "run_gemini_on_vertex",
    srcs = ["run_gemini_on_vertex.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "run_local_model_with_llamacpp",
    srcs = ["run_local_model_with_llamacpp.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/interop/backends:llamacpp",
        "//third_party/llama.cpp/intrinsics:handler_sets",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:executor_stacks",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "run_gemini_on_ai_studio",
    srcs = ["run_gemini_on_ai_studio.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/authoring:smart_chain",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "run_wolfram",
    srcs = ["run_wolfram.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary(
    name = "math_tool_agent",
    srcs = ["math_tool_agent.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/authoring:smart_chain",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "custom_concurrency_interface",
    srcs = ["custom_concurrency_interface.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/examples/executors:executor_stacks",
        "//third_party/llama.cpp/runtime:concurrency",
        "//third_party/llama.cpp/runtime:executor",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "multivariate_prompt_template_test",
    srcs = ["multivariate_prompt_template_test.cc"],
    deps = [
        "//third_party/llama.cpp/authoring:constructor",
        "//third_party/llama.cpp/runtime:executor_stacks",
        "//third_party/llama.cpp/runtime:runner",
        "//third_party/llama.cpp/runtime:status_macros",
        "//third_party/llama.cpp/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
