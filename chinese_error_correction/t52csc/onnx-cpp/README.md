# Onnxruntime deploy

## 介绍

1. 使用onnxruntime 对训练的模型进行c++部署

2. 使用bazel进行编译

## 环境

- bazel 8.0.0
- onnxruntime-linux-x64-1.20.1

## 效果
title 级别
test courps len count: 700

Totle run Time : 139136ms

## 使用说明
```
# 编译：

bazel build //chinese_error_correction/t52csc/onnx-cpp/model:t5_error_correction_test

# onnx file and tokenizer config file

config.json
decoder_model_quant.onnx
decoder_with_past_model_quant.onnx
encoder_model_quant.onnx
generation_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json

data/chinese_error_correction/t52csc/onnx-cpp/model

# 执行

nohup ./bazel-bin/chinese_error_correction/t52csc/onnx-cpp/model/t5_error_correction_test &

```

## 参考
[1] https://github.com/shibing624/pycorrector
