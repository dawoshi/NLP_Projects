# Onnxruntime deploy

## 介绍

1. 使用onnxruntime 对训练的模型进行c++部署

2. 使用bazel进行编译

## 环境

- bazel 8.0.0
- onnxruntime-linux-x64-1.20.1

## 效果

test courps len count: 

Totle run Time : 

title级别的推理速度达到 

## 使用说明
```
# 编译：

bazel build //chinese_error_correction/macbert2csc/onnx-cpp/model:mac_bert_error_correction_test


# model.onnx和vocab.txt

data/chinese_error_correction/macbert2csc/onnx-cpp/model/

# 执行

nohup ./bazel-bin/chinese_error_correction/macbert2csc/onnx-cpp/model/mac_bert_error_correction_test &

```

## 参考
[1] https://github.com/shibing624/pycorrector
