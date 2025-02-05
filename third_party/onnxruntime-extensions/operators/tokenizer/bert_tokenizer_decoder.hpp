// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "third_party/onnxruntime-extensions/include/ocos.h"
#include "third_party/onnxruntime-extensions/base/ustring.h"
#include "third_party/onnxruntime-extensions/base/string_utils.h"
#include "third_party/onnxruntime-extensions/base/string_tensor.h"

class BertTokenizerDecoder {
 public:
  BertTokenizerDecoder(std::string vocab, std::string unk_token, std::string sep_token, std::string pad_token,
                       std::string cls_token, std::string mask_token, std::string suffix_indicator);
  std::string Decode(const std::vector<int64_t>& ids, bool skip_special_tokens, bool clean_up_tokenization_spaces);

 private:
  std::string unk_token_;
  int32_t unk_token_id_ = -1;
  int32_t sep_token_id_ = -1;
  int32_t pad_token_id_ = -1;
  int32_t cls_token_id_ = -1;
  int32_t mask_token_id_ = -1;
  std::string suffix_indicator_;
  std::vector<std::string_view> vocab_;
  std::string raw_vocab_;
  std::vector<bool> is_substr_;

  bool RemoveTokenizeSpace(int64_t pre_token_id, int64_t new_token_id);
};

struct KernelBertTokenizerDecoder : BaseKernel {
  KernelBertTokenizerDecoder(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<int64_t>& ids,
               const ortc::Tensor<int64_t>& positions,
               ortc::Tensor<std::string>& output) const;

 private:
  std::shared_ptr<BertTokenizerDecoder> decoder_;
  bool use_indices_;
  bool skip_special_tokens_;
  bool clean_up_tokenization_spaces_;
};
