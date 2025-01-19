// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"

#include "c_only_test.h"
#include "ortx_cpp_helper.h"
#include "tokenizer_impl.h"

static void DumpTokenIds(const std::vector<std::vector<extTokenId_t>>& token_ids) {
#ifdef _DEBUG
  for (const auto& tokens : token_ids) {
    for (const auto& token : tokens) {
      std::cout << token << " ";
    }

    std::cout << std::endl;
  }

  std::cout << std::endl;
#endif
}


int main(){

  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-3");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  // validate tokenizer is not null
  // ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  std::vector<std::string_view> input = {R"(在中国，感受“冰雪经济”的热辣滚烫|积极投身全面依法治国伟大实践。)"};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  // Add an extra byte token for decoding tests
  // token_ids[0].push_back(35);  // <0x20>
  DumpTokenIds(token_ids);
  std::vector<int64_t> tokenids;
  for(int i = 0; i < token_ids[0].size(); ++i){
      int64_t id = static_cast<int64_t>(token_ids[0][i]);
      std::cout << id << std::endl;
      tokenids.push_back(id);
  }
  std::string text;
  std::unique_ptr<ort_extensions::TokenizerDecodingState> decoder_cache;
  // std::cout << "\"";
  for(const auto& token_id : token_ids[0]) {
    std::string token;
    auto status = tokenizer->Id2Token(token_id, token, decoder_cache);
    EXPECT_TRUE(status.IsOk());
    std::cout << token << " ";
    text.append(token);
  }
  return 0;
}
