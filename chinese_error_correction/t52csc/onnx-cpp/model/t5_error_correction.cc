#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include "gflags.h"
#include "gtest/gtest.h"
#include "base/logging.h"
#include "base/json_utils.hpp"
#include "base/strings/string_util.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "chinese_error_correction/t52csc/onnx-cpp/model/t5_error_correction.h"

DEFINE_int32(model_max_len, 512, "pre_trained_model sequence length");
DEFINE_int32(sess_thread_number, 1, "session thread number");
DEFINE_int32(generate_max_len, 32, "generate sequence length");

// DEFINE_int32(sess_number, 8, "session number");

DEFINE_string(encoder_model_file,
		"data/chinese_error_correction/t52csc/onnx-cpp/model/encoder_model_quant.onnx",
		"decode onnx model file");
DEFINE_string(decoder_model_file,
		"data/chinese_error_correction/t52csc/onnx-cpp/model/decoder_model_quant.onnx",
		"encode onnx model file");
DEFINE_string(past_model_file,
		"data/chinese_error_correction/t52csc/onnx-cpp/model/decoder_with_past_model_quant.onnx",
		"init decode onnx model file");

DEFINE_string(special_tokens_file,
		"data/chinese_error_correction/t52csc/onnx-cpp/model/special_tokens_map.json",
		"init decode onnx model file");

DEFINE_string(generation_config_file,
		"data/chinese_error_correction/t52csc/onnx-cpp/model/generation_config.json",
		"init decode onnx model file");

DEFINE_string(tokenizer_config_file,
		"data/chinese_error_correction/t52csc/onnx-cpp/model",
		"tokenizer config file");

namespace error_correction {

template <typename T>
int argmax(T begin, T end) {
    return std::distance(begin, std::max_element(begin, end));
}




T5ErrorCorrection::T5ErrorCorrection() {
  curr_sess_id_ = 0;
  session_inited_ = false;
}

T5ErrorCorrection::~T5ErrorCorrection() {
  if (session_inited_) session_inited_ = false;
}
void T5ErrorCorrection::GetInputOutputInfo(
    const std::shared_ptr<Ort::Session> &session,
    std::vector<const char *> *in_names, std::vector<const char *> *out_names){
    Ort::AllocatorWithDefaultOptions allocator;
    // Input info
    int num_nodes = session->GetInputCount();
    in_names->resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
    {
        std::unique_ptr<char, Ort::detail::AllocatedFree> name = session->GetInputNameAllocated(i, allocator);
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
        std::stringstream shape;
        for (auto j : node_dims)
        {
            shape << j;
            shape << " ";
        }
        std::cout << "\tInput " << i << " : name=" << name.get() << " type=" << type
                  << " dims=" << shape.str() << std::endl;
        (*in_names)[i] = name.get();
        name.release();
    }
    // Output info
    num_nodes = session->GetOutputCount();
    out_names->resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
    {
        std::unique_ptr<char, Ort::detail::AllocatedFree> name = session->GetOutputNameAllocated(i, allocator);
        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
        std::stringstream shape;
        for (auto j : node_dims)
        {
            shape << j;
            shape << " ";
        }
        std::cout << "\tOutput " << i << " : name=" << name.get() << " type=" << type
                  << " dims=" << shape.str() << std::endl;
        (*out_names)[i] = name.get();
        name.release();
    }
}


bool T5ErrorCorrection::Init() {
 
  std::string decoder_model_file = FLAGS_decoder_model_file;
  std::string encoder_model_file = FLAGS_encoder_model_file;
  std::string past_model_file = FLAGS_past_model_file;
  std::string tokenizer_config_file = FLAGS_tokenizer_config_file;
  std::string special_tokens_file = FLAGS_special_tokens_file;
  std::string generation_config_file = FLAGS_generation_config_file;

  if(!base::PathExists(base::FilePath(decoder_model_file))
                  || !base::PathExists(base::FilePath(encoder_model_file))
                  || !base::PathExists(base::FilePath(past_model_file))
                  || !base::PathExists(base::FilePath(special_tokens_file))
                  || !base::PathExists(base::FilePath(generation_config_file))
        	  || !base::PathExists(base::FilePath(tokenizer_config_file))){
          LOG(INFO) << "model file or vocab file not exist please check it";
      return false;
  }

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(FLAGS_sess_thread_number);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.DisableMemPattern();
  session_options.DisableCpuMemArena();



  // for (int i = 0; i < FLAGS_sess_number; ++i) {
  //   auto session =
  //     new Ort::Session(env, model_file.c_str(), session_options);
  //   session_list_.push_back(session);
  // }
  encoder_session_  = std::make_unique<Ort::Session>(env, encoder_model_file.c_str(), session_options);
  GetInputOutputInfo(encoder_session_, &encoder_input_names_, &encoder_output_names_);

  decoder_session_  = std::make_unique<Ort::Session>(env, decoder_model_file.c_str(), session_options);
  GetInputOutputInfo(decoder_session_, &decoder_input_names_, &decoder_output_names_);

  past_session_ = std::make_unique<Ort::Session>(env, past_model_file.c_str(), session_options);
  GetInputOutputInfo(past_session_, &past_input_names_, &past_output_names_);

  tokenizer_ = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer_->Load(tokenizer_config_file);
  if (!status.IsOk()) {
    LOG(INFO) << status.ToString();
    tokenizer_.reset();
  }


  std::ifstream special_tokens_f(special_tokens_file);
  if (!special_tokens_f.is_open())
      return false;

  nlohmann::json special_tokens_data = nlohmann::json::parse(special_tokens_f);

  // they are in the format {"bos_token": { "content": "<s>",... }}
  auto special_token_content_str = [&special_tokens_data](const std::string& key_name, std::string& val) {
      if (val.empty() && special_tokens_data.contains(key_name)) {
          utils::read_json_param(special_tokens_data[key_name], "content", val);
      }
  };
  special_token_content_str("eos_token", eos_token);
  special_token_content_str("unk_token", unk_token);
  special_token_content_str("pad_token", pad_token);

  std::ifstream generation_config_f(generation_config_file);
  if (!generation_config_f.is_open())
      return false;

  nlohmann::json generation_config_data = nlohmann::json::parse(generation_config_f);

  // they are in the format {"bos_token": { "content": "<s>",... }}
  auto generation_token_content_str = [&generation_config_data](const std::string& key_name, int& val) {
      if (generation_config_data.contains(key_name)) {
          utils::read_json_param(generation_config_data[key_name], "content", val);
      }
  };
  generation_token_content_str("eos_token_id", eos_token_id);
  generation_token_content_str("decoder_start_token_id", decoder_start_token_id);
  generation_token_content_str("pad_token_id", pad_token_id);

  session_inited_ = true;
  LOG(INFO) << "session init num thread: " << FLAGS_sess_thread_number;
  LOG(INFO) << "model init succuss!";
  return true;
}

void T5ErrorCorrection::infer( 
		const std::string &query,
		std::string *res) {
    int64_t token_index;
    std::vector<std::string_view> input;
    input.push_back(query);
    std::vector<std::vector<extTokenId_t>> token_ids;
    auto status = tokenizer_->Tokenize(input, token_ids);    
    EXPECT_TRUE(status.IsOk());
    std::vector<int64_t> encoder_input_ids;
    for(const auto& e: token_ids[0]){
        encoder_input_ids.push_back(static_cast<int64_t>(e));
    }
    std::vector<int64_t> encoder_attention_mask(encoder_input_ids.size(), 1);
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> encoder_input_ids_shape{ 1,(int64_t)encoder_input_ids.size()};
    Ort::Value encoder_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, encoder_input_ids.data(), encoder_input_ids.size(),
                                                    encoder_input_ids_shape.data(), encoder_input_ids_shape.size());

    std::array<int64_t, 2> encoder_attention_mask_shape{ 1,(int64_t)encoder_attention_mask.size()};
    Ort::Value encoder_attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, encoder_attention_mask.data(), encoder_attention_mask.size(), encoder_attention_mask_shape.data(), encoder_attention_mask_shape.size());

    std::vector<Ort::Value> encoder_input_onnx;
    encoder_input_onnx.emplace_back(std::move(encoder_input_ids_tensor));
    encoder_input_onnx.emplace_back(std::move(encoder_attention_mask_tensor));
    auto encoder_outputTensor = encoder_session_->Run(Ort::RunOptions(nullptr),
                                                     encoder_input_names_.data(),
                                                     encoder_input_onnx.data(),
                                                     encoder_input_names_.size(),
                                                     encoder_output_names_.data(),
                                                     encoder_output_names_.size());
    float *output_data = encoder_outputTensor[0].GetTensorMutableData<float>();
    size_t output_size = encoder_outputTensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
    int64_t decoder_start_token_id = 0;
    int64_t eos_token_id = 1;
    std::vector<int64_t> decoder_input_ids = {decoder_start_token_id};
    std::vector<int64_t> final_ids;
    final_ids.push_back(decoder_start_token_id);
    std::array<int64_t, 2> decoder_input_ids_shape{1, (int64_t)decoder_input_ids.size()};
    Ort::Value decoder_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, decoder_input_ids.data(), decoder_input_ids.size(),
                                                                            decoder_input_ids_shape.data(), decoder_input_ids_shape.size());
    Ort::Value decoder_attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, encoder_attention_mask.data(), encoder_attention_mask.size(),
                                                                                 encoder_attention_mask_shape.data(), encoder_attention_mask_shape.size());

    std::vector<Ort::Value> decoder_input_onnx;
    decoder_input_onnx.emplace_back(std::move(decoder_attention_mask_tensor));
    decoder_input_onnx.emplace_back(std::move(decoder_input_ids_tensor));
    decoder_input_onnx.emplace_back(std::move(encoder_outputTensor[0]));
    auto decoder_outputTensor = decoder_session_->Run(Ort::RunOptions(nullptr),
                                                     decoder_input_names_.data(),
                                                     decoder_input_onnx.data(),
                                                     decoder_input_names_.size(),
                                                     decoder_output_names_.data(),
                                                     decoder_output_names_.size());
    float *decoder_output_data = decoder_outputTensor[0].GetTensorMutableData<float>();
    size_t decoder_output_size = decoder_outputTensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> logits(decoder_output_data, decoder_output_data + decoder_output_size);
    token_index = argmax(logits.begin(), logits.end());
    // token_index = sample_top_p_with_penalty(logits, temperature, final_ids, penalty);
    final_ids.push_back(token_index);

    std::vector<int64_t> past_input_ids = {token_index};
    std::array<int64_t, 2> past_input_ids_shape{1, (int64_t)past_input_ids.size()};
    Ort::Value past_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, past_input_ids.data(), past_input_ids.size(),
                                                                            past_input_ids_shape.data(), past_input_ids_shape.size());
    Ort::Value past_attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, encoder_attention_mask.data(), encoder_attention_mask.size(),
                                                                                 encoder_attention_mask_shape.data(), encoder_attention_mask_shape.size());

    std::vector<Ort::Value> past_input_onnx;
    past_input_onnx.emplace_back(std::move(past_attention_mask_tensor));
    past_input_onnx.emplace_back(std::move(past_input_ids_tensor));
    past_input_onnx.emplace_back(std::move(decoder_input_onnx[2]));

    for (size_t i = 1; i < decoder_outputTensor.size(); i++)
    {
      past_input_onnx.emplace_back(std::move(decoder_outputTensor[i]));
    }

    int64_t *current_input_id = past_input_onnx[1].GetTensorMutableData<int64_t>();
    for (size_t i = 0; i < FLAGS_generate_max_len; i += 1)
    {
        auto past_outputTensor = past_session_->Run(Ort::RunOptions(nullptr),
                                                          past_input_names_.data(),
                                                          past_input_onnx.data(),
                                                          past_input_names_.size(),
                                                          past_output_names_.data(),
                                                          past_output_names_.size());

        float *past_output_data = past_outputTensor[0].GetTensorMutableData<float>();
        size_t past_output_size = past_outputTensor[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> past_logits(past_output_data, past_output_data + past_output_size);

        token_index = argmax(past_logits.begin(), past_logits.end());
        // token_index = sample_top_p_with_penalty(past_logits, temperature, final_ids, penalty);
        final_ids.push_back(token_index);
        current_input_id[0] = token_index;

        for (size_t j = 1; j < past_outputTensor.size(); j += 2)
        {
          past_input_onnx[2*j+1] = std::move(past_outputTensor[j]);
          past_input_onnx[2*j+2] = std::move(past_outputTensor[j + 1]);
        }

        if (token_index == eos_token_id)
        {
          break;
        }

    }
    std::unique_ptr<ort_extensions::TokenizerDecodingState> decoder_cache;

    for (size_t j = 1; j < final_ids.size() - 1; j += 1)
    {
      std::string token;
      
      status = tokenizer_->Id2Token(final_ids[j], token, decoder_cache); 
      if (token == "<unk>")
      {
        continue;
      }
      EXPECT_TRUE(status.IsOk());
      res->append(token);
    }
    base::ReplaceSubstringsAfterOffset(res, 0, "\u2581", "");
    LOG(INFO) << *res;
}


void T5ErrorCorrection::predict(const std::string& content, std::string *res) {
  if (!session_inited_) {
    return;
  }
  if (content.empty()) return;
  // curr_sess_id_ = (curr_sess_id_ + 1) % session_list_.size();
  T5ErrorCorrection::infer(content, res);
}
} // namespace
