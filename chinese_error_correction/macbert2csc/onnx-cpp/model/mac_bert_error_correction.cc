#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include "gflags.h"
#include "base/logging.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "chinese_error_correction/macbert2csc/onnx-cpp/model/mac_bert_error_correction.h"

DEFINE_int32(model_max_len, 512, "pre_trained_model sequence length");
DEFINE_int32(sess_thread_number, 16, "session thread number");
DEFINE_int32(sess_number, 8, "session number");

DEFINE_string(model_file,
		"data/chinese_error_correction/macbert2csc/onnx-cpp/model/model.onnx",
		"onnx model file");
DEFINE_string(vocab_file,
		"data/chinese_error_correction/macbert2csc/onnx-cpp/model/vocab.txt",
		"pretrained model vocab file");

namespace error_correction {

template <typename T>
int argmax(T begin, T end) {
    return std::distance(begin, std::max_element(begin, end));
}
MacBertErrorCorrection::MacBertErrorCorrection() {
  curr_sess_id_ = 0;
  session_inited_ = false;
}

MacBertErrorCorrection::~MacBertErrorCorrection() {
  if (session_inited_) session_inited_ = false;
  // for(int i = 0; i < session_list_.size(); ++i) {
  //     if(session_list_[i] != nullptr)
  //         delete session_list_[i];
  // }
  session_list_.clear();
}

bool MacBertErrorCorrection::Init() {
  
  std::string model_file = FLAGS_model_file;
  std::string vocab_file = FLAGS_vocab_file;
  if(!base::PathExists(base::FilePath(model_file)) 
        	  || !base::PathExists(base::FilePath(vocab_file))){
          LOG(INFO) << "model file or vocab file not exist please check it";
      return false;
  }
  session_list_.clear();

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(FLAGS_sess_thread_number);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  for (int i = 0; i < FLAGS_sess_number; ++i) {
    auto session =
      new Ort::Session(env, model_file.c_str(), session_options);
    session_list_.push_back(session);
  }

  session_inited_ = true;
  LOG(INFO) << "session init num thread: " << FLAGS_sess_thread_number;
  tokenizer_.reset(new base::FullTokenizer(vocab_file.c_str()));
  LOG(INFO) << "model init succuss!";
  return true;
}

void MacBertErrorCorrection::infer( 
		const std::string &query,
		Ort::Session* session,
		std::vector<uint64_t>* res) {

    if(session == nullptr){
        return;
    }
    std::vector<std::string> query_tokens;
    query_tokens.clear();
    tokenizer_->tokenize(query.c_str(), &query_tokens, 10000);
    query_tokens.insert(query_tokens.begin(), "[CLS]");
    query_tokens.push_back("[SEP]");
    std::vector<int64_t> token_ids;
    token_ids.clear();
    tokenizer_->convert_tokens_to_ids(query_tokens, token_ids); 

 
    size_t sz = (int64_t)(token_ids.size());
    if(sz > FLAGS_model_max_len) {
        sz = FLAGS_model_max_len;
	token_ids[sz-1] = 102;
    }
    std::vector<int64_t> input(sz);
    std::vector<int64_t> mask(sz, 1);
    std::vector<int64_t> type(sz, (int64_t)0);
    for (int i = 0; i < sz; ++i) {
        input[i] = token_ids[i];
    }
    std::vector<int64_t> input_node_dims = {1, sz};
    size_t input_tensor_size = sz;
    // create input tensor object from data values ！！！！！！！！！！
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);

    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, mask.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);
    Ort::Value type_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, type.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(mask_tensor));
    ort_inputs.push_back(std::move(type_tensor));
   
    std::vector<const char*> input_node_names = {"input_ids", "token_type_ids", "attention_mask"};
    std::vector<const char*> output_node_names = {"logits"};
    // curr_sess_id_ = (curr_sess_id_ + 1) % session_list_.size();
    LOG(INFO) << "input data ok";
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
		    ort_inputs.size(), output_node_names.data(), output_node_names.size());
    
    assert(output_tensors.size() == 1);

    LOG(INFO) << "output ok"; 
    const int64_t* logits = output_tensors[0].GetTensorMutableData<int64_t>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    LOG(INFO) << "shape: " << shape[1];
     
    // std::vector<uint64_t> token_ids;
    for(int i = 0; i < shape[1]; ++i) {
       LOG(INFO) << logits[i];
       res->push_back(logits[i]);
    }
    // if(!tmp.empty()) {
    //     res->push_back(tmp);
    // }
}


void MacBertErrorCorrection::predict(const std::string& content, std::vector<uint64_t>* res) {
  if (!session_inited_) {
    return;
  }
  if (content.empty()) return;
  curr_sess_id_ = (curr_sess_id_ + 1) % session_list_.size();
  MacBertErrorCorrection::infer(content,
		   session_list_[curr_sess_id_], res);
}
} // namespace
