#ifndef MODEL_ERROR_CORRECTION_T5_H
#define MODEL_ERROR_CORRECTION_T5_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "c_only_test.h"
#include "ortx_cpp_helper.h"
#include "tokenizer_impl.h"

#include "nlohmann/json.hpp"
#include "onnxruntime_cxx_api.h"

namespace error_correction {

  class T5ErrorCorrection {
    public:
        T5ErrorCorrection();
        virtual ~T5ErrorCorrection();
        bool Init();
        void predict(const std::string& text, std::string *res);
    private:
        void infer(
                   const std::string &query,
	           std::string *res);
        void GetInputOutputInfo(
                   const std::shared_ptr<Ort::Session> &session,
                   std::vector<const char *> *in_names, 
                   std::vector<const char *> *out_names);
        int curr_sess_id_;
        bool session_inited_;
	// Ort::Env env;
        
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Decoder");
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        int decoder_start_token_id;
        int eos_token_id;
        int pad_token_id;

        std::string eos_token;
        std::string pad_token;
        std::string unk_token;

        float temperature = 0.7;
        float penalty = 1.5;

	std::unordered_map<int, int> token_to_orig_map_;
        std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
        std::shared_ptr<Ort::Session> decoder_session_ = nullptr;
        std::shared_ptr<Ort::Session> past_session_ = nullptr;
        std::unique_ptr<ort_extensions::TokenizerImpl> tokenizer_ = nullptr;
        std::vector<const char *> encoder_input_names_;
        std::vector<const char *> encoder_output_names_;
        std::vector<const char *> decoder_input_names_;
        std::vector<const char *> decoder_output_names_;
        std::vector<const char *> past_input_names_;
        std::vector<const char *> past_output_names_;
        DISALLOW_COPY_AND_ASSIGN(T5ErrorCorrection);
  };
} // namespace
#endif // MODEL_ERROR_CORRECTION_T5_H
