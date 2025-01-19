#ifndef MODEL_ERROR_CORRECTION_H
#define MODEL_ERROR_CORRECTION_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "base/tokenization.h"
#include "onnxruntime_cxx_api.h"

namespace error_correction {

  class MacBertErrorCorrection {
    public:
        MacBertErrorCorrection();
        virtual ~MacBertErrorCorrection();
        bool Init();
        void predict(const std::string& text, std::vector<uint64_t> *res);
    private:
        void infer(const std::string &query,
			Ort::Session* session,
			std::vector<uint64_t>* res);
        int curr_sess_id_;
        bool session_inited_;
	Ort::Env env;
	std::unordered_map<int, int> token_to_orig_map_;
        std::vector<Ort::Session*> session_list_;
        std::unique_ptr<base::FullTokenizer> tokenizer_ = nullptr;
        DISALLOW_COPY_AND_ASSIGN(MacBertErrorCorrection);
  };
} // namespace
#endif // MODEL_ERROR_CORRECTION_H
