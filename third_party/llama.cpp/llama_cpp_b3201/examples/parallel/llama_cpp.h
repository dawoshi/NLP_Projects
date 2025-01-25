// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.
#ifndef MODEL_LLAMA_CPP_H
#define MODEL_LLAMA_CPP_H


#include "third_party/llama.cpp/llama_cpp_b3201/common/common.h"
#include "third_party/llama.cpp/llama_cpp_b3201/llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>

namespace llama_cpp {

struct client {
    ~client() {
        if (ctx_sampling) {
            llama_sampling_free(ctx_sampling);
        }
    }

    int32_t id = 0;

    llama_seq_id seq_id = -1;

    llama_token sampled;

    int64_t t_start_prompt;
    int64_t t_start_gen;

    int32_t n_prompt  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;

    std::string input;
    std::string prompt;
    std::string response;

    struct llama_sampling_context * ctx_sampling = nullptr;
};

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();

    while (start < end && isspace(str[start])) {
        start += 1;
    }

    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }

    return str.substr(start, end - start);
}

static void print_date_time() {
    std::time_t current_time = std::time(nullptr);
    std::tm* local_time = std::localtime(&current_time);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);

    printf("\n\033[35mrun parameters as at %s\033[0m\n", buffer);
}

// Define a split string function to ...
static std::vector<std::string> split_string(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}



class LLamaCpp {
   public:
       LLamaCpp();
       virtual ~LLamaCpp();
       bool Init();
       // void predict(const std::string& text, std::string *res);
       void infer(std::string &k_system, std::vector<std::string> &k_prompts);
   private:
       int32_t n_clients_;
       int32_t n_seq_;
       bool cont_batching_;
       bool dump_kv_cache_;
       int n_ctx_;

       gpt_params params_;
       // llama_batch batch_;
       llama_model * model_ = NULL;
       llama_context * ctx_ = NULL;
       std::vector<client> clients_;
       DISALLOW_COPY_AND_ASSIGN(LLamaCpp);
  };
} // namespace
#endif // MODEL_LLAMA_CPP_H
