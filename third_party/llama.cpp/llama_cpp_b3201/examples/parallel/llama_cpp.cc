// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>

#include "gflags.h"
#include "base/logging.h"
#include "base/json_utils.hpp"
#include "base/strings/string_util.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"

#include "third_party/llama.cpp/llama_cpp_b3201/llama.h"
#include "third_party/llama.cpp/llama_cpp_b3201/common/common.h"
#include "third_party/llama.cpp/llama_cpp_b3201/examples/parallel/llama_cpp.h"

int LLAMA_BUILD_NUMBER = 1;
char const *LLAMA_COMMIT = "@BUILD_COMMIT@";
char const *LLAMA_COMPILER = "@BUILD_COMPILER@";
char const *LLAMA_BUILD_TARGET = "@BUILD_TARGET@";

DEFINE_int32(model_max_len, 512, "pre_trained_model sequence length");
DEFINE_int32(parallel_number, 1, "session thread number");
DEFINE_int32(generate_max_len, 32, "generate sequence length");


DEFINE_string(model_file,
		"data/third_party/llama.cpp/models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
		"encode onnx model file");

namespace llama_cpp {


LLamaCpp::LLamaCpp() {
}

LLamaCpp::~LLamaCpp() {
    llama_free(ctx_);
    llama_free_model(model_);
    llama_backend_free();
}


bool LLamaCpp::Init() {
 
    std::string model_file = FLAGS_model_file;
    if(!base::PathExists(base::FilePath(model_file))){
        LOG_TEE("model file or vocab file not exist please check it");
        return false;
    }
    params_.model = model_file;
    
    // number of simultaneous "clients" to simulate
    
    n_clients_ = FLAGS_parallel_number;

    // dedicate one sequence to the system prompt
    params_.n_parallel += 3;

    // requests to simulate
    n_seq_ = params_.n_sequences;

    // insert new requests as soon as the previous one is done
    cont_batching_ = params_.cont_batching;

    dump_kv_cache_ = params_.dump_kv_cache;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params_.numa);

    // load the target model
    std::tie(model_, ctx_) = llama_init_from_gpt_params(params_);


    // n_ctx_ = llama_n_ctx(ctx_);

    clients_.resize(n_clients_);

    for (size_t i = 0; i < clients_.size(); ++i) {
        auto & client = clients_[i];
        client.id = i;
        client.ctx_sampling = llama_sampling_init(params_.sparams);
    }

    return true;
}
void LLamaCpp::infer(std::string &k_system, std::vector<std::string> &k_prompts){
     // load the prompts from an external file if there are any
    if (params_.prompt.empty()) {
        printf("\n\033[32mNo new questions so proceed with build-in defaults.\033[0m\n");
    } else {
        // Output each line of the input params.prompts vector and copy to k_prompts
        int index = 0;
        printf("\n\033[32mNow printing the external prompt file %s\033[0m\n\n", params_.prompt_file.c_str());

        std::vector<std::string> prompts = split_string(params_.prompt, '\n');
        for (const auto& prompt : prompts) {
            k_prompts.resize(index + 1);
            k_prompts[index] = prompt;
            index++;
            printf("%3d prompt: %s\n", index, prompt.c_str());
        }
    }

    std::vector<llama_token> tokens_system;
    tokens_system = ::llama_tokenize(ctx_, k_system, true);
    const int32_t n_tokens_system = tokens_system.size();
    llama_seq_id g_seq_id = 0;

    // the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
    // users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
    llama_batch batch = llama_batch_init(n_ctx_, 0, 1);

    int32_t n_total_prompt = 0;
    int32_t n_total_gen    = 0;
    int32_t n_cache_miss   = 0;

    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx_, n_clients_);

    const auto t_main_start = ggml_time_us();

    LOG_TEE("%s: Evaluating the system prompt ...\n", __func__);

    for (int32_t i = 0; i < n_tokens_system; ++i) {
        llama_batch_add(batch, tokens_system[i], i, { 0 }, false);
    }

    // LOG_TEE("llama_decode before ok");
    
    if (llama_decode(ctx_, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return;
    }

    // assign the system KV cache to all parallel sequences
    for (int32_t i = 1; i <= n_clients_; ++i) {
        llama_kv_cache_seq_cp(ctx_, 0, i, -1, -1);
    }

    LOG_TEE("\n");

    LOG_TEE("Processing requests ...\n\n");

    while (true) {
        if (dump_kv_cache_) {
            llama_kv_cache_view_update(ctx_, &kvc_view);
            llama_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        llama_batch_clear(batch);

        // decode any currently ongoing sequences
        for (auto & client : clients_) {
            if (client.seq_id == -1) {
                continue;
            }

            client.i_batch = batch.n_tokens;

            llama_batch_add(batch, client.sampled, n_tokens_system + client.n_prompt + client.n_decoded, { client.id + 1 }, true);

            client.n_decoded += 1;
        }

        if (batch.n_tokens == 0) {
            // all sequences have ended - clear the entire KV cache
            for (int i = 1; i <= n_clients_; ++i) {
                llama_kv_cache_seq_rm(ctx_, i, -1, -1);
                // but keep the system prompt
                llama_kv_cache_seq_cp(ctx_, 0, i, -1, -1);
            }

            LOG_TEE("%s: clearing the KV cache\n", __func__);
        }

        // insert new sequences for decoding
        if (cont_batching_ || batch.n_tokens == 0) {
            for (auto & client : clients_) {
                if (client.seq_id == -1 && g_seq_id < n_seq_) {
                    client.seq_id = g_seq_id;

                    client.t_start_prompt = ggml_time_us();
                    client.t_start_gen    = 0;

                    client.input    = k_prompts[rand() % k_prompts.size()];
                    client.prompt   = client.input + "\nAssistant:";
                    client.response = "";

                    llama_sampling_reset(client.ctx_sampling);

                    // do not prepend BOS because we have a system prompt!
                    std::vector<llama_token> tokens_prompt;
                    tokens_prompt = ::llama_tokenize(ctx_, client.prompt, false);

                    for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                        llama_batch_add(batch, tokens_prompt[i], i + n_tokens_system, { client.id + 1 }, false);
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    client.n_prompt  = tokens_prompt.size();
                    client.n_decoded = 0;
                    client.i_batch   = batch.n_tokens - 1;

                    LOG_TEE("\033[31mClient %3d, seq %4d, started decoding ...\033[0m\n", client.id, client.seq_id);

                    g_seq_id += 1;

                    // insert new requests one-by-one
                    //if (cont_batching) {
                    //    break;
                    //}
                }
            }
        }

        if (batch.n_tokens == 0) {
            break;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params_.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            // experiment: process in powers of 2
            //if (i + n_batch > (int32_t) batch.n_tokens && n_batch > 32) {
            //    n_batch /= 2;
            //    i -= n_batch;
            //    continue;
            //}

            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx_, batch_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_TEE("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return;
                }

                LOG("%s : failed to decode the batch, retrying with n_batch = %d\n", __func__, n_batch / 2);

                n_cache_miss += 1;

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                continue;
            }

            LOG("%s : decoded batch of %d tokens\n", __func__, n_tokens);

            for (auto & client : clients_) {
                if (client.i_batch < (int) i || client.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                //printf("client %d, seq %d, token %d, pos %d, batch %d\n",
                //        client.id, client.seq_id, client.sampled, client.n_decoded, client.i_batch);

                const llama_token id = llama_sampling_sample(client.ctx_sampling, ctx_, NULL, client.i_batch - i);

                llama_sampling_accept(client.ctx_sampling, ctx_, id, true);

                if (client.n_decoded == 1) {
                    // start measuring generation time after the first token to make sure all concurrent clients
                    // have their prompt already processed
                    client.t_start_gen = ggml_time_us();
                }

                const std::string token_str = llama_token_to_piece(ctx_, id);

                client.response += token_str;
                client.sampled = id;

                //printf("client %d, seq %d, token %d, pos %d, batch %d: %s\n",
                //        client.id, client.seq_id, id, client.n_decoded, client.i_batch, token_str.c_str());

                if (client.n_decoded > 2 &&
                        (llama_token_is_eog(model_, id) ||
                         (params_.n_predict > 0 && client.n_decoded + client.n_prompt >= params_.n_predict) ||
                         client.response.find("User:") != std::string::npos ||
                         client.response.find('\n') != std::string::npos)) {
                    // basic reverse prompt
                    const size_t pos = client.response.find("User:");
                    if (pos != std::string::npos) {
                        client.response = client.response.substr(0, pos);
                    }

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    llama_kv_cache_seq_rm(ctx_, client.id + 1, -1, -1);
                    llama_kv_cache_seq_cp(ctx_, 0, client.id + 1, -1, -1);

                    const auto t_main_end = ggml_time_us();

                    LOG_TEE("\033[31mClient %3d, seq %3d/%3d, prompt %4d t, response %4d t, time %5.2f s, speed %5.2f t/s, cache miss %d \033[0m \nInput:    %s\n\033[35mResponse: %s\033[0m\n\n",
                            client.id, client.seq_id, n_seq_, client.n_prompt, client.n_decoded,
                            (t_main_end - client.t_start_prompt) / 1e6,
                            (double) (client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6,
                            n_cache_miss,
                            llama_cpp::trim(client.input).c_str(),
                            llama_cpp::trim(client.response).c_str());

                    n_total_prompt += client.n_prompt;
                    n_total_gen    += client.n_decoded;

                    client.seq_id = -1;
                }

                client.i_batch = -1;
            }
        }
    }

    const auto t_main_end = ggml_time_us();

    print_date_time();

    LOG_TEE("\n%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients_, n_seq_, cont_batching_, n_tokens_system);
    if (params_.prompt_file.empty()) {
        params_.prompt_file = "used built-in defaults";
    }
    LOG_TEE("External prompt file: \033[32m%s\033[0m\n", params_.prompt_file.c_str());
    LOG_TEE("Model and path used:  \033[32m%s\033[0m\n\n", params_.model.c_str());

    LOG_TEE("Total prompt tokens: %6d, speed: %5.2f t/s\n", n_total_prompt, (double) (n_total_prompt              ) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Total gen tokens:    %6d, speed: %5.2f t/s\n", n_total_gen,    (double) (n_total_gen                 ) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Total speed (AVG):   %6s  speed: %5.2f t/s\n", "",             (double) (n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Cache misses:        %6d\n", n_cache_miss);

    LOG_TEE("\n");
    llama_batch_free(batch);
 }
} //namespace
