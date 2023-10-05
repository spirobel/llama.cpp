#include <stddef.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include "common.h"
#include "llama.h"
#include "api-llama.h"

bool cstring_to_params(char *argstring, gpt_params &params);
bool cstring_to_params(char *argstring, gpt_params &params)
{
    enum
    {
        kMaxArgs = 64
    };
    int argc = 0;
    char *argv[kMaxArgs];

    char *p2 = strtok(argstring, " ");
    while (p2 && argc < kMaxArgs - 1)
    {
        argv[argc++] = p2;
        p2 = strtok(0, " ");
        fprintf(stderr, "argc %i %s ", argc, p2);
    }
    argv[argc] = 0;
    fprintf(stderr, "argstring: %s argc: %i", argstring, argc);
    return gpt_params_parse(argc, argv, params);
}

struct llama_model *load_model(char *args)
{
    gpt_params params;
    if (!cstring_to_params(args, params))
    {
        return NULL;
    }

    // init LLM
    llama_backend_init(params.numa);

    // initialize the model
    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model *model = llama_load_model_from_file(params.model.c_str(), mparams);
    return model;
}

void prompt(char *prompt, struct llama_model *model, char *args, void (*prompt_callback)(const char *response, uint64_t len))
{

    gpt_params params;
    if (!cstring_to_params(args, params))
    {
        return;
    }
    auto lparams = llama_context_params_from_gpt_params(params);

    llama_context *ctx = llama_new_context_with_model(model, lparams);
    llama_set_rng_seed(ctx, params.seed);

    // total length of the sequence including the prompt
    const int n_len = 200;

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, prompt, true);

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list)
    {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0);

    // evaluate the initial prompt
    batch.n_tokens = tokens_list.size();

    for (int32_t i = 0; i < batch.n_tokens; i++)
    {
        batch.token[i] = tokens_list[i];
        batch.pos[i] = i;
        batch.seq_id[i] = 0;
        batch.logits[i] = false;
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0)
    {
        printf("%s: llama_decode() failed\n", __func__);
        return;
    }

    // main loop

    int n_cur = batch.n_tokens;
    int n_decode = 0;

    while (n_cur <= n_len)
    {
        // sample the next token
        {
            auto n_vocab = llama_n_vocab(model);
            auto *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++)
            {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of stream?
            if (new_token_id == llama_token_eos(ctx) || n_cur == n_len)
            {
                printf("\n");

                break;
            }
            auto new_chars = llama_token_to_piece(ctx, new_token_id);
            prompt_callback(new_chars.c_str(), new_chars.size());
            std::string carrot = "🥕";
            if (new_chars.find("bot") != std::string::npos)
            {
                prompt_callback(carrot.c_str(), carrot.size());
            }

            // prepare the next batch
            batch.n_tokens = 0;

            // push this new token for next evaluation
            batch.token[batch.n_tokens] = new_token_id;
            batch.pos[batch.n_tokens] = n_cur;
            batch.seq_id[batch.n_tokens] = 0;
            batch.logits[batch.n_tokens] = true;

            batch.n_tokens += 1;

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch))
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return;
        }
    }
}