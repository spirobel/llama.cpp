#include <stddef.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include "common.h"
#include "llama.h"
#include "grammar-parser.h"
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

struct llama_parts *load_model(char *args)
{
    gpt_params params;
    if (!cstring_to_params(args, params))
    {
        return NULL;
    }
    // fprintf(stderr, "params->model: %s ", params.model.c_str());

    struct llama_parts *lparts = new struct llama_parts;

    llama_backend_init(params.numa);
    std::tie(lparts->model, lparts->ctx) = llama_init_from_gpt_params(params);

    return lparts;
}

void prompt(char *prompt, struct llama_parts *parts, char *args, void (*prompt_callback)(const char *response), char *grammar_str)
{

    gpt_params params;
    if (!cstring_to_params(args, params))
    {
        return;
    }
    auto lparams = llama_context_params_from_gpt_params(params);

    llama_context *ctx = llama_new_context_with_model(parts->model, lparams);

    llama_grammar *grammar = nullptr;
    if (grammar_str)
    {
        auto parsed_grammar = grammar_parser::parse(grammar_str);
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty())
        {
            fprintf(stderr, "grammar parse error");
        }
        grammar_parser::print_grammar(stderr, parsed_grammar);

        std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
        grammar = llama_grammar_init(
            grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, prompt, true);

    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int)tokens_list.size() > max_tokens_list_size)
    {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int)tokens_list.size(), max_tokens_list_size);
        return;
    }

    fprintf(stderr, "\n\n");

    for (auto id : tokens_list)
    {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    std::vector<llama_token> last_tokens(max_context_size);
    std::fill(last_tokens.begin(), last_tokens.end(), 0);

    for (auto &id : tokens_list)
    {
        last_tokens.erase(last_tokens.begin());
        last_tokens.push_back(id);
    }

    const int n_gen = std::min(32, max_context_size);

    const int n_vocab = llama_n_vocab(ctx);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    while (llama_get_kv_cache_token_count(ctx) < n_gen)
    {
        // evaluate the transformer

        if (llama_eval(ctx, tokens_list.data(), int(tokens_list.size()), llama_get_kv_cache_token_count(ctx), params.n_threads))
        {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return;
        }

        tokens_list.clear();

        // sample the next token

        llama_token new_token_id = 0;
        new_token_id = llama_sample_token(ctx, NULL, grammar, params, last_tokens, candidates);

        // is it an end of stream ?
        if (new_token_id == llama_token_eos(ctx))
        {
            fprintf(stderr, " [end of text]\n");
            break;
        }
        // print the new token :
        prompt_callback(llama_token_to_piece(ctx, new_token_id).c_str());

        // push this new token for next evaluation
        tokens_list.push_back(new_token_id);
    }
}