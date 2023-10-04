#ifdef __cplusplus
extern "C"
{
#endif
    struct llama_model;
    struct llama_context;
    struct llama_parts
    {
        struct llama_model *model;
        struct llama_context *ctx;
    };
    struct llama_parts *load_model(char *args);
    void prompt(char *prompt, struct llama_parts *parts, char *args, void (*prompt_callback)(const char *response), char *grammar_str);

#ifdef __cplusplus
}
#endif