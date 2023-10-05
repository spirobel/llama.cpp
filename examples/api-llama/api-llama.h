#include "stdint.h"
#ifdef __cplusplus
extern "C"
{
#endif
    struct llama_model;
    struct llama_model *load_model(char *args);
    void prompt(char *prompt, struct llama_model *model, char *args, void (*prompt_callback)(const char *response, uint64_t len));

#ifdef __cplusplus
}
#endif