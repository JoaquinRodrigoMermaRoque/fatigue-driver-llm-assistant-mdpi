#include <jni.h>

#include "llama.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct PersistentLlmState {
    std::mutex mutex;
    std::atomic<bool> cancel_requested{false};
    bool backend_initialized = false;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string model_path;
};

PersistentLlmState g_state;

jstring toJString(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

std::string tokenPiece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(256);
    int32_t written = llama_token_to_piece(
        vocab,
        token,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        0,
        true
    );

    if (written < 0) {
        buffer.resize(static_cast<size_t>(-written));
        written = llama_token_to_piece(
            vocab,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            true
        );
    }

    if (written < 0) {
        return {};
    }

    return std::string(buffer.data(), static_cast<size_t>(written));
}

bool abortCallback(void * data) {
    auto * flag = static_cast<std::atomic<bool> *>(data);
    return flag != nullptr && flag->load(std::memory_order_relaxed);
}

void releasePersistentLocked() {
    if (g_state.sampler != nullptr) {
        llama_sampler_free(g_state.sampler);
        g_state.sampler = nullptr;
    }

    if (g_state.ctx != nullptr) {
        llama_free(g_state.ctx);
        g_state.ctx = nullptr;
    }

    if (g_state.model != nullptr) {
        llama_model_free(g_state.model);
        g_state.model = nullptr;
    }

    g_state.vocab = nullptr;
    g_state.model_path.clear();
    g_state.cancel_requested.store(false, std::memory_order_relaxed);

    if (g_state.backend_initialized) {
        llama_backend_free();
        g_state.backend_initialized = false;
    }
}

std::string formatPrompt(llama_model * model, const std::string & user_prompt) {
    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template == nullptr) {
        return {};
    }

    llama_chat_message message = {"user", user_prompt.c_str()};

    int32_t formatted_size = llama_chat_apply_template(
        chat_template,
        &message,
        1,
        true,
        nullptr,
        0
    );

    if (formatted_size <= 0) {
        return {};
    }

    std::vector<char> formatted(static_cast<size_t>(formatted_size) + 1);
    formatted_size = llama_chat_apply_template(
        chat_template,
        &message,
        1,
        true,
        formatted.data(),
        static_cast<int32_t>(formatted.size())
    );

    if (formatted_size <= 0) {
        return {};
    }

    return std::string(formatted.data(), static_cast<size_t>(formatted_size));
}

}  // namespace

extern "C"
JNIEXPORT jstring JNICALL
Java_com_google_mediapipe_examples_facelandmarker_LocalLlmBridge_nativeSmokeTest(
        JNIEnv * env,
        jobject /* thiz */) {
    return env->NewStringUTF("H7.10.1 JNI_OK arm64-v8a");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_google_mediapipe_examples_facelandmarker_LocalLlmBridge_nativeLlamaBackendSmokeTest(
        JNIEnv * env,
        jobject /* thiz */) {
    llama_backend_init();
    llama_backend_free();
    return env->NewStringUTF("H7.10.2 LLAMA_BACKEND_OK b11073 CPU");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_google_mediapipe_examples_facelandmarker_LocalLlmBridge_nativeModelLoadSmokeTest(
        JNIEnv * env,
        jobject /* thiz */,
        jstring model_path_java) {
    if (model_path_java == nullptr) {
        return env->NewStringUTF("H7.10.3 MODEL_LOAD_ERROR null_path");
    }

    const char * model_path = env->GetStringUTFChars(model_path_java, nullptr);
    if (model_path == nullptr) {
        return env->NewStringUTF("H7.10.3 MODEL_LOAD_ERROR utf8_path");
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(model_path_java, model_path);

    if (model == nullptr) {
        llama_backend_free();
        return env->NewStringUTF("H7.10.3 MODEL_LOAD_ERROR llama_model_load_from_file");
    }

    char description[256] = {};
    llama_model_desc(model, description, sizeof(description));

    const unsigned long long model_size =
        static_cast<unsigned long long>(llama_model_size(model));
    const unsigned long long model_params_count =
        static_cast<unsigned long long>(llama_model_n_params(model));

    char result[768] = {};
    std::snprintf(
        result,
        sizeof(result),
        "H7.10.3 MODEL_LOAD_OK desc=%s size_bytes=%llu params=%llu backend=CPU ngl=0",
        description,
        model_size,
        model_params_count
    );

    llama_model_free(model);
    llama_backend_free();

    return env->NewStringUTF(result);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_google_mediapipe_examples_facelandmarker_LocalLlmBridge_nativeGenerateSmokeTest(
        JNIEnv * env,
        jobject /* thiz */,
        jstring model_path_java,
        jstring prompt_java) {
    if (model_path_java == nullptr || prompt_java == nullptr) {
        return env->NewStringUTF("H7.10.4 GENERATE_ERROR null_argument");
    }

    const char * model_path_chars = env->GetStringUTFChars(model_path_java, nullptr);
    const char * prompt_chars = env->GetStringUTFChars(prompt_java, nullptr);

    if (model_path_chars == nullptr || prompt_chars == nullptr) {
        if (model_path_chars != nullptr) {
            env->ReleaseStringUTFChars(model_path_java, model_path_chars);
        }
        if (prompt_chars != nullptr) {
            env->ReleaseStringUTFChars(prompt_java, prompt_chars);
        }
        return env->NewStringUTF("H7.10.4 GENERATE_ERROR utf8_argument");
    }

    const std::string model_path(model_path_chars);
    const std::string user_prompt(prompt_chars);
    env->ReleaseStringUTFChars(model_path_java, model_path_chars);
    env->ReleaseStringUTFChars(prompt_java, prompt_chars);

    const auto started = std::chrono::steady_clock::now();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        llama_backend_free();
        return env->NewStringUTF("H7.10.4 GENERATE_ERROR model_load");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string formatted_prompt = formatPrompt(model, user_prompt);
    if (formatted_prompt.empty()) {
        llama_model_free(model);
        llama_backend_free();
        return env->NewStringUTF("H7.10.4 GENERATE_ERROR chat_template");
    }

    int32_t token_count = -llama_tokenize(
        vocab,
        formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        nullptr,
        0,
        true,
        true
    );

    if (token_count <= 0) {
        llama_model_free(model);
        llama_backend_free();
        return env->NewStringUTF("H7.10.4 GENERATE_ERROR tokenize_size");
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(token_count));
    const int32_t tokenized = llama_tokenize(
        vocab,
        formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        prompt_tokens.data(),
        static_cast<int32_t>(prompt_tokens.size()),
        true,
        true
    );

    if (tokenized <= 0) {
        llama_model_free(model);
        llama_backend_free();
        return env->NewStringUTF("H7.10.4 GENERATE_ERROR tokenize");
    }

    constexpr int32_t max_new_tokens = 16;
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 256;
    ctx_params.n_batch = 256;
    ctx_params.n_threads = 6;
    ctx_params.n_threads_batch = 6;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_backend_free();
        return env->NewStringUTF("H7.10.4 GENERATE_ERROR context_init");
    }

    llama_sampler * sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(
        prompt_tokens.data(),
        static_cast<int32_t>(prompt_tokens.size())
    );

    std::string response;
    int32_t generated_tokens = 0;
    llama_token next_token = LLAMA_TOKEN_NULL;
    bool decode_error = false;

    for (int32_t i = 0; i < max_new_tokens; ++i) {
        if (llama_decode(ctx, batch) != 0) {
            decode_error = true;
            break;
        }

        next_token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, next_token)) {
            break;
        }

        response += tokenPiece(vocab, next_token);
        ++generated_tokens;
        batch = llama_batch_get_one(&next_token, 1);
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    const auto ended = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(ended - started).count();

    if (decode_error || response.empty()) {
        return env->NewStringUTF("H7.10.4 GENERATE_ERROR decode_or_empty");
    }

    for (char & ch : response) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }

    char prefix[256] = {};
    std::snprintf(
        prefix,
        sizeof(prefix),
        "H7.10.4 GENERATE_OK prompt_tokens=%d generated_tokens=%d threads=4 elapsed_ms=%lld response=",
        tokenized,
        generated_tokens,
        static_cast<long long>(elapsed_ms)
    );

    return toJString(env, std::string(prefix) + response);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_google_mediapipe_examples_facelandmarker_LocalLlmBridge_nativeInitializeModel(
        JNIEnv * env,
        jobject /* thiz */,
        jstring model_path_java) {
    if (model_path_java == nullptr) {
        return env->NewStringUTF("H7.10.5 INIT_ERROR null_path");
    }

    const char * model_path_chars =
        env->GetStringUTFChars(model_path_java, nullptr);
    if (model_path_chars == nullptr) {
        return env->NewStringUTF("H7.10.5 INIT_ERROR utf8_path");
    }

    const std::string model_path(model_path_chars);
    env->ReleaseStringUTFChars(model_path_java, model_path_chars);

    std::lock_guard<std::mutex> lock(g_state.mutex);

    if (g_state.model != nullptr && g_state.model_path == model_path) {
        return env->NewStringUTF("H7.10.5 INIT_OK already_loaded");
    }

    releasePersistentLocked();

    const auto started = std::chrono::steady_clock::now();

    llama_backend_init();
    g_state.backend_initialized = true;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    g_state.model =
        llama_model_load_from_file(model_path.c_str(), model_params);
    if (g_state.model == nullptr) {
        releasePersistentLocked();
        return env->NewStringUTF("H7.10.5 INIT_ERROR model_load");
    }

    g_state.vocab = llama_model_get_vocab(g_state.model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.no_perf = false;

    g_state.ctx = llama_init_from_model(g_state.model, ctx_params);
    if (g_state.ctx == nullptr) {
        releasePersistentLocked();
        return env->NewStringUTF("H7.10.5 INIT_ERROR context_init");
    }

    llama_set_abort_callback(
        g_state.ctx,
        abortCallback,
        &g_state.cancel_requested
    );

    g_state.sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (g_state.sampler == nullptr) {
        releasePersistentLocked();
        return env->NewStringUTF("H7.10.5 INIT_ERROR sampler_init");
    }
    llama_sampler_chain_add(g_state.sampler, llama_sampler_init_greedy());

    g_state.model_path = model_path;

    const auto ended = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(ended - started).count();

    char result[256] = {};
    std::snprintf(
        result,
        sizeof(result),
        "H7.10.5 INIT_OK threads=4 n_ctx=512 elapsed_ms=%lld",
        static_cast<long long>(elapsed_ms)
    );

    return env->NewStringUTF(result);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_google_mediapipe_examples_facelandmarker_LocalLlmBridge_nativeGenerate(
        JNIEnv * env,
        jobject /* thiz */,
        jstring prompt_java,
        jint max_new_tokens_java) {
    if (prompt_java == nullptr) {
        return env->NewStringUTF("H7.10.5 GENERATE_ERROR null_prompt");
    }

    const char * prompt_chars = env->GetStringUTFChars(prompt_java, nullptr);
    if (prompt_chars == nullptr) {
        return env->NewStringUTF("H7.10.5 GENERATE_ERROR utf8_prompt");
    }

    const std::string user_prompt(prompt_chars);
    env->ReleaseStringUTFChars(prompt_java, prompt_chars);

    const int32_t max_new_tokens =
        max_new_tokens_java > 0 && max_new_tokens_java <= 128
            ? max_new_tokens_java
            : 32;

    std::lock_guard<std::mutex> lock(g_state.mutex);

    if (g_state.model == nullptr ||
        g_state.ctx == nullptr ||
        g_state.sampler == nullptr ||
        g_state.vocab == nullptr) {
        return env->NewStringUTF("H7.10.5 GENERATE_ERROR not_initialized");
    }

    g_state.cancel_requested.store(false, std::memory_order_relaxed);
    llama_memory_clear(llama_get_memory(g_state.ctx), true);
    llama_sampler_reset(g_state.sampler);

    const std::string formatted_prompt =
        formatPrompt(g_state.model, user_prompt);
    if (formatted_prompt.empty()) {
        return env->NewStringUTF("H7.10.5 GENERATE_ERROR chat_template");
    }

    int32_t token_count = -llama_tokenize(
        g_state.vocab,
        formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        nullptr,
        0,
        true,
        true
    );

    if (token_count <= 0 ||
        token_count + max_new_tokens >= static_cast<int32_t>(llama_n_ctx(g_state.ctx))) {
        return env->NewStringUTF("H7.10.5 GENERATE_ERROR context_limit");
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(token_count));
    const int32_t tokenized = llama_tokenize(
        g_state.vocab,
        formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        prompt_tokens.data(),
        static_cast<int32_t>(prompt_tokens.size()),
        true,
        true
    );

    if (tokenized <= 0) {
        return env->NewStringUTF("H7.10.5 GENERATE_ERROR tokenize");
    }

    const auto started = std::chrono::steady_clock::now();

    llama_batch batch = llama_batch_get_one(
        prompt_tokens.data(),
        static_cast<int32_t>(prompt_tokens.size())
    );

    std::string response;
    int32_t generated_tokens = 0;
    llama_token next_token = LLAMA_TOKEN_NULL;
    long long prefill_ms = -1;

    for (int32_t i = 0; i < max_new_tokens; ++i) {
        const int decode_result = llama_decode(g_state.ctx, batch);

        if (i == 0) {
            const auto first_decode_at =
                std::chrono::steady_clock::now();
            prefill_ms =
                std::chrono::duration_cast<
                    std::chrono::milliseconds
                >(first_decode_at - started).count();
        }

        if (g_state.cancel_requested.load(std::memory_order_relaxed)) {
            const auto cancelled_at =
                std::chrono::steady_clock::now();
            const auto cancel_elapsed_ms =
                std::chrono::duration_cast<
                    std::chrono::milliseconds
                >(cancelled_at - started).count();

            if (generated_tokens > 0 && !response.empty()) {
                for (char & ch : response) {
                    if (ch == '\n' || ch == '\r') {
                        ch = ' ';
                    }
                }

                char prefix[320] = {};
                std::snprintf(
                    prefix,
                    sizeof(prefix),
                    "H7.10.5 GENERATE_PARTIAL prompt_tokens=%d generated_tokens=%d threads=4 prefill_ms=%lld elapsed_ms=%lld response=",
                    tokenized,
                    generated_tokens,
                    prefill_ms,
                    static_cast<long long>(
                        cancel_elapsed_ms
                    )
                );
                return toJString(
                    env,
                    std::string(prefix) + response
                );
            }

            char cancelled[320] = {};
            std::snprintf(
                cancelled,
                sizeof(cancelled),
                "H7.10.5 GENERATE_CANCELLED prompt_tokens=%d generated_tokens=%d threads=4 prefill_ms=%lld elapsed_ms=%lld stage=%s",
                tokenized,
                generated_tokens,
                prefill_ms,
                static_cast<long long>(
                    cancel_elapsed_ms
                ),
                generated_tokens == 0
                    ? "prefill_or_first_decode"
                    : "generation"
            );
            return env->NewStringUTF(cancelled);
        }

        if (decode_result != 0) {
            const auto failed_at =
                std::chrono::steady_clock::now();
            const auto fail_elapsed_ms =
                std::chrono::duration_cast<
                    std::chrono::milliseconds
                >(failed_at - started).count();

            char failure[256] = {};
            std::snprintf(
                failure,
                sizeof(failure),
                "H7.10.5 GENERATE_ERROR decode code=%d prompt_tokens=%d generated_tokens=%d threads=4 elapsed_ms=%lld",
                decode_result,
                tokenized,
                generated_tokens,
                static_cast<long long>(
                    fail_elapsed_ms
                )
            );
            return env->NewStringUTF(failure);
        }

        next_token = llama_sampler_sample(g_state.sampler, g_state.ctx, -1);
        if (llama_vocab_is_eog(g_state.vocab, next_token)) {
            break;
        }

        response += tokenPiece(g_state.vocab, next_token);
        ++generated_tokens;
        batch = llama_batch_get_one(&next_token, 1);
    }

    const auto ended = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(ended - started).count();

    for (char & ch : response) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }

    char prefix[256] = {};
    std::snprintf(
        prefix,
        sizeof(prefix),
        "H7.10.5 GENERATE_OK prompt_tokens=%d generated_tokens=%d threads=4 prefill_ms=%lld elapsed_ms=%lld response=",
        tokenized,
        generated_tokens,
        prefill_ms,
        static_cast<long long>(elapsed_ms)
    );

    return toJString(env, std::string(prefix) + response);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_google_mediapipe_examples_facelandmarker_LocalLlmBridge_nativeCancelGeneration(
        JNIEnv * /* env */,
        jobject /* thiz */) {
    g_state.cancel_requested.store(true, std::memory_order_relaxed);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_google_mediapipe_examples_facelandmarker_LocalLlmBridge_nativeReleaseModel(
        JNIEnv * env,
        jobject /* thiz */) {
    g_state.cancel_requested.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_state.mutex);
    releasePersistentLocked();
    return env->NewStringUTF("H7.10.5 RELEASE_OK");
}
