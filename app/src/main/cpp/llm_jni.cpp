// Ponte JNI mínima entre o app (Kotlin) e o llama.cpp.
// Fluxo: nativeLoad (carrega o GGUF) -> nativeGenerate (gera token a token, chamando o
// Kotlin a cada token com o texto e a probabilidade) -> nativeStop / nativeFree.
// Cada geração cria um contexto novo: assim não dependemos da API de cache de memória.
#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "llama.h"

#define TAG "SayotoLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

static llama_model *g_model = nullptr;
static int g_nctx = 2048;
static int g_threads = 4;
static bool g_backend = false;
static std::atomic<bool> g_stop{false};
static std::atomic<bool> g_busy{false};

static void throw_rt(JNIEnv *env, const std::string &msg) {
    jclass c = env->FindClass("java/lang/RuntimeException");
    if (c) env->ThrowNew(c, msg.c_str());
}

// Quantos bytes do começo de s formam caracteres UTF-8 completos (o resto é um caractere
// cortado no meio, que espera o próximo token). Evita mandar meio acento para o Kotlin.
static size_t utf8_complete_prefix(const std::string &s) {
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char) s[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        if (len == 0) { i += 1; continue; }  // byte solto: segue (o Kotlin troca por �)
        if (i + len > n) break;               // caractere incompleto no fim
        i += len;
    }
    return i;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arvoreneural_app_ai_Cerebro_nativeLoad(JNIEnv *env, jobject, jstring jpath, jint nctx, jint threads) {
    if (g_busy.load()) { throw_rt(env, "O cerebro esta gerando; espere terminar."); return nullptr; }
    const char *cpath = env->GetStringUTFChars(jpath, nullptr);
    std::string path(cpath);
    env->ReleaseStringUTFChars(jpath, cpath);

    if (!g_backend) { llama_backend_init(); g_backend = true; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path.c_str(), mp);
    if (!g_model) {
        throw_rt(env, "Nao consegui carregar o GGUF (arquivo corrompido, sem memoria ou modelo novo demais para esta versao do llama.cpp).");
        return nullptr;
    }
    g_nctx = nctx;
    g_threads = threads;

    char desc[256] = {0};
    llama_model_desc(g_model, desc, sizeof(desc));
    double params_m = (double) llama_model_n_params(g_model) / 1e6;
    double size_mb = (double) llama_model_size(g_model) / (1024.0 * 1024.0);
    char info[512];
    snprintf(info, sizeof(info), "%s | %.0f M parametros | %.0f MB", desc, params_m, size_mb);
    LOGI("modelo carregado: %s", info);
    return env->NewStringUTF(info);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_arvoreneural_app_ai_Cerebro_nativeGenerate(JNIEnv *env, jobject, jbyteArray jprompt, jint maxNew,
                                                    jfloat temp, jfloat topP, jint seed, jobject listener) {
    if (!g_model) { throw_rt(env, "Nenhum modelo carregado."); return -1; }
    bool expected = false;
    if (!g_busy.compare_exchange_strong(expected, true)) { throw_rt(env, "Ja esta gerando."); return -1; }
    struct BusyGuard { ~BusyGuard() { g_busy.store(false); } } guard;
    g_stop.store(false);

    const llama_vocab *vocab = llama_model_get_vocab(g_model);

    // 1) prompt (bytes UTF-8) -> tokens
    jsize plen = env->GetArrayLength(jprompt);
    std::string prompt((size_t) plen, '\0');
    if (plen > 0) env->GetByteArrayRegion(jprompt, 0, plen, reinterpret_cast<jbyte *>(&prompt[0]));
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), true, true);
    if (n < 0) {
        toks.resize((size_t) (-n));
        n = llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), true, true);
    }
    if (n <= 0) { throw_rt(env, "Falha ao converter o texto em tokens."); return -1; }
    toks.resize((size_t) n);

    int maxTokens = maxNew;
    if (n + 8 > g_nctx) { throw_rt(env, "Conversa grande demais para o contexto; limpe a conversa."); return -1; }
    if (n + maxTokens > g_nctx) maxTokens = g_nctx - n;

    // 2) contexto novo para esta resposta
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) g_nctx;
    cp.n_batch = 512;
    cp.n_threads = g_threads;
    cp.n_threads_batch = g_threads;
    llama_context *ctx = llama_init_from_model(g_model, cp);
    if (!ctx) { throw_rt(env, "Sem memoria para criar o contexto (tente um contexto menor)."); return -1; }

    // 3) amostrador
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temp <= 0.01f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t) seed));
    }

    jclass lcls = env->GetObjectClass(listener);
    jmethodID mid = env->GetMethodID(lcls, "onToken", "([BF)Z");

    int produced = 0;
    const auto T0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count(); };
    auto status = [&](const std::string &m) {
        if (!mid) return;
        jbyteArray a = env->NewByteArray((jsize) m.size());
        env->SetByteArrayRegion(a, 0, (jsize) m.size(), reinterpret_cast<const jbyte *>(m.data()));
        env->CallBooleanMethod(listener, mid, a, (jfloat) -2.0f);
        env->DeleteLocalRef(a);
    };
    status("lendo o prompt: " + std::to_string(n) + " tokens...");
    bool failed = false;

    // 4) lê o prompt em pedaços
    for (int i = 0; i < n && !g_stop.load(); i += (int) cp.n_batch) {
        int cnt = std::min<int>((int) cp.n_batch, n - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, cnt);
        if (llama_decode(ctx, b) != 0) { failed = true; break; }
    }

    status("prompt lido em " + std::to_string(secs()) + " s");
    // 5) escreve token a token
    const int nv = llama_vocab_n_tokens(vocab);
    std::string pending;
    while (!failed && !g_stop.load() && produced < maxTokens) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) { status("o modelo encerrou no token " + std::to_string(produced) + " (fim de texto)"); break; }

        // probabilidade que o modelo deu a esta palavra (softmax simples dos logits)
        float prob = -1.0f;
        const float *lg = llama_get_logits_ith(ctx, -1);
        if (lg && tok >= 0 && tok < nv) {
            float mx = lg[0];
            for (int k = 1; k < nv; k++) if (lg[k] > mx) mx = lg[k];
            double sum = 0.0;
            for (int k = 0; k < nv; k++) sum += std::exp((double) (lg[k] - mx));
            prob = (float) (std::exp((double) (lg[tok] - mx)) / sum);
        }

        char buf[256];
        int len = llama_token_to_piece(vocab, tok, buf, (int32_t) sizeof(buf), 0, false);
        if (len < 0) {
            std::string big((size_t) (-len), '\0');
            int len2 = llama_token_to_piece(vocab, tok, &big[0], (int32_t) big.size(), 0, false);
            if (len2 > 0) pending.append(big.data(), (size_t) len2);
        } else if (len > 0) {
            pending.append(buf, (size_t) len);
        }
        produced++;
        if (produced == 1) status("1o token em " + std::to_string(secs()) + " s");

        bool cont = true;
        size_t k = utf8_complete_prefix(pending);
        if (k > 0 && mid) {
            jbyteArray arr = env->NewByteArray((jsize) k);
            env->SetByteArrayRegion(arr, 0, (jsize) k, reinterpret_cast<const jbyte *>(pending.data()));
            jboolean r = env->CallBooleanMethod(listener, mid, arr, (jfloat) prob);
            env->DeleteLocalRef(arr);
            pending.erase(0, k);
            if (env->ExceptionCheck() || r == JNI_FALSE) cont = false;
        }
        if (!cont) break;

        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, b) != 0) { failed = true; break; }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    if (failed && !env->ExceptionCheck()) { throw_rt(env, "O llama.cpp falhou ao processar (llama_decode)."); return -1; }
    return produced;
}

extern "C" JNIEXPORT void JNICALL
Java_com_arvoreneural_app_ai_Cerebro_nativeStop(JNIEnv *, jobject) { g_stop.store(true); }

extern "C" JNIEXPORT void JNICALL
Java_com_arvoreneural_app_ai_Cerebro_nativeFree(JNIEnv *, jobject) {
    g_stop.store(true);
    if (g_busy.load()) return;  // não libera no meio de uma resposta
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
}
