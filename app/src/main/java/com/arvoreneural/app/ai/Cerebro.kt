package com.arvoreneural.app.ai

/** Chamado pelo C++ a cada token. Devolva false para interromper a resposta. */
fun interface TokenListener {
    fun onToken(bytes: ByteArray, prob: Float): Boolean
}

/**
 * Ponte com o llama.cpp (libsayotollm.so). Só chamadas nativas.
 * A lógica (copiar o GGUF, montar o prompt, tela) fica em CerebroState.kt.
 */
object Cerebro {
    val libOk: Boolean = try {
        System.loadLibrary("sayotollm")
        true
    } catch (e: Throwable) {
        false
    }

    /** Carrega o GGUF. Devolve uma linha de descrição; lança RuntimeException se falhar. */
    external fun nativeLoad(path: String, nCtx: Int, threads: Int): String

    /** Gera a resposta chamando [listener] a cada token. Devolve quantos tokens gerou. */
    external fun nativeGenerate(
        prompt: ByteArray, maxNew: Int, temp: Float, topP: Float, seed: Int, listener: TokenListener
    ): Int

    external fun nativeStop()
    external fun nativeFree()
}
