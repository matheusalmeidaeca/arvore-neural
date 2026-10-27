package com.arvoreneural.app

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.arvoreneural.app.ai.Cerebro
import com.arvoreneural.app.ai.TokenListener
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import kotlin.random.Random

/**
 * Cérebro = um modelo GGUF (ex.: Qwen) rodando dentro do app via llama.cpp.
 * Aqui ficam o estado da tela, a cópia do arquivo GGUF e a montagem do prompt.
 */
object CerebroState {
    var status by mutableStateOf("Nenhum modelo carregado. Escolha um arquivo .gguf.")
    var modelInfo by mutableStateOf("")
    var hasFile by mutableStateOf(false)
    var loaded by mutableStateOf(false)
    var working by mutableStateOf(false)      // copiando ou carregando
    var generating by mutableStateOf(false)
    var stats by mutableStateOf("")
    private var crashAviso by mutableStateOf("")
    var input by mutableStateOf("")
    val msgs = mutableStateListOf<Msg>()

    private const val N_CTX = 2048
    private const val THREADS = 4
    private const val MAX_NEW = 400
    private const val PROMPT_CHARS = 4500
    private const val SYSTEM = "Você é o cérebro do app Árvore Neural. Responda em português do Brasil, de forma curta e clara."

    private fun modelFile(ctx: Context) = File(File(ctx.filesDir, "modelos"), "modelo.gguf")

    private fun flagFile(ctx: Context) = File(ctx.filesDir, "gerando.flag")

    fun refresh(ctx: Context) {
        if (flagFile(ctx).exists()) {
            flagFile(ctx).delete()
            crashAviso = "O app fechou durante a última resposta. Isso costuma ser instrução do processador incompatível com o llama.cpp."
        }
        val f = modelFile(ctx)
        hasFile = f.exists() && f.length() > 0
        if (!loaded && !working) {
            val nome = ctx.getSharedPreferences("cerebro_prefs", Context.MODE_PRIVATE).getString("nome", "") ?: ""
            status = if (hasFile) "Modelo guardado${if (nome.isNotEmpty()) ": $nome" else ""}. Toque em Carregar."
            else "Nenhum modelo carregado. Escolha um arquivo .gguf."
        }
        if (crashAviso.isNotEmpty() && !loaded && !working) status = crashAviso + " " + status
        if (!Cerebro.libOk) status = "A parte nativa (llama.cpp) não foi encontrada neste APK."
    }

    /** Copia o GGUF escolhido para dentro do app (precisa de espaço livre igual ao tamanho do arquivo). */
    fun importModel(ctx: Context, uri: Uri) {
        if (working || generating) return
        working = true
        status = "Copiando o modelo…"
        Engine.scope.launch(Dispatchers.IO) {
            try {
                val dir = File(ctx.filesDir, "modelos").apply { mkdirs() }
                val tmp = File(dir, "modelo.gguf.tmp")
                var nome = ""
                ctx.contentResolver.query(uri, null, null, null, null)?.use { c ->
                    val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (i >= 0 && c.moveToFirst()) nome = c.getString(i) ?: ""
                }
                var copied = 0L
                ctx.contentResolver.openInputStream(uri)!!.use { inp ->
                    val head = ByteArray(4)
                    val got = inp.read(head)
                    if (got < 4 || String(head, Charsets.ISO_8859_1) != "GGUF") {
                        throw RuntimeException("Esse arquivo não é um GGUF (precisa começar com GGUF).")
                    }
                    tmp.outputStream().use { out ->
                        out.write(head)
                        copied = 4
                        val buf = ByteArray(1 shl 20)
                        var lastShown = 0L
                        while (true) {
                            val r = inp.read(buf)
                            if (r < 0) break
                            out.write(buf, 0, r)
                            copied += r
                            if (copied - lastShown >= (32L shl 20)) {
                                lastShown = copied
                                status = "Copiando o modelo… ${copied shr 20} MB"
                            }
                        }
                    }
                }
                Cerebro.nativeFree()
                loaded = false
                val dest = modelFile(ctx)
                if (dest.exists()) dest.delete()
                if (!tmp.renameTo(dest)) throw RuntimeException("Não consegui salvar o modelo.")
                ctx.getSharedPreferences("cerebro_prefs", Context.MODE_PRIVATE).edit().putString("nome", nome).apply()
                status = "Modelo copiado (${copied shr 20} MB). Toque em Carregar."
            } catch (e: Throwable) {
                File(File(ctx.filesDir, "modelos"), "modelo.gguf.tmp").delete()
                status = "Erro ao copiar: ${e.message ?: e.javaClass.simpleName}"
            } finally {
                working = false
                hasFile = modelFile(ctx).let { it.exists() && it.length() > 0 }
            }
        }
    }

    fun load(ctx: Context) {
        if (working || generating || loaded) return
        if (!Cerebro.libOk) { refresh(ctx); return }
        working = true
        status = "Carregando o modelo na memória…"
        Engine.scope.launch(Dispatchers.Default) {
            try {
                val info = Cerebro.nativeLoad(modelFile(ctx).absolutePath, N_CTX, THREADS)
                modelInfo = info
                loaded = true
                status = "Cérebro pronto."
            } catch (e: Throwable) {
                loaded = false
                status = "Erro ao carregar: ${e.message ?: e.javaClass.simpleName}"
            } finally {
                working = false
            }
        }
    }

    fun unload() {
        if (generating) return
        Cerebro.nativeFree()
        loaded = false
        modelInfo = ""
        status = "Memória liberada. Toque em Carregar para usar de novo."
    }

    fun stop() {
        if (generating) Cerebro.nativeStop()
    }

    fun clear() {
        if (!generating) { msgs.clear(); stats = "" }
    }

    /** Formato ChatML (o do Qwen), com o bloco de pensamento vazio para a resposta sair direta. */
    private fun buildPrompt(): String {
        val turns = ArrayList<Msg>()
        var total = 0
        for (i in msgs.indices.reversed()) {
            val m = msgs[i]
            if (i == msgs.lastIndex && !m.fromUser && m.text.isEmpty()) continue  // resposta em branco que vai ser escrita
            if (total + m.text.length > PROMPT_CHARS && turns.isNotEmpty()) break
            total += m.text.length
            turns.add(0, m)
        }
        val sb = StringBuilder()
        sb.append("<|im_start|>system\n").append(SYSTEM).append("<|im_end|>\n")
        for (m in turns) {
            sb.append("<|im_start|>").append(if (m.fromUser) "user" else "assistant").append('\n')
            sb.append(m.text).append("<|im_end|>\n")
        }
        sb.append("<|im_start|>assistant\n")
        // Só os Qwen3 em diante conhecem o bloco de pensamento; o Qwen2.5 e outros não.
        if (modelInfo.startsWith("qwen3", ignoreCase = true)) sb.append("<think>\n\n</think>\n\n")
        return sb.toString()
    }

    fun send(ctx: Context, text: String) {
        if (!loaded || generating || working) return
        msgs.add(Msg(true, text))
        msgs.add(Msg(false, ""))
        val slot = msgs.lastIndex
        generating = true
        stats = ""
        Engine.scope.launch(Dispatchers.Default) {
            val sb = StringBuilder()
            var n = 0
            var probSum = 0.0
            var probN = 0
            val t0 = System.nanoTime()
            try {
                flagFile(ctx).writeText("1")
                val prompt = buildPrompt()
                Cerebro.nativeGenerate(
                    prompt.toByteArray(Charsets.UTF_8), MAX_NEW, 0.7f, 0.9f, Random.nextInt(),
                    TokenListener { bytes, prob ->
                        if (prob == -2f) { stats = String(bytes, Charsets.UTF_8); return@TokenListener true }
                        sb.append(String(bytes, Charsets.UTF_8))
                        n++
                        if (prob >= 0f) { probSum += prob; probN++ }
                        if (slot < msgs.size) msgs[slot] = Msg(false, sb.toString())
                        true
                    }
                )
                val secs = (System.nanoTime() - t0) / 1e9
                val conf = if (probN > 0) " | confiança média ${(probSum / probN * 100).toInt()}%" else ""
                stats = "$n tokens em ${"%.1f".format(secs)} s (${"%.1f".format(n / maxOf(secs, 0.001))} tok/s)$conf"
                if (sb.isEmpty() && slot < msgs.size) msgs[slot] = Msg(false, "(sem resposta)")
            } catch (e: Throwable) {
                val aviso = "⚠️ ${e.message ?: e.javaClass.simpleName}"
                if (slot < msgs.size) msgs[slot] = Msg(false, if (sb.isEmpty()) aviso else sb.toString() + "\n" + aviso)
            } finally {
                flagFile(ctx).delete()
                generating = false
            }
        }
    }
}
