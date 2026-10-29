package com.arvoreneural.app.ai

import android.content.Context
import android.net.Uri
import com.google.ai.edge.litertlm.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.io.File

object CerebroLite {
    private var engine: Engine? = null
    private var conversa: Conversation? = null

    val pronto get() = conversa != null

    fun copiar(ctx: Context, uri: Uri): String {
        val destino = File(ctx.filesDir, "modelo.litertlm")
        ctx.contentResolver.openInputStream(uri)!!.use { i ->
            destino.outputStream().use { o -> i.copyTo(o) }
        }
        return destino.path
    }

    fun carregar(caminho: String) {
        liberar()
        val e = Engine(EngineConfig(modelPath = caminho))
        e.initialize()
        engine = e
        conversa = e.createConversation()
    }

    fun responder(texto: String): Flow<String> =
        conversa!!.sendMessageAsync(texto).map { it.toString() }

    fun liberar() {
        conversa?.close(); engine?.close()
        conversa = null; engine = null
    }
}
