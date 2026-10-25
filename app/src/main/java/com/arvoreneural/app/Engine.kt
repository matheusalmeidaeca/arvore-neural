package com.arvoreneural.app

import android.content.Context
import android.content.Intent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import com.arvoreneural.app.ai.HashEmbedder
import com.arvoreneural.app.ai.MiniLab
import com.arvoreneural.app.ai.Osmose
import com.arvoreneural.app.ai.Sizes
import com.arvoreneural.app.ai.Tree
import com.arvoreneural.app.ai.miniSelfTest
import com.arvoreneural.app.data.Db
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/** Estado do treino manual (aba Treino). Vive no Engine, então trocar de aba não perde nada. */
class TState {
    var status by mutableStateOf("")
    var training by mutableStateOf(false)
    var busy by mutableStateOf(false)
    var output by mutableStateOf("")
    var corpusInfo by mutableStateOf("")
    var prompt by mutableStateOf("")
    val stop = AtomicBoolean(false)
}

/**
 * v7.1: tudo que não pode morrer ao trocar de aba mora aqui (processo inteiro), não nas telas.
 * Respostas, treino manual e importações rodam no `scope` do Engine; o TrainService mantém o processo vivo.
 */
object Engine {
    lateinit var db: Db
    lateinit var tree: Tree
    lateinit var lab: MiniLab
    lateinit var osmose: Osmose
    @Volatile var ready = false
    private lateinit var appCtx: Context

    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val msgs = mutableStateListOf<Msg>()
    var liked by mutableStateOf(setOf<Int>())
    var pending by mutableIntStateOf(0)
    var correcting by mutableIntStateOf(-1)
    var correctText by mutableStateOf("")
    var version by mutableIntStateOf(0)
    val ts = TState()

    // rascunhos e mensagens das telas (sobrevivem à troca de aba)
    var chatInput by mutableStateOf("")
    var learnText by mutableStateOf("")
    var osmoseMsg by mutableStateOf("")
    var backupMsg by mutableStateOf("")
    var treeMsg by mutableStateOf("")
    var selfTestMsg by mutableStateOf("")
    var scriptPy by mutableStateOf(ScriptLab.DEFAULT_PY)
    var scriptJs by mutableStateOf(ScriptLab.DEFAULT_JS)
    var scriptIn by mutableStateOf(ScriptLab.DEFAULT_IN)
    var scriptOut by mutableStateOf("")
    var scriptBusy by mutableStateOf(false)
    var size by mutableIntStateOf(0)

    private val chatLock = Any()

    @Synchronized
    fun init(ctx: Context) {
        if (ready) return
        appCtx = ctx.applicationContext
        val d = Db(appCtx)
        val tr = Tree(d, HashEmbedder())
        // tamanho do modelo: se a tentativa anterior com um tamanho grande não terminou (app morto por falta
        // de memória), volta ao Mini para o app não ficar preso num ciclo de falhas
        val sp = appCtx.getSharedPreferences("arvore_prefs", Context.MODE_PRIVATE)
        var sz = sp.getInt("model_size", 0).coerceIn(0, Sizes.names.size - 1)
        var note = ""
        if (sp.getInt("size_trying", -1) >= 0) {
            sp.edit().putInt("model_size", 0).putInt("size_trying", -1).commit()
            sz = 0
            note = "O modelo grande não coube na memória do celular. Voltei ao Mini."
        }
        if (sz > 0) sp.edit().putInt("size_trying", sz).commit()
        val l = try {
            MiniLab(d, sz)
        } catch (e: OutOfMemoryError) {
            sp.edit().putInt("model_size", 0).putInt("size_trying", -1).commit()
            sz = 0
            note = "Faltou memória para o modelo grande. Voltei ao Mini."
            MiniLab(d, 0)
        }
        sp.edit().putInt("size_trying", -1).commit()
        size = sz
        tr.lab = l
        db = d
        tree = tr
        lab = l
        osmose = Osmose(appCtx, d, tr, l)
        osmose.start()
        osmose.seedIfNeeded()
        scriptPy = sp.getString("script_py", ScriptLab.DEFAULT_PY) ?: ScriptLab.DEFAULT_PY
        scriptJs = sp.getString("script_js", ScriptLab.DEFAULT_JS) ?: ScriptLab.DEFAULT_JS
        scriptIn = sp.getString("script_in", ScriptLab.DEFAULT_IN) ?: ScriptLab.DEFAULT_IN
        // autoteste (gradientes, especialistas, crescimento): roda sozinho uma vez a cada versão nova do app
        val stVer = 10
        if (sp.getInt("selftest_ver", 0) != stVer) {
            scope.launch(Dispatchers.Default) {
                val r = try {
                    miniSelfTest()
                } catch (e: Throwable) {
                    "Autoteste não rodou: " + (e.message ?: e.javaClass.simpleName)
                }
                sp.edit().putInt("selftest_ver", stVer).putString("selftest_msg", r).apply()
                selfTestMsg = r
            }
        } else {
            selfTestMsg = sp.getString("selftest_msg", "") ?: ""
        }
        ts.status = if (note.isNotEmpty()) note else "Passos treinados: " + l.gpt.step
        ready = true
        scope.launch(Dispatchers.IO) {
            val h = db.chatHistory()
            val n = lab.corpusChars()
            withContext(Dispatchers.Main) { msgs.addAll(0, h.map { Msg(it.first, it.second) }) }
            ts.corpusInfo = "Corpus de treino em massa: " + n + " caracteres"
        }
    }

    fun saveScripts() {
        appCtx.getSharedPreferences("arvore_prefs", Context.MODE_PRIVATE).edit()
            .putString("script_py", scriptPy).putString("script_js", scriptJs).putString("script_in", scriptIn).apply()
    }

    /** Roda o Python e depois o JavaScript, fora da tela principal (o primeiro uso demora um pouco). */
    fun runScripts() {
        if (scriptBusy) return
        scriptBusy = true
        scriptOut = "Rodando… (a primeira vez o Python demora alguns segundos para ligar)"
        saveScripts()
        val py = scriptPy
        val js = scriptJs
        val ins = ScriptLab.parseInputs(scriptIn)
        scope.launch(Dispatchers.Default) {
            val r = try {
                ScriptLab.run(appCtx, py, js, ins)
            } catch (e: Throwable) {
                "❌ Erro: " + (e.message ?: e.javaClass.simpleName)
            }
            scriptOut = r
            scriptBusy = false
        }
    }

    fun wantsService(): Boolean = ready && (osmose.enabled || pending > 0 || ts.training)

    fun ensureService(ctx: Context) {
        if (!wantsService()) return
        try {
            ContextCompat.startForegroundService(ctx, Intent(ctx, TrainService::class.java))
        } catch (e: Throwable) {
        }
    }

    // ------------------------------------------------------------------ conversa

    fun send(text: String) {
        val tx = text.trim()
        if (tx.isEmpty()) return
        msgs.add(Msg(true, tx))
        pending++
        osmose.hold(true)
        ensureService(appCtx)
        scope.launch(Dispatchers.IO) {
            var a = ""
            try {
                synchronized(chatLock) {
                    db.addChat(true, tx)
                    a = osmose.chatTurn(tx)
                    db.addChat(false, a)
                }
            } catch (e: Throwable) {
                a = "Deu erro ao responder. Tente de novo."
            }
            withContext(Dispatchers.Main) {
                msgs.add(Msg(false, a))
                pending--
                version++
            }
            osmose.hold(false)
        }
    }

    fun clearChat() {
        msgs.clear()
        liked = emptySet()
        scope.launch(Dispatchers.IO) { db.clearChat() }
    }

    fun correct(idx: Int, right: String) {
        liked = liked + idx
        val snap = msgs.map { it.fromUser to it.text }
        scope.launch(Dispatchers.IO) { osmose.correct(snap, idx, right) }
    }

    fun like(idx: Int) {
        if (idx in liked) return
        liked = liked + idx
        val snap = msgs.map { it.fromUser to it.text }
        scope.launch(Dispatchers.IO) { osmose.like(snap, idx) }
    }

    // ------------------------------------------------------------------ treino manual

    fun startTrain(total: Int) {
        if (ts.busy) return
        ts.busy = true
        ts.training = true
        ts.stop.set(false)
        osmose.pauseNow()
        osmose.manual.set(true)
        osmose.hold(true)
        ensureService(appCtx)
        scope.launch(Dispatchers.Default) {
            val msg = try {
                if (lab.corpusSize() < 80) {
                    "Ensine mais texto (ou importe um arquivo) antes de treinar."
                } else {
                    val t0 = System.currentTimeMillis()
                    lab.train(total, 4, -1L, ts.stop) { i, loss ->
                        val sec = (System.currentTimeMillis() - t0) / 1000
                        ts.status = "passo " + lab.gpt.step + " | perda " + "%.3f".format(loss) +
                            " | " + i + "/" + total + " | " + sec + "s | " + "%.1f".format(sec.toDouble() / i) + " s/passo"
                    }
                    "Treino concluído: " + lab.gpt.step + " passos no total, perda " + "%.3f".format(lab.gpt.lastLoss) + "."
                }
            } catch (e: Throwable) {
                "Erro no treino: " + (e.message ?: "")
            }
            ts.status = msg
            osmose.manual.set(false)
            osmose.hold(false)
            ts.training = false
            ts.busy = false
        }
    }

    // ------------------------------------------------------------------ tamanho do modelo

    /** Grava o novo tamanho e para tudo. O app precisa fechar e abrir de novo; os pesos do tamanho antigo ficam no banco até o novo modelo salvar. */
    fun changeSize(n: Int): String {
        return try {
            ts.stop.set(true)
            osmose.pauseNow()
            osmose.frozen = true
            lab.frozen = true
            Thread.sleep(1500)
            appCtx.getSharedPreferences("arvore_prefs", Context.MODE_PRIVATE)
                .edit().putInt("model_size", n).putInt("size_trying", -1).commit()
            "Tamanho trocado para " + Sizes.names[n] + ". O app vai fechar; abra de novo."
        } catch (e: Throwable) {
            "Erro ao trocar o tamanho: " + (e.message ?: "")
        }
    }

    // ------------------------------------------------------------------ backup do banco (pesos, corpus, árvore, conversa)

    private fun dbFile(): File = appCtx.getDatabasePath("arvore.db")

    fun backupTo(uri: android.net.Uri): String {
        return try {
            lab.saveNow()
            synchronized(lab.lock) {
                synchronized(db) {
                    appCtx.contentResolver.openOutputStream(uri)?.use { out ->
                        dbFile().inputStream().use { it.copyTo(out) }
                    } ?: return "Não consegui abrir o destino."
                }
            }
            "Backup salvo (" + (dbFile().length() / 1024) + " KB)."
        } catch (e: Throwable) {
            "Erro no backup: " + (e.message ?: "")
        }
    }

    /** Devolve true se restaurou. Depois disso o app precisa fechar e abrir de novo. */
    fun restoreFrom(uri: android.net.Uri): Pair<Boolean, String> {
        val tmp = File(appCtx.cacheDir, "restore.db")
        try {
            appCtx.contentResolver.openInputStream(uri)?.use { inp ->
                tmp.outputStream().use { inp.copyTo(it) }
            } ?: return false to "Não consegui abrir o arquivo."
            val head = ByteArray(15)
            val n = tmp.inputStream().use { it.read(head) }
            if (n < 15 || String(head, Charsets.ISO_8859_1) != "SQLite format 3") {
                tmp.delete()
                return false to "Esse arquivo não é um backup da Árvore Neural."
            }
            ts.stop.set(true)
            osmose.pauseNow()
            osmose.frozen = true
            lab.frozen = true
            Thread.sleep(1500)
            synchronized(lab.lock) {
                synchronized(db) {
                    db.close()
                    val f = dbFile()
                    tmp.copyTo(f, overwrite = true)
                    File(f.path + "-wal").delete()
                    File(f.path + "-shm").delete()
                    File(f.path + "-journal").delete()
                }
            }
            tmp.delete()
            return true to "Backup restaurado. O app vai fechar; abra de novo."
        } catch (e: Throwable) {
            return false to "Erro ao restaurar: " + (e.message ?: "")
        }
    }
}
