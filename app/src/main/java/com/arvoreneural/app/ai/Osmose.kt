package com.arvoreneural.app.ai

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import com.arvoreneural.app.data.Db
import java.util.concurrent.atomic.AtomicBoolean

private const val TU = "@u "
private const val TA = "@a "

/**
 * Osmose (v7): o app aprende sozinho, sem você tocar em "Treinar".
 *  1) Tudo que você escreve na Conversa vira texto de treino do transformer.
 *  2) Enquanto o app está aberto, um laço treina em passos pequenos (com pausas, para não esquentar).
 *  3) Textos e conversas do WhatsApp compartilhados com o app são absorvidos (corpus + nós da árvore).
 *  4) A resposta da Conversa é uma só: o transformer lê o diálogo recente e a árvore escolhe o galho.
 */
class Osmose(
    private val ctx: Context,
    private val db: Db,
    private val tree: Tree,
    private val lab: MiniLab
) {
    private val prefs = ctx.getSharedPreferences("arvore_prefs", Context.MODE_PRIVATE)
    private val stopFlag = AtomicBoolean(false)
    private val started = AtomicBoolean(false)

    /** Ligado pelo botão "Treinar" da aba Transformer: a osmose espera. */
    val manual = AtomicBoolean(false)

    @Volatile private var foreground = false
    @Volatile private var why = ""
    @Volatile private var working = false
    @Volatile private var dirtySteps = 0

    @Volatile var enabled: Boolean = prefs.getBoolean("osmose", true)
        set(v) {
            field = v
            prefs.edit().putBoolean("osmose", v).apply()
            if (!v) stopFlag.set(true)
        }

    @Volatile var myName: String = prefs.getString("my_name", "") ?: ""
        set(v) {
            field = v
            prefs.edit().putString("my_name", v).apply()
        }

    /** false: a IA imita as outras pessoas da conversa; true: a IA imita você. Vale para as próximas importações. */
    @Volatile var aiPlaysMe: Boolean = prefs.getBoolean("ai_plays_me", false)
        set(v) {
            field = v
            prefs.edit().putBoolean("ai_plays_me", v).apply()
        }

    @Volatile var detail: Boolean = prefs.getBoolean("detail", false)
        set(v) {
            field = v
            prefs.edit().putBoolean("detail", v).apply()
        }

    // ------------------------------------------------------------------ laço de treino

    fun start() {
        if (started.getAndSet(true)) return
        val th = Thread { loop() }
        th.isDaemon = true
        th.start()
    }

    fun setForeground(v: Boolean) {
        foreground = v
        if (!v) {
            stopFlag.set(true)
            Thread {
                try {
                    lab.saveNow()
                    dirtySteps = 0
                } catch (e: Throwable) {
                }
            }.start()
        }
    }

    /** Interrompe o passo de osmose em andamento (usado antes de um treino manual). */
    fun pauseNow() {
        stopFlag.set(true)
    }

    private fun loop() {
        while (true) {
            var pause = 3000L
            try {
                if (!enabled) {
                    why = "desligada"
                } else if (!foreground) {
                    why = "app em segundo plano"
                } else if (manual.get()) {
                    why = "treino manual em andamento"
                } else if (!canTrain()) {
                    // canTrain já preencheu o motivo
                } else if (db.corpusChars() < 200 && db.nodeCount() == 0) {
                    why = "nada para aprender ainda (converse, ensine ou importe uma conversa)"
                } else {
                    why = ""
                    stopFlag.set(false)
                    working = true
                    val t0 = System.currentTimeMillis()
                    val n = lab.train(10, 4, -1L, stopFlag, false) { _, _ -> }
                    working = false
                    dirtySteps += n
                    if (dirtySteps >= 200) {
                        lab.saveNow()
                        dirtySteps = 0
                    }
                    // descansa o dobro do tempo trabalhado: o celular não esquenta nem gasta bateria demais
                    pause = maxOf(1500L, 2 * (System.currentTimeMillis() - t0))
                }
            } catch (e: Throwable) {
                working = false
                pause = 5000L
            }
            try {
                Thread.sleep(pause)
            } catch (e: InterruptedException) {
            }
        }
    }

    private fun canTrain(): Boolean {
        val b = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        if (b != null) {
            val lvl = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = b.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
            val st = b.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            val temp = b.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)
            val charging = st == BatteryManager.BATTERY_STATUS_CHARGING || st == BatteryManager.BATTERY_STATUS_FULL
            val pct = if (lvl >= 0 && scale > 0) lvl * 100 / scale else 100
            if (!charging && pct < 40) {
                why = "bateria em " + pct + "% (conecte na tomada para treinar)"
                return false
            }
            if (temp >= 410) {
                why = "celular quente (" + (temp / 10) + " °C), esperando esfriar"
                return false
            }
        }
        return true
    }

    fun status(): String {
        val g = lab.gpt
        val loss = if (g.lastLoss.isNaN()) "—" else "%.2f".format(g.lastLoss)
        val st = when {
            !enabled -> "🌊 Osmose desligada"
            working -> "🌊 Osmose treinando…"
            why.isNotEmpty() -> "🌊 Osmose em pausa: " + why
            else -> "🌊 Osmose ligada"
        }
        return st + "\npassos " + g.step + " | perda " + loss + " | corpus " + db.corpusChars() +
            " caracteres | nós " + db.nodeCount()
    }

    // ------------------------------------------------------------------ absorver textos

    /** Texto ou conversa do WhatsApp (compartilhado ou escolhido num arquivo). Devolve um resumo para mostrar. */
    fun absorb(raw: String): String {
        if (raw.isBlank()) return "Nada para absorver."
        if (Importer.isWhatsApp(raw)) {
            val me = myName.trim()
            if (me.isEmpty()) {
                return "É uma conversa do WhatsApp. Digite seu nome na aba Aprender (seção Osmose) e compartilhe de novo."
            }
            val lines = Importer.dialogue(raw, me, aiPlaysMe)
            if (lines.isEmpty()) {
                return "Não achei mensagens aproveitáveis. O nome \"" + me + "\" bate com o que aparece no arquivo?"
            }
            val chars = lab.addCorpus(lines)
            if (chars == 0) return "O corpus está cheio (8 milhões de caracteres). Apague parte dele na aba Transformer."
            val nodes = tree.addRepresentatives(Importer.whatsappBlocks(raw, me), 20)
            return "Absorvido: " + lines.count { it.isNotEmpty() } + " falas (+" + chars + " caracteres) e " + nodes +
                " nós novos. A osmose treina sozinha com o app aberto."
        }
        var blocks = Importer.massBlocks(raw, "")
        if (blocks.isEmpty() && raw.trim().length >= 3) {
            blocks = listOf(raw.trim().replace(Regex("\\s+"), " "))
        }
        if (blocks.isEmpty()) return "Não encontrei texto aproveitável."
        val chars = lab.addCorpus(blocks)
        if (chars == 0) return "O corpus está cheio (8 milhões de caracteres). Apague parte dele na aba Transformer."
        val nodes = tree.addRepresentatives(blocks, 15)
        return "Absorvido: " + blocks.size + " trechos (+" + chars + " caracteres) e " + nodes +
            " nós novos. A osmose treina sozinha com o app aberto."
    }

    // ------------------------------------------------------------------ conversa (resposta única)

    private fun clip(s: String, max: Int): String {
        val t = s.replace(Regex("\\s+"), " ").trim()
        if (t.length <= max) return t
        val cut = t.substring(0, max)
        val sp = cut.lastIndexOf(' ')
        return if (sp > max / 2) cut.substring(0, sp) else cut
    }

    private fun firstLine(s: String): String? = s.lineSequence().firstOrNull { it.isNotBlank() }

    /** Diálogo recente no mesmo formato do treino ("@u ...", "@a ..."), terminando com "@a " para o modelo continuar. */
    private fun buildPrompt(h: List<Pair<Boolean, String>>, cur: String): String {
        val lines = ArrayList<String>()
        for ((fromUser, tx) in h.takeLast(4)) {
            val one = firstLine(tx) ?: continue
            lines.add((if (fromUser) TU else TA) + clip(one, 70))
        }
        lines.add(TU + clip(cur, 100))
        var prompt = lines.joinToString("\n") + "\n" + TA
        while (prompt.length > 130 && lines.size > 1) {
            lines.removeAt(0)
            prompt = lines.joinToString("\n") + "\n" + TA
        }
        return prompt
    }

    private fun cleanReply(s: String): String {
        var r = s.substringBefore('\n').trim()
        r = r.substringBefore("@u").substringBefore("@a").trim()
        return if (r.length < 2) "" else r
    }

    /**
     * Uma resposta só. A árvore escolhe o galho do assunto (vetor de galho do transformer) e o transformer
     * continua o diálogo recente. Enquanto o modelo ainda está imaturo, a resposta vem da memória da árvore.
     * O que você escreve vira treino (osmose); afirmações viram nós.
     */
    fun chatTurn(text: String): String {
        val t = text.trim()
        if (t.isEmpty()) return ""
        val h = db.chatHistory(9).toMutableList()
        if (h.isNotEmpty() && h.last().first && h.last().second == t) h.removeAt(h.size - 1)
        val found = tree.ask(t, 3).filter { it.second >= 0.15f }
        val best = found.firstOrNull()
        val question = t.contains("?")

        // osmose: o que você escreve vira texto de treino (uma linha "@u", separada por linha vazia)
        lab.addCorpus(listOf("", TU + clip(t, 160)))

        var newId = -1L
        if (!question && (best == null || best.second < 0.85f)) {
            newId = tree.learn(t)
            if (newId > 0) lab.learnInBackground(newId)
        }

        val g = lab.gpt
        val mature = g.step >= 300 && g.lastLoss.isFinite() && g.lastLoss < 1.9f
        var reply = ""
        if (mature) {
            val nodeId = if (best != null && best.second >= Tree.MIN_SIM) best.first.id else -1L
            reply = cleanReply(lab.generateChat(nodeId, buildPrompt(h, t)))
        }

        val sb = StringBuilder()
        if (reply.isNotEmpty()) {
            sb.append(reply)
        } else if (question && best != null) {
            sb.append("Lembro disto: ").append(tree.tidy(best.first.content, 300))
        } else if (newId > 0) {
            sb.append("Anotado.")
        } else {
            sb.append("Hum… ainda não sei responder isso.")
        }
        if (!mature) {
            sb.append("\n(Ainda estou aprendendo a conversar: ").append(g.step)
                .append(" passos. Deixe a osmose ligada ou importe conversas.)")
        }
        if (detail) {
            sb.append("\n— galho: ")
            if (best != null) sb.append(tree.tidy(best.first.title, 40)).append(" (").append((best.second * 100).toInt()).append("%)")
            else sb.append("nenhum")
            sb.append(" | passos ").append(g.step).append(" | perda ")
                .append(if (g.lastLoss.isNaN()) "—" else "%.2f".format(g.lastLoss))
        }
        return sb.toString()
    }

    /** 👍: guarda este trecho do diálogo (3 cópias, para pesar mais no treino). history[index] é a resposta da IA. */
    fun like(history: List<Pair<Boolean, String>>, index: Int): String {
        if (index <= 0 || index >= history.size || history[index].first) return ""
        val lines = ArrayList<String>()
        for (k in maxOf(0, index - 3)..index) {
            val (fromUser, tx) = history[k]
            val one = firstLine(tx) ?: continue
            lines.add((if (fromUser) TU else TA) + clip(one, 100))
        }
        if (lines.isEmpty()) return ""
        val blocks = ArrayList<String>()
        repeat(3) {
            blocks.add("")
            blocks.addAll(lines)
        }
        lab.addCorpus(blocks)
        return "Guardado."
    }
}
