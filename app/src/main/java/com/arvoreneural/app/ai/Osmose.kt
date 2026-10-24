package com.arvoreneural.app.ai

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import com.arvoreneural.app.data.Db
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

private const val TU = "@u "
private const val TA = "@a "
/** A partir desta semelhança, uma pergunta é respondida direto pela árvore. */
private const val DIRECT_SIM = 0.5f
private const val DEFAULT_SYS = "Você é a Árvore: uma IA pequena, curiosa e gentil."

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

    /** Ligado pelo botão "Treinar" da aba Treino: a osmose espera. */
    val manual = AtomicBoolean(false)

    @Volatile private var foreground = false
    @Volatile private var loopWants = false
    private val extUsers = AtomicInteger(0)
    private val wl: PowerManager.WakeLock? = try {
        (ctx.getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "arvore:osmose")
            .also { it.setReferenceCounted(false) }
    } catch (e: Throwable) {
        null
    }

    /** Impede o CPU de dormir com a tela apagada (treino, resposta ou geração em andamento). */
    private fun cpu(on: Boolean) {
        try {
            val w = wl ?: return
            if (on) {
                if (!w.isHeld) w.acquire(10 * 60 * 1000L)
            } else if (w.isHeld) {
                w.release()
            }
        } catch (e: Throwable) {
        }
    }

    private fun syncCpu() {
        cpu(loopWants || extUsers.get() > 0)
    }

    /** Usado por quem trabalha fora do laço (resposta da conversa, treino manual). Chame hold(true) e depois hold(false). */
    fun hold(on: Boolean) {
        if (on) extUsers.incrementAndGet() else if (extUsers.get() > 0) extUsers.decrementAndGet()
        syncCpu()
    }

    /** Usado ao restaurar um backup: a osmose para de treinar e de gravar. */
    @Volatile var frozen = false
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

    /** Prompt de sistema (até 60 caracteres): vai antes de cada conversa, no treino e na resposta. Vazio desliga. */
    @Volatile var systemPrompt: String = prefs.getString("sys_prompt", DEFAULT_SYS) ?: DEFAULT_SYS
        set(v) {
            val t = v.trim().replace(Regex("\\s+"), " ").take(60)
            field = t
            prefs.edit().putString("sys_prompt", t).apply()
            lab.sysPrompt = t
            lab.invalidateCorpus()
        }

    @Volatile var detail: Boolean = prefs.getBoolean("detail", false)
        set(v) {
            field = v
            prefs.edit().putBoolean("detail", v).apply()
        }

    // ------------------------------------------------------------------ laço de treino

    fun start() {
        if (started.getAndSet(true)) return
        lab.sysPrompt = systemPrompt
        try {
            lab.evalPairs = ctx.assets.open("avaliacao.txt").bufferedReader().use { r ->
                r.readLines().mapNotNull { ln ->
                    val k = ln.indexOf('|')
                    if (k <= 0) null else ln.substring(0, k).trim() to ln.substring(k + 1).trim()
                }
            }
        } catch (e: Throwable) {
        }
        val th = Thread { loop() }
        th.isDaemon = true
        th.start()
    }

    fun setForeground(v: Boolean) {
        foreground = v
        if (!v) {
            // o treino continua em segundo plano (serviço); só garante que os pesos estejam gravados
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
                loopWants = false
                if (frozen) {
                    why = "restaurando backup"
                } else if (!enabled) {
                    why = "desligada"
                } else if (manual.get()) {
                    loopWants = true
                    why = "treino manual em andamento"
                } else if (!canTrain()) {
                    // canTrain já preencheu o motivo
                } else if (db.corpusChars() < 200 && db.nodeCount() == 0) {
                    why = "nada para aprender ainda (converse, ensine ou importe uma conversa)"
                } else if (lab.overfitting()) {
                    why = "parou para não decorar (o erro em texto novo subiu). Traga mais texto."
                } else {
                    why = ""
                    loopWants = true
                    syncCpu()
                    stopFlag.set(false)
                    working = true
                    val t0 = System.currentTimeMillis()
                    val n = lab.train(10, 4, -1L, stopFlag, false) { _, _ -> }
                    working = false
                    dirtySteps += n
                    if (dirtySteps >= 100) {
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
            syncCpu()
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
            " caracteres | nós " + db.nodeCount() + "\n" + lab.healthInfo()
    }

    // ------------------------------------------------------------------ semente

    /**
     * Primeira abertura: o corpus ainda está vazio, então a osmose recebe o texto-semente (assets/semente.txt:
     * pequenos diálogos @u/@a em português) para já ter o que treinar. Só acontece uma vez.
     */
    fun seedIfNeeded() {
        Thread {
            try {
                if (!prefs.getBoolean("seeded", false)) {
                    if (db.corpusChars() < 200) {
                        val lines = ctx.assets.open("semente.txt").bufferedReader().use { it.readLines() }
                        lab.addCorpus(lines)
                        tree.addRepresentatives(
                            lines.filter { it.startsWith(TA) && it.length > 40 }.map { it.removePrefix(TA) }, 6
                        )
                    }
                    prefs.edit().putBoolean("seeded", true).apply()
                }
                // segunda semente (v10): mais assunto para a osmose; entra uma vez só, mesmo em quem já tinha a primeira
                if (!prefs.getBoolean("seeded2", false)) {
                    val lines = ctx.assets.open("semente2.txt").bufferedReader().use { it.readLines() }
                    lab.addCorpus(lines)
                    tree.addRepresentatives(
                        lines.filter { it.startsWith(TA) && it.length > 40 }.map { it.removePrefix(TA) }, 6
                    )
                    prefs.edit().putBoolean("seeded2", true).apply()
                }
                if (prefs.getLong("hits_since", 0L) == 0L) {
                    prefs.edit().putLong("hits_since", System.currentTimeMillis()).apply()
                }
            } catch (e: Throwable) {
            }
        }.start()
    }

    /** Desde quando a árvore conta quantas vezes cada nó foi usado (para saber o que nunca é usado). */
    fun hitsSince(): Long = prefs.getLong("hits_since", 0L)

    // ------------------------------------------------------------------ absorver textos

    /** Texto ou conversa do WhatsApp (compartilhado ou escolhido num arquivo). Devolve um resumo para mostrar. */
    fun absorb(raw: String): String {
        if (raw.isBlank()) return "Nada para absorver."
        if (Importer.isWhatsApp(raw)) {
            val me = myName.trim()
            if (me.isEmpty()) {
                return "É uma conversa do WhatsApp. Digite seu nome na aba Treino (seção Osmose) e compartilhe de novo."
            }
            val lines = Importer.dialogue(raw, me, aiPlaysMe)
            if (lines.isEmpty()) {
                return "Não achei mensagens aproveitáveis. O nome \"" + me + "\" bate com o que aparece no arquivo?"
            }
            val chars = lab.addCorpus(lines)
            if (chars == 0) return "O corpus está cheio (8 milhões de caracteres). Apague parte dele na aba Treino (Avançado)."
            val nodes = tree.addRepresentatives(Importer.whatsappBlocks(raw, me), 20)
            return "Absorvido: " + lines.count { it.isNotEmpty() } + " falas (+" + chars + " caracteres) e " + nodes +
                " nós novos. A osmose treina sozinha com o app aberto."
        }
        if (Importer.isSpeakerChat(raw)) {
            // formato "Nome: fala": os nomes saem; só as falas entram no corpus e na árvore
            val lines = Importer.speakerDialogue(raw)
            if (lines.isEmpty()) return "Não achei conversas aproveitáveis nesse arquivo."
            val chars = lab.addCorpus(lines)
            if (chars == 0) return "O corpus está cheio (8 milhões de caracteres). Apague parte dele na aba Treino (Avançado)."
            val nodes = tree.addRepresentatives(Importer.speakerBlocks(raw), 20)
            return "Absorvido: " + lines.count { it.isNotEmpty() } + " falas (+" + chars + " caracteres, sem os nomes) e " + nodes +
                " nós novos. A osmose treina sozinha com o app aberto."
        }
        var blocks = Importer.massBlocks(raw, "")
        if (blocks.isEmpty() && raw.trim().length >= 3) {
            blocks = listOf(raw.trim().replace(Regex("\\s+"), " "))
        }
        if (blocks.isEmpty()) return "Não encontrei texto aproveitável."
        val chars = lab.addCorpus(blocks)
        if (chars == 0) return "O corpus está cheio (8 milhões de caracteres). Apague parte dele na aba Treino (Avançado)."
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
        val sys = lab.sysLine()
        val lines = ArrayList<String>()
        for ((fromUser, tx) in h.takeLast(4)) {
            val one = firstLine(tx) ?: continue
            lines.add((if (fromUser) TU else TA) + clip(one, 70))
        }
        lines.add(TU + clip(cur, if (sys.isEmpty()) 100 else 80))
        val limit = if (sys.isEmpty()) 130 else 150
        var prompt = sys + lines.joinToString("\n") + "\n" + TA
        while (prompt.length > limit && lines.size > 1) {
            lines.removeAt(0)
            prompt = sys + lines.joinToString("\n") + "\n" + TA
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

        // aprendizado instantâneo: pergunta parecida com algo que a árvore já guarda é respondida com isso, sem esperar treino
        val direct = question && best != null && best.second >= DIRECT_SIM
        if (best != null && best.second >= Tree.MIN_SIM) {
            try {
                db.bumpHit(best.first.id)
            } catch (e: Throwable) {
            }
        }

        val g = lab.gpt
        val mature = g.step >= 300 && g.lastLoss.isFinite() && g.lastLoss < 1.9f
        var reply = ""
        if (mature && !direct) {
            val nodeId = if (best != null && best.second >= Tree.MIN_SIM) best.first.id else -1L
            reply = cleanReply(lab.generateChat(nodeId, buildPrompt(h, t)))
        }

        val sb = StringBuilder()
        if (direct && best != null) {
            sb.append("Lembro disto: ").append(tree.tidy(best.first.content, 300))
        } else if (reply.isNotEmpty()) {
            sb.append(reply)
        } else if (question && best != null) {
            sb.append("Lembro disto: ").append(tree.tidy(best.first.content, 300))
        } else if (newId > 0) {
            sb.append("Anotado.")
        } else {
            sb.append("Hum… ainda não sei responder isso.")
        }
        if (!mature && !direct) {
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
