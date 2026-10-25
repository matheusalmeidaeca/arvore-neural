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
        lab.persPairs = loadPersonal()
        lab.onMeasure = { st, v, n, p -> recordMeasure(st, v, n, p) }
        lab.onReset = { clearHistory() }
        val oldHist = prefs.getString("hist", "") ?: ""
        if (oldHist.isNotEmpty()) refreshProgress(oldHist.split(';'))
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
                } else if (db.corpusChars() < 200) {
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
            " caracteres\n" + lab.healthInfo()
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
                    }
                    prefs.edit().putBoolean("seeded", true).apply()
                }
                // segunda semente (v10): mais assunto para a osmose; entra uma vez só, mesmo em quem já tinha a primeira
                if (!prefs.getBoolean("seeded2", false)) {
                    val lines = ctx.assets.open("semente2.txt").bufferedReader().use { it.readLines() }
                    lab.addCorpus(lines)
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

    // ------------------------------------------------------------------ correções e histórico de progresso

    private fun loadPersonal(): List<Pair<String, String>> {
        val raw = prefs.getString("pers_eval", "") ?: ""
        val out = ArrayList<Pair<String, String>>()
        for (ln in raw.split('\n')) {
            val k = ln.indexOf('\t')
            if (k > 0 && k < ln.length - 1) out.add(Pair(ln.substring(0, k), ln.substring(k + 1)))
        }
        return out
    }

    /** Guarda a pergunta corrigida (até 60; a mesma pergunta corrigida de novo troca a resposta antiga). */
    @Synchronized
    private fun addPersonal(q: String, a: String) {
        val cq = q.replace('\t', ' ').replace('\n', ' ')
        val ca = a.replace('\t', ' ').replace('\n', ' ')
        val list = ArrayList<Pair<String, String>>()
        for (p in loadPersonal()) if (p.first != cq) list.add(p)
        list.add(Pair(cq, ca))
        while (list.size > 60) list.removeAt(0)
        val sb = StringBuilder()
        for (p in list) {
            if (sb.isNotEmpty()) sb.append('\n')
            sb.append(p.first).append('\t').append(p.second)
        }
        prefs.edit().putString("pers_eval", sb.toString()).apply()
        lab.persPairs = list
    }

    /** Uma linha por medição: hora, passos, erro em texto novo, nota fixa, nota das correções (até 200 linhas). */
    @Synchronized
    private fun recordMeasure(step: Int, v: Float, nota: Float, pers: Float) {
        val old = prefs.getString("hist", "") ?: ""
        val items = ArrayList<String>()
        if (old.isNotEmpty()) items.addAll(old.split(';'))
        if (items.isNotEmpty() && items[items.size - 1].split(',').getOrNull(1) == step.toString()) {
            items.removeAt(items.size - 1)
        }
        items.add(System.currentTimeMillis().toString() + "," + step + "," + v + "," + nota + "," + pers)
        while (items.size > 200) items.removeAt(0)
        prefs.edit().putString("hist", items.joinToString(";")).apply()
        refreshProgress(items)
    }

    private fun clearHistory() {
        prefs.edit().remove("hist").apply()
        lab.progressText = ""
    }

    /** Monta o resumo "como a IA evoluiu" que aparece em Saúde do modelo. */
    private fun refreshProgress(items: List<String>) {
        val notas = ArrayList<Float>()
        val erros = ArrayList<Float>()
        var firstStep = -1
        var lastStep = 0
        for (row in items) {
            val p = row.split(',')
            if (p.size < 5) continue
            val st = p[1].toIntOrNull() ?: continue
            val v = p[2].toFloatOrNull() ?: Float.NaN
            val n = p[3].toFloatOrNull() ?: Float.NaN
            if (firstStep < 0) firstStep = st
            lastStep = st
            if (!n.isNaN()) notas.add(n)
            if (!v.isNaN()) erros.add(v)
        }
        if (notas.size < 2) {
            lab.progressText = ""
            return
        }
        var best = notas[0]
        for (n in notas) if (n > best) best = n
        val sb = StringBuilder()
        sb.append("Evolução (").append(notas.size).append(" medições, passos ").append(firstStep)
            .append(" a ").append(lastStep).append("): nota ")
            .append("%.0f".format(notas[0])).append(" → ").append("%.0f".format(notas[notas.size - 1]))
            .append(" (melhor ").append("%.0f".format(best)).append(")")
        if (erros.size >= 2) {
            sb.append("; erro em texto novo ").append("%.2f".format(erros[0])).append(" → ")
                .append("%.2f".format(erros[erros.size - 1]))
        }
        val from = maxOf(0, notas.size - 8)
        val last = ArrayList<String>()
        for (k in from until notas.size) last.add("%.0f".format(notas[k]))
        sb.append("\nÚltimas notas: ").append(last.joinToString(", "))
        lab.progressText = sb.toString()
    }

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
            return "Absorvido: " + lines.count { it.isNotEmpty() } + " falas (+" + chars +
                " caracteres). A osmose treina os pesos sozinha com o app aberto."
        }
        if (Importer.isSpeakerChat(raw)) {
            // formato "Nome: fala": os nomes saem; só as falas entram no corpus e na árvore
            val lines = Importer.speakerDialogue(raw)
            if (lines.isEmpty()) return "Não achei conversas aproveitáveis nesse arquivo."
            val chars = lab.addCorpus(lines)
            if (chars == 0) return "O corpus está cheio (8 milhões de caracteres). Apague parte dele na aba Treino (Avançado)."
            return "Absorvido: " + lines.count { it.isNotEmpty() } + " falas (+" + chars +
                " caracteres, sem os nomes). A osmose treina os pesos sozinha com o app aberto."
        }
        var blocks = Importer.massBlocks(raw, "")
        if (blocks.isEmpty() && raw.trim().length >= 3) {
            blocks = listOf(raw.trim().replace(Regex("\\s+"), " "))
        }
        if (blocks.isEmpty()) return "Não encontrei texto aproveitável."
        val chars = lab.addCorpus(blocks)
        if (chars == 0) return "O corpus está cheio (8 milhões de caracteres). Apague parte dele na aba Treino (Avançado)."
        return "Absorvido: " + blocks.size + " trechos (+" + chars +
            " caracteres). A osmose treina os pesos sozinha com o app aberto."
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
     * Uma resposta só, escrita pelo transformer a partir do diálogo recente. v11: não existe árvore nem memória de
     * textos; tudo que a IA sabe está nos pesos. O que você escreve vira texto de treino (osmose).
     */
    fun chatTurn(text: String): String {
        val t = text.trim()
        if (t.isEmpty()) return ""
        val h = db.chatHistory(9).toMutableList()
        if (h.isNotEmpty() && h.last().first && h.last().second == t) h.removeAt(h.size - 1)

        // osmose: o que você escreve vira texto de treino (uma linha "@u", separada por linha vazia)
        lab.addCorpus(listOf("", TU + clip(t, 160)))

        val g = lab.gpt
        val reply = cleanReply(lab.generateChat(-1L, buildPrompt(h, t)))
        val sb = StringBuilder()
        sb.append(if (reply.isNotEmpty()) reply else "…")
        val mature = g.step >= 300 && g.lastLoss.isFinite() && g.lastLoss < 1.9f
        if (!mature) {
            sb.append("\n(Ainda estou aprendendo a conversar: ").append(g.step)
                .append(" passos. Isto é o que os meus pesos produzem hoje.)")
        }
        if (detail) {
            sb.append("\n— passos ").append(g.step).append(" | perda ")
                .append(if (g.lastLoss.isNaN()) "—" else "%.2f".format(g.lastLoss))
        }
        return sb.toString()
    }

    /**
     * ✏️: o usuário diz qual seria a resposta certa. history[index] é a resposta errada da IA e a pergunta é a
     * mensagem do usuário logo antes. v11: a correção treina os PESOS na hora (até o modelo dar alta probabilidade à
     * resposta certa) e também entra 5 vezes no corpus, para os treinos seguintes continuarem revendo. Devolve um
     * relatório com a probabilidade antes e depois e a resposta atual do modelo.
     */
    fun correct(history: List<Pair<Boolean, String>>, index: Int, right: String): String {
        if (index <= 0 || index >= history.size || history[index].first) return ""
        if (!history[index - 1].first) return ""
        val q = firstLine(history[index - 1].second) ?: return ""
        val a = firstLine(right) ?: return ""
        val qc = clip(q, 100)
        val ac = clip(a, 160)
        val blocks = ArrayList<String>()
        repeat(5) {
            blocks.add("")
            blocks.add(TU + qc)
            blocks.add(TA + ac)
        }
        lab.addCorpus(blocks)
        addPersonal(qc, ac)
        manual.set(true)
        pauseNow()
        hold(true)
        try {
            val r = lab.fitCorrection(qc, ac)
                ?: return "Essa correção é longa demais para o contexto do modelo. Use uma resposta mais curta."
            val before = "%.0f".format(r[0] * 100f)
            val after = "%.0f".format(r[1] * 100f)
            val prompt = buildPrompt(history.subList(0, index - 1), qc)
            val now = cleanReply(lab.generateChat(-1L, prompt, 90, 0.05f))
            val sb = StringBuilder()
            sb.append("Pesos ajustados em ").append(r[2].toInt()).append(" passos. Chance da resposta certa (por letra): ")
                .append(before).append("% → ").append(after).append("%.")
            sb.append("\nAgora ela responde: ").append(if (now.isEmpty()) "…" else now)
            if (r[1] < 0.5f) sb.append("\nAinda não fixou bem. Corrija de novo ou deixe a osmose treinar mais.")
            return sb.toString()
        } finally {
            hold(false)
            manual.set(false)
        }
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
