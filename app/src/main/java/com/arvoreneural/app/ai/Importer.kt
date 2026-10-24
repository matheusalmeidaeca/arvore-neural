package com.arvoreneural.app.ai

/** Lê conversas exportadas do WhatsApp (Android e iPhone) e devolve só o texto das mensagens. */
object Importer {
    private val waAndroid = Regex("^(\\d{1,2}/\\d{1,2}/\\d{2,4})[ ,]+(\\d{1,2}:\\d{2})(?::\\d{2})?(?:\\s?[APap][Mm])?\\s*-\\s*(.*)$")
    private val waIos = Regex("^\\[(\\d{1,2}/\\d{1,2}/\\d{2,4})[ ,]+(\\d{1,2}:\\d{2})(?::\\d{2})?(?:\\s?[APap][Mm])?\\]\\s*(.*)$")

    private val noise = listOf(
        "mídia oculta", "midia oculta", "mídia omitida", "media omitted", "apagou esta mensagem",
        "mensagem apagada", "arquivo anexado", "<anexado", "imagem ocultada", "áudio omitido"
    )

    private fun clean(l: String): String =
        l.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "").replace("\u202f", " ")

    private fun header(line: String): String? {
        val m = waAndroid.matchEntire(line) ?: waIos.matchEntire(line) ?: return null
        return m.groupValues[3]
    }

    fun isWhatsApp(raw: String): Boolean {
        var hits = 0
        var seen = 0
        for (l in raw.lineSequence()) {
            val line = clean(l)
            if (line.isBlank()) continue
            seen++
            if (header(line) != null) hits++
            if (seen >= 30) break
        }
        return hits >= 3
    }

    private fun isNoise(t: String): Boolean {
        val low = t.lowercase()
        return noise.any { low.contains(it) }
    }

    /** Para o treino em massa: cada bloco vira uma linha só (conversa ou parágrafo), sem quebras. */
    fun massBlocks(raw: String, onlyFrom: String): List<String> {
        if (isWhatsApp(raw)) return whatsappBlocks(raw, onlyFrom)
        val out = ArrayList<String>()
        for (p in raw.split(Regex("\\n\\s*\\n"))) {
            val s = p.trim().replace(Regex("\\s+"), " ")
            if (s.length >= 20) out.add(s)
        }
        return out
    }

    private fun clipTurn(s: String, max: Int = 160): String {
        val t = s.replace(Regex("\\s+"), " ").trim()
        if (t.length <= max) return t
        val cut = t.substring(0, max)
        val sp = cut.lastIndexOf(' ')
        return if (sp > max / 2) cut.substring(0, sp) else cut
    }

    /**
     * Diálogo para o transformer: cada fala vira uma linha "@u texto" (o usuário) ou "@a texto" (a IA).
     * me = parte do nome do usuário no arquivo. aiPlaysMe = false: a IA imita as outras pessoas;
     * true: a IA imita o usuário. Mensagens seguidas da mesma pessoa viram uma fala (até ~160 caracteres).
     * Uma linha vazia separa os dias (conversas diferentes).
     */
    fun dialogue(raw: String, me: String, aiPlaysMe: Boolean): List<String> {
        class M(val sender: String, val text: StringBuilder, val date: String)
        val msgs = ArrayList<M>()
        for (l in raw.lineSequence()) {
            val line = clean(l)
            val m = waAndroid.matchEntire(line) ?: waIos.matchEntire(line)
            if (m != null) {
                val rest = m.groupValues[3]
                val date = m.groupValues[1]
                val i = rest.indexOf(": ")
                if (i in 1..40) msgs.add(M(rest.substring(0, i).trim(), StringBuilder(rest.substring(i + 2)), date))
                else msgs.add(M("", StringBuilder(), date))
            } else if (msgs.isNotEmpty()) {
                msgs.last().text.append(' ').append(line.trim())
            }
        }
        val key = me.trim().lowercase()
        val out = ArrayList<String>()
        val cur = StringBuilder()
        var curTag = ""
        var lastDate = ""
        fun flush() {
            if (cur.isNotEmpty()) {
                out.add(curTag + clipTurn(cur.toString()))
                cur.clear()
            }
        }
        for (m in msgs) {
            if (m.sender.isEmpty()) continue
            val txt = m.text.toString().trim()
            if (txt.isEmpty() || isNoise(txt) || txt.contains("http")) continue
            val isMe = m.sender.lowercase().contains(key)
            val isAi = if (aiPlaysMe) isMe else !isMe
            val tag = if (isAi) "@a " else "@u "
            if (m.date != lastDate) {
                flush()
                if (lastDate.isNotEmpty()) out.add("")
                lastDate = m.date
            }
            if (tag != curTag && cur.isNotEmpty()) flush()
            curTag = tag
            if (cur.isNotEmpty()) cur.append(' ')
            cur.append(txt)
            if (cur.length >= 160) flush()
        }
        flush()
        return out
    }

    /** onlyFrom: parte do nome de quem falou (vazio = todos). Junta mensagens seguidas da mesma pessoa. */
    fun whatsappBlocks(raw: String, onlyFrom: String): List<String> {
        val msgs = mutableListOf<Pair<String, StringBuilder>>()
        for (l in raw.lineSequence()) {
            val line = clean(l)
            val rest = header(line)
            if (rest != null) {
                val i = rest.indexOf(": ")
                if (i in 1..40) msgs.add(rest.substring(0, i).trim() to StringBuilder(rest.substring(i + 2)))
                else msgs.add("" to StringBuilder())
            } else if (msgs.isNotEmpty()) {
                msgs.last().second.append(' ').append(line.trim())
            }
        }
        val filter = onlyFrom.trim().lowercase()
        val blocks = mutableListOf<String>()
        val sb = StringBuilder()
        var last = ""
        for ((sender, t) in msgs) {
            if (sender.isEmpty()) continue
            if (filter.isNotEmpty() && !sender.lowercase().contains(filter)) continue
            val txt = t.toString().trim()
            if (txt.isEmpty() || isNoise(txt)) continue
            if (sender != last && sb.isNotEmpty()) {
                blocks.add(sb.toString())
                sb.clear()
            }
            last = sender
            if (sb.isNotEmpty()) sb.append(' ')
            sb.append(txt)
            if (sb.length >= 450) {
                blocks.add(sb.toString())
                sb.clear()
            }
        }
        if (sb.isNotEmpty()) blocks.add(sb.toString())
        return blocks.filter { it.length >= 20 }
    }

    // ------------------------------------------------------------------ conversas "Nome: fala"

    private val speaker = Regex("^\\s*(\\p{L}[\\p{L}\\p{N} _.'-]{0,24}):\\s+(.+)$")

    /** Arquivo de conversas no formato "Nome: fala", com linha vazia entre uma conversa e outra. */
    fun isSpeakerChat(raw: String): Boolean {
        var seen = 0
        var hits = 0
        val names = HashSet<String>()
        for (l in raw.lineSequence()) {
            val line = clean(l)
            if (line.isBlank()) continue
            seen++
            val m = speaker.matchEntire(line)
            if (m != null) {
                hits++
                names.add(m.groupValues[1].trim().lowercase())
            }
            if (seen >= 40) break
        }
        return seen >= 5 && hits * 10 >= seen * 7 && names.size >= 2
    }

    /** Conversas separadas por linha vazia; cada uma vira uma lista de falas (nome, texto), juntando falas seguidas da mesma pessoa. */
    private fun speakerConversations(raw: String): List<List<Pair<String, String>>> {
        val all = ArrayList<List<Pair<String, String>>>()
        var cur = ArrayList<Pair<String, String>>()
        fun close() {
            if (cur.isNotEmpty()) all.add(cur)
            cur = ArrayList()
        }
        for (l in raw.lineSequence()) {
            val line = clean(l)
            if (line.isBlank()) {
                close()
                continue
            }
            val m = speaker.matchEntire(line)
            if (m != null) {
                val name = m.groupValues[1].trim()
                val txt = m.groupValues[2].trim()
                if (cur.isNotEmpty() && cur.last().first == name) {
                    cur[cur.size - 1] = name to (cur.last().second + " " + txt)
                } else {
                    cur.add(name to txt)
                }
            } else if (cur.isNotEmpty()) {
                cur[cur.size - 1] = cur.last().first to (cur.last().second + " " + line.trim())
            }
        }
        close()
        return all
    }

    /** Diálogo sem nomes: em cada conversa, quem fala primeiro é "@u" e os outros são "@a". */
    fun speakerDialogue(raw: String): List<String> {
        val out = ArrayList<String>()
        for (conv in speakerConversations(raw)) {
            val turns = conv.filter { it.second.isNotBlank() && !isNoise(it.second) && !it.second.contains("http") }
            if (turns.size < 2) continue
            val first = turns[0].first
            out.add("")
            for ((name, txt) in turns) out.add((if (name == first) "@u " else "@a ") + clipTurn(txt))
        }
        return out
    }

    /** Textos sem nomes (uma conversa por bloco, até ~450 caracteres) para os nós da árvore. */
    fun speakerBlocks(raw: String): List<String> {
        val out = ArrayList<String>()
        for (conv in speakerConversations(raw)) {
            var t = conv.joinToString(" ") { it.second }.replace(Regex("\\s+"), " ").trim()
            while (t.length > 450) {
                var cut = t.lastIndexOf(' ', 450)
                if (cut < 200) cut = 450
                out.add(t.substring(0, cut).trim())
                t = t.substring(cut).trim()
            }
            if (t.length >= 20) out.add(t)
        }
        return out
    }
}
