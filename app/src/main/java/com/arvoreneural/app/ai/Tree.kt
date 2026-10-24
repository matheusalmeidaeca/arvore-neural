package com.arvoreneural.app.ai

import com.arvoreneural.app.data.Db
import com.arvoreneural.app.data.Node

/** A "árvore neural": cada texto novo é encaixado sob o nó mais parecido e ajusta esse nó. */
class Tree(private val db: Db, private val emb: Embedder) {

    companion object {
        const val MIN_SIM = 0.30f
        const val MAX_CHUNKS = 400
    }

    /** Transformer ligado à árvore (definido na inicialização do app). */
    var lab: MiniLab? = null

    private val stamp = Regex("\\d{1,2}/\\d{1,2}/\\d{2,4}[ ,]+\\d{1,2}:\\d{2}(:\\d{2})?\\s*-?\\s*")

    fun learn(text: String): Long {
        val clean = text.trim()
        if (clean.isEmpty()) return -1L
        val e = emb.embed(clean)
        var best: Node? = null
        var bestSim = 0f
        for (n in db.allNodes()) {
            val s = cosine(e, n.embedding)
            if (s > bestSim) {
                bestSim = s
                best = n
            }
        }
        val parent = if (best != null && bestSim >= MIN_SIM) best.id else 0L
        val title = if (clean.length > 40) clean.take(40) + "…" else clean
        val id = db.insertNode(parent, title, clean, e)
        if (best != null && parent != 0L) {
            val p = best.embedding
            val m = FloatArray(p.size) { p[it] * 0.9f + e[it] * 0.1f }
            db.updateEmbedding(parent, normalize(m))
        }
        return id
    }

    private fun titleOf(t: String): String = if (t.length > 40) t.take(40) + "…" else t

    /** Cria um nó à mão sob o pai indicado (0 = raiz). */
    fun addNode(parentId: Long, text: String): Long {
        val t = text.trim()
        if (t.isEmpty()) return -1L
        return db.insertNode(parentId, titleOf(t), t, emb.embed(t))
    }

    /** Troca o texto de um nó (título, conteúdo e vetor de busca). */
    fun editNode(id: Long, text: String) {
        val t = text.trim()
        if (t.isEmpty()) return
        db.updateNode(id, titleOf(t), t, emb.embed(t))
    }

    /**
     * Importa uma lista indentada: cada "_" no começo da linha (ou tab, ou 2 espaços) é um nível.
     * Linhas sem "_" são raízes. Se o nó já existe (mesmo pai e mesmo texto), é reaproveitado,
     * então dá para importar a lista inteira de novo depois de acrescentar itens.
     * Com withPaths, cada nó novo também gera uma linha de treino com o caminho ("B > Br > Bra").
     */
    fun importOutline(raw: String, withPaths: Boolean = true): String {
        class Item(val depth: Int, val text: String)
        val items = ArrayList<Item>()
        for (ln in raw.replace("\r", "").split('\n')) {
            var sp = 0
            while (sp < ln.length && ln[sp] == ' ') sp++
            var i = sp
            var d = 0
            while (i < ln.length && (ln[i] == '_' || ln[i] == '\t')) {
                d++
                i++
            }
            if (d == 0) d = sp / 2
            val text = ln.substring(i).trim()
            if (text.isEmpty()) continue
            items.add(Item(d, text))
        }
        if (items.isEmpty()) return "Não encontrei nenhuma linha para importar."
        val cut = items.size > 3000
        val list = if (cut) items.subList(0, 3000) else items
        val existing = HashMap<String, Long>()
        for (n in db.allNodes()) existing[n.parentId.toString() + "|" + n.content] = n.id
        val ids = ArrayList<Long>()
        val texts = ArrayList<String>()
        val lines = ArrayList<String>()
        var created = 0
        var reused = 0
        db.batch {
            for (item in list) {
                val d = minOf(item.depth, ids.size)
                while (ids.size > d) {
                    ids.removeAt(ids.size - 1)
                    texts.removeAt(texts.size - 1)
                }
                val parent = if (d == 0) 0L else ids[d - 1]
                val key = parent.toString() + "|" + item.text
                val found = existing[key]
                val id: Long
                if (found != null) {
                    id = found
                    reused++
                } else {
                    id = db.insertNode(parent, titleOf(item.text), item.text, emb.embed(item.text))
                    existing[key] = id
                    created++
                    if (withPaths && d > 0) lines.add((texts + item.text).joinToString(" > "))
                }
                ids.add(id)
                texts.add(item.text)
            }
        }
        val l = lab
        val corpus = if (l != null && lines.isNotEmpty()) l.addCorpus(lines) else 0
        return "Lista importada: " + created + " nós novos, " + reused + " já existiam" +
            (if (corpus > 0) "; +" + corpus + " caracteres de treino com os caminhos" else "") +
            (if (cut) ". Só as 3000 primeiras linhas entraram." else ".")
    }

    /** onlyFrom só vale para conversas do WhatsApp: aprende apenas mensagens de quem tem esse nome. */
    fun learnFile(raw: String, onlyFrom: String = ""): Int {
        var count = 0
        val parts = if (Importer.isWhatsApp(raw)) Importer.whatsappBlocks(raw, onlyFrom) else chunk(raw)
        for (c in parts.take(MAX_CHUNKS)) {
            if (learn(c) > 0) count++
        }
        return count
    }

    /**
     * Treino em massa: o arquivo inteiro vai para o corpus do transformer e a árvore recebe só
     * alguns nós representativos (assuntos bem diferentes entre si).
     */
    fun importMass(raw: String, onlyFrom: String = "", maxNodes: Int = 40): String {
        val blocks = Importer.massBlocks(raw, onlyFrom)
        if (blocks.isEmpty()) return "Não encontrei texto aproveitável nesse arquivo."
        val l = lab
        val chars = if (l != null) l.addCorpus(blocks) else 0
        val reps = ArrayList<FloatArray>()
        var nodes = 0
        for (i in blocks.indices.shuffled(java.util.Random(7))) {
            if (nodes >= maxNodes) break
            val b = blocks[i]
            val e = emb.embed(b)
            var dup = false
            for (r in reps) {
                if (cosine(e, r) >= 0.5f) {
                    dup = true
                    break
                }
            }
            if (dup) continue
            reps.add(e)
            learn(b)
            nodes++
        }
        return "Importado: " + blocks.size + " trechos. Corpus +" + chars + " caracteres; árvore +" + nodes +
            " nós representativos. Agora toque em Treinar."
    }

    /** Cria até maxNodes nós representativos (assuntos bem diferentes entre si) a partir dos blocos. */
    fun addRepresentatives(blocks: List<String>, maxNodes: Int): Int {
        val reps = ArrayList<FloatArray>()
        var nodes = 0
        for (i in blocks.indices.shuffled(java.util.Random(7))) {
            if (nodes >= maxNodes) break
            val b = blocks[i]
            if (b.length < 20) continue
            val e = emb.embed(b)
            var dup = false
            for (r in reps) {
                if (cosine(e, r) >= 0.5f) {
                    dup = true
                    break
                }
            }
            if (dup) continue
            reps.add(e)
            learn(b)
            nodes++
        }
        return nodes
    }

    fun ask(q: String, k: Int = 3): List<Pair<Node, Float>> {
        val e = emb.embed(q)
        return db.allNodes()
            .map { it to cosine(e, it.embedding) }
            .filter { it.second > 0.05f }
            .sortedByDescending { it.second }
            .take(k)
    }

    /** Conversa: pergunta (com "?") consulta a árvore; frase afirmativa é aprendida. */
    fun chat(msg: String): String {
        val text = msg.trim()
        if (text.isEmpty()) return ""
        val found = ask(text, 3).filter { it.second >= 0.15f }
        if (text.contains("?")) {
            if (found.isEmpty()) return "Ainda não sei nada sobre isso. Me ensine com uma frase (sem interrogação)."
            val (best, s) = found[0]
            val sb = StringBuilder()
            val l = lab
            if (l != null) {
                val gen = l.generateFor(best.id)
                sb.append("🧠 [").append(l.gpt.step).append(" passos] ")
                sb.append(if (gen.isEmpty()) "(ainda sem resposta)" else gen)
                sb.append("\n🌳 ")
            }
            sb.append(tidy(best.content, 300))
            sb.append("\n(confiança ").append((s * 100).toInt()).append("%)")
            val rel = found.drop(1).filter { it.second >= 0.25f }
            if (rel.isNotEmpty()) {
                sb.append("\nVeja também: ")
                sb.append(rel.joinToString(" | ") { tidy(it.first.content, 80) })
            }
            return sb.toString()
        }
        val newId = learn(text)
        if (newId > 0) lab?.learnInBackground(newId)
        val extra = if (lab != null) "\n🧠 treinando o transformer com isso em segundo plano…" else ""
        if (found.isEmpty()) return "Aprendi! Assunto novo: abri um galho novo na árvore." + extra
        val (best, s) = found[0]
        return if (s >= MIN_SIM) "Aprendi! Encaixei sob: " + tidy(best.title, 60) + extra
        else "Aprendi! Abri um galho novo (o mais parecido era: " + tidy(best.title, 60) + ")." + extra
    }

    /** Remove carimbos de data/hora e corta em fim de frase ou de palavra. */
    fun tidy(s: String, max: Int): String {
        val t = s.replace(stamp, "").replace(Regex("\\s+"), " ").trim()
        if (t.length <= max) return t
        val cut = t.substring(0, max)
        val end = maxOf(cut.lastIndexOf(". "), cut.lastIndexOf("! "), cut.lastIndexOf("? "))
        if (end > max / 2) return cut.substring(0, end + 1)
        return cut.substring(0, cut.lastIndexOf(' ').coerceAtLeast(1)) + "…"
    }

    private fun chunk(raw: String, max: Int = 500): List<String> {
        val out = mutableListOf<String>()
        for (p in raw.split(Regex("\\n\\s*\\n"))) {
            var s = p.trim().replace(Regex("\\s+"), " ")
            while (s.length > max) {
                var cut = s.lastIndexOf(". ", max)
                cut = if (cut < 100) max else cut + 1
                out.add(s.substring(0, cut).trim())
                s = s.substring(cut).trim()
            }
            if (s.length > 20) out.add(s)
        }
        return out
    }
}
