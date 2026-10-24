package com.arvoreneural.app.ai

import com.arvoreneural.app.data.Db
import com.arvoreneural.app.data.Node

/** A "árvore neural": cada texto novo é encaixado sob o nó mais parecido e ajusta esse nó. */
class Tree(private val db: Db, private val emb: Embedder) {

    companion object {
        const val MIN_SIM = 0.30f
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

    /** Remove carimbos de data/hora e corta em fim de frase ou de palavra. */
    fun tidy(s: String, max: Int): String {
        val t = s.replace(stamp, "").replace(Regex("\\s+"), " ").trim()
        if (t.length <= max) return t
        val cut = t.substring(0, max)
        val end = maxOf(cut.lastIndexOf(". "), cut.lastIndexOf("! "), cut.lastIndexOf("? "))
        if (end > max / 2) return cut.substring(0, end + 1)
        return cut.substring(0, cut.lastIndexOf(' ').coerceAtLeast(1)) + "…"
    }
}
