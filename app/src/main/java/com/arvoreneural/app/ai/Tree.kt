package com.arvoreneural.app.ai

import com.arvoreneural.app.data.Db
import com.arvoreneural.app.data.Node

/** A "árvore neural": cada texto novo é encaixado sob o nó mais parecido e ajusta esse nó. */
class Tree(private val db: Db, private val emb: Embedder) {

    companion object {
        const val MIN_SIM = 0.30f
        const val MAX_CHUNKS = 300
    }

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

    fun learnFile(raw: String): Int {
        var count = 0
        for (c in chunk(raw).take(MAX_CHUNKS)) {
            if (learn(c) > 0) count++
        }
        return count
    }

    fun ask(q: String, k: Int = 3): List<Pair<Node, Float>> {
        val e = emb.embed(q)
        return db.allNodes()
            .map { it to cosine(e, it.embedding) }
            .filter { it.second > 0.05f }
            .sortedByDescending { it.second }
            .take(k)
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
