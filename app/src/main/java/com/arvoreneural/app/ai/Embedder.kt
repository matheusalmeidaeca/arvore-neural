package com.arvoreneural.app.ai

import java.text.Normalizer
import kotlin.math.sqrt

/** Transforma texto em vetor. Troque por um transformer ONNX na Etapa 6 sem mexer no resto. */
interface Embedder {
    fun embed(text: String): FloatArray
}

fun normalize(v: FloatArray): FloatArray {
    var s = 0f
    for (x in v) s += x * x
    val n = sqrt(s)
    if (n > 0f) for (i in v.indices) v[i] = v[i] / n
    return v
}

/** Vetores já normalizados: produto escalar = similaridade de cosseno. */
fun cosine(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return 0f
    var s = 0f
    for (i in a.indices) s += a[i] * b[i]
    return s
}

class HashEmbedder(private val dim: Int = 256) : Embedder {
    private val stop = setOf(
        "que", "uma", "com", "para", "por", "dos", "das", "nos", "nas", "mais", "como",
        "mas", "foi", "ser", "tem", "sao", "isso", "esse", "essa", "este", "esta", "seu", "sua"
    )

    override fun embed(text: String): FloatArray {
        val v = FloatArray(dim)
        val t = tokenize(text)
        for (i in t.indices) {
            add(v, t[i], 1f)
            if (i + 1 < t.size) add(v, t[i] + "_" + t[i + 1], 0.5f)
        }
        return normalize(v)
    }

    private fun tokenize(s: String): List<String> =
        Normalizer.normalize(s.lowercase(), Normalizer.Form.NFD)
            .replace(Regex("\\p{M}+"), "")
            .split(Regex("[^a-z0-9]+"))
            .filter { it.length > 2 && it !in stop }

    private fun add(v: FloatArray, tok: String, w: Float) {
        val idx = (tok.hashCode() and 0x7fffffff) % dim
        val sign = if (((tok + "#").hashCode() and 1) == 0) 1f else -1f
        v[idx] += sign * w
    }
}
