package com.arvoreneural.app.ai

/** Aprendizado online simples: reajusta a reta de tendência a cada novo valor (últimos 12). */
object Forecaster {
    fun predict(values: List<Double>): Double? {
        if (values.size < 2) return null
        val ys = values.takeLast(12)
        val n = ys.size
        val mx = (n - 1) / 2.0
        val my = ys.average()
        var num = 0.0
        var den = 0.0
        for (i in 0 until n) {
            num += (i - mx) * (ys[i] - my)
            den += (i - mx) * (i - mx)
        }
        val slope = if (den == 0.0) 0.0 else num / den
        return my + slope * (n - mx)
    }

    /**
     * Segunda opinião (experimental): mistura de especialistas que cresce sozinha, treinada nas janelas
     * da própria série (6 valores -> o próximo, tudo relativo à média da janela). Só para séries positivas
     * com pelo menos 14 valores. Devolve (previsão, número de especialistas).
     * Em testes com séries sintéticas ela NÃO bate a tendência linear em séries simples; serve de comparação.
     */
    fun predictMoe(values: List<Double>): Pair<Double, Int>? {
        val w = 6
        if (values.size < w + 8) return null
        val xs = ArrayList<FloatArray>()
        val ts = ArrayList<FloatArray>()
        for (i in 0 until values.size - w) {
            val sc = values.subList(i, i + w).average()
            if (sc <= 0.0) return null
            xs.add(FloatArray(w) { (values[i + it] / sc).toFloat() })
            ts.add(floatArrayOf((values[i + w] / sc).toFloat()))
        }
        val ai = DynamicSuperAI(w, 1, 16, 0.002f, 300, 0.05f, 5L)
        val order = xs.indices.toMutableList()
        val rnd = java.util.Random(5)
        repeat(60) {
            order.shuffle(rnd)
            for (k in order) ai.train(xs[k], ts[k])
        }
        val last = values.takeLast(w)
        val sc = last.average()
        if (sc <= 0.0) return null
        val x = FloatArray(w) { (last[it] / sc).toFloat() }
        return Pair(ai.process(x)[0] * sc, ai.experts.size)
    }
}
