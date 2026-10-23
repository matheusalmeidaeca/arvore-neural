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
}
