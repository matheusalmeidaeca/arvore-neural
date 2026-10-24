package com.arvoreneural.app.ai

import java.util.Random
import kotlin.math.sqrt

/** Especialista: sub-rede pequena (uma camada densa + ReLU) com pesos próprios. */
class Expert(val id: Int, val inputSize: Int, val outputSize: Int, rnd: Random = Random()) {
    val weights = FloatArray(inputSize * outputSize) { rnd.nextFloat() * 0.1f }
    val bias = FloatArray(outputSize)

    fun forward(input: FloatArray): FloatArray {
        val output = FloatArray(outputSize)
        for (i in 0 until outputSize) {
            var sum = bias[i]
            for (j in 0 until inputSize) sum += input[j] * weights[j * outputSize + i]
            output[i] = if (sum > 0f) sum else 0f // ReLU
        }
        return output
    }

    /** Um passo de gradiente (erro quadrático médio). Devolve a perda desta amostra. */
    fun learn(input: FloatArray, target: FloatArray, lr: Float): Float {
        val y = forward(input)
        var loss = 0f
        for (i in 0 until outputSize) {
            val e = y[i] - target[i]
            loss += e * e
            if (y[i] > 0f) { // derivada do ReLU
                val g = 2f * e / outputSize
                for (j in 0 until inputSize) weights[j * outputSize + i] -= lr * g * input[j]
                bias[i] -= lr * g
            }
        }
        return loss / outputSize
    }
}

/**
 * Roteador (o gerente): uma linha de pesos por especialista; escolhe o de maior afinidade.
 * Guardado como lista de linhas, então crescer não embaralha os pesos antigos.
 */
class Router(val inputSize: Int, startExperts: Int = 2, private val rnd: Random = Random()) {
    private val rows = ArrayList<FloatArray>()
    val numExperts: Int get() = rows.size

    init {
        repeat(startExperts) { rows.add(unit(FloatArray(inputSize) { rnd.nextFloat() - 0.5f })) }
    }

    private fun unit(v: FloatArray): FloatArray {
        var s = 0f
        for (x in v) s += x * x
        val n = sqrt(s)
        return if (n > 1e-8f) FloatArray(v.size) { v[it] / n } else v.copyOf()
    }

    fun route(input: FloatArray): Int {
        val u = unit(input)
        var best = 0
        var max = -Float.MAX_VALUE
        for (e in rows.indices) {
            var score = 0f
            val r = rows[e]
            for (j in 0 until inputSize) score += u[j] * r[j]
            if (score > max) { max = score; best = e }
        }
        return best
    }

    /** Puxa a linha do especialista que acertou em direção à entrada (aprendizado competitivo). */
    fun reward(e: Int, input: FloatArray, lr: Float) {
        val u = unit(input)
        val r = rows[e]
        rows[e] = unit(FloatArray(inputSize) { r[it] + lr * (u[it] - r[it]) })
    }

    /** Novo especialista nasce: ganha uma linha própria (protótipo das entradas difíceis); as antigas ficam intactas. */
    fun expandRouter(prototype: FloatArray) {
        rows.add(unit(prototype))
    }
}

/**
 * Mistura de especialistas que cresce sozinha: o roteador escolhe UM especialista por entrada
 * e, se o erro (média móvel) continua alto, nasce um especialista novo.
 */
class DynamicSuperAI(
    val inputSize: Int,
    val outputSize: Int,
    private val maxExperts: Int = 16,
    private val errorThreshold: Float = 0.05f,   // tolerância (média móvel da perda)
    private val cooldown: Int = 300,             // passos mínimos entre dois nascimentos
    private val lr: Float = 0.1f,
    seed: Long = 42L
) {
    private val rnd = Random(seed)
    val experts = ArrayList<Expert>()
    val router = Router(inputSize, 2, rnd)

    var avgLoss = 1f
        private set
    var steps = 0
        private set
    private var sinceGrow = 0
    private val hard = ArrayList<FloatArray>() // entradas recentes com erro alto

    var onGrow: ((id: Int, avgLoss: Float) -> Unit)? = null

    init {
        experts.add(Expert(0, inputSize, outputSize, rnd))
        experts.add(Expert(1, inputSize, outputSize, rnd))
    }

    @Synchronized
    fun process(input: FloatArray): FloatArray = experts[router.route(input)].forward(input)

    /** Um passo de treino: roteia, ajusta o especialista escolhido, depois avalia se precisa crescer. Devolve a perda. */
    @Synchronized
    fun train(input: FloatArray, target: FloatArray): Float {
        val e = router.route(input)
        val loss = experts[e].learn(input, target, lr)
        if (loss < avgLoss) router.reward(e, input, 0.05f)
        avgLoss = 0.98f * avgLoss + 0.02f * loss
        steps++
        sinceGrow++
        if (loss > errorThreshold * 2f) {
            hard.add(input.copyOf())
            if (hard.size > 20) hard.removeAt(0)
        }
        evaluateAndGrow(avgLoss)
        return loss
    }

    /** Algoritmo de expansão autônoma. */
    fun evaluateAndGrow(currentLoss: Float) {
        if (currentLoss <= errorThreshold) return
        if (experts.size >= maxExperts || sinceGrow < cooldown || hard.size < 10) return
        val proto = FloatArray(inputSize)
        for (h in hard) for (j in 0 until inputSize) proto[j] += h[j] / hard.size
        val newId = experts.size
        experts.add(Expert(newId, inputSize, outputSize, rnd))
        router.expandRouter(proto)
        hard.clear()
        sinceGrow = 0
        onGrow?.invoke(newId, currentLoss)
    }
}
