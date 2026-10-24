package com.arvoreneural.app.ai

import com.arvoreneural.app.data.Db
import com.arvoreneural.app.data.Node
import java.util.concurrent.atomic.AtomicBoolean
import java.nio.ByteBuffer
import java.util.Arrays
import java.util.Random
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tanh

/*
 * Mini-transformer próprio (port do mini.py): GPT de caracteres, pré-LayerNorm, atenção causal
 * multi-cabeça, MLP com GELU, AdamW. Treino com backpropagation escrito à mão, tudo em Kotlin.
 * Contexto de 192 caracteres (v7; era 64), C=128, H=4, L=4. Os pesos aprendidos ficam no banco de dados.
 */

class MiniGptConfig(
    val v: Int, val t: Int = 192, val c: Int = 128, val h: Int = 4, val l: Int = 4,
    /** quantidade de "vetores de galho": cada nó da árvore usa um deles */
    val s: Int = 1024,
    /** threads do treino: cada uma tem uma cópia do gradiente, então modelos grandes usam menos */
    val maxThreads: Int = 4,
    /** taxa de aprendizado: modelos maiores pedem passos menores */
    val lr: Float = 1e-3f,
    /** especialistas por camada que o layout já reserva ao criar (1 = só o feed-forward comum; os outros nascem ao crescer) */
    val ex: Int = 1
)

/** Tamanhos escolhíveis do modelo (v7.3). Cabeças de 32 dimensões em todos. */
object Sizes {
    val names = listOf("Mini", "Médio", "Grande")
    val info = listOf(
        "≈1M parâmetros, ~70 MB de memória (+~15 MB por especialista novo em cada camada)",
        "≈5M parâmetros, ~270 MB, passos ~5x mais lentos (cada especialista novo custa bem mais memória)",
        "≈15M parâmetros, ~420 MB, passos ~15x mais lentos (quase sem espaço para especialistas novos)"
    )

    fun config(size: Int, v: Int): MiniGptConfig = when (size) {
        1 -> MiniGptConfig(v = v, c = 256, h = 8, l = 6, maxThreads = 4, lr = 6e-4f)
        2 -> MiniGptConfig(v = v, c = 384, h = 12, l = 8, maxThreads = 2, lr = 4e-4f)
        else -> MiniGptConfig(v = v)
    }
}

/** Vocabulário fixo (o mini.py montava pelo texto; aqui precisa ser fixo para treinar aos poucos). */
object Vocab {
    val chars: String = "\n" + (32..126).map { it.toChar() }.joinToString("") +
        "áàâãäéèêëíìîïóòôõöúùûüçñÁÀÂÃÄÉÈÊÍÌÓÒÔÕÖÚÙÛÜÇÑ–—“”‘’…°ªº€•"
    private val index = HashMap<Char, Int>()

    init {
        for (i in chars.indices) index[chars[i]] = i
    }

    val size: Int get() = chars.length

    fun encode(s: String): IntArray {
        val out = IntArray(s.length)
        var k = 0
        for (ch in s) {
            val id = index[ch]
            if (id != null) out[k++] = id
        }
        return out.copyOf(k)
    }
}

class Layout(cfg: MiniGptConfig, val ex: Int = cfg.ex) {
    private val c = cfg.c
    private val nl = cfg.l
    var n = 0
        private set

    private fun alloc(sz: Int): Int {
        val o = n
        n += sz
        return o
    }

    val tok = alloc(cfg.v * c)
    val pos = alloc(cfg.t * c)
    val node = alloc(cfg.s * c)
    val ln1g = IntArray(nl)
    val ln1b = IntArray(nl)
    val wqkv = IntArray(nl)
    val bqkv = IntArray(nl)
    val wo = IntArray(nl)
    val bo = IntArray(nl)
    val ln2g = IntArray(nl)
    val ln2b = IntArray(nl)
    val w1 = IntArray(nl)
    val b1 = IntArray(nl)
    val w2 = IntArray(nl)
    val b2 = IntArray(nl)
    val lnfg: Int
    val lnfb: Int
    val wout: Int
    val bout: Int

    /** Especialistas extras (1..ex-1) de cada camada e o roteador. O especialista 0 é o w1/b1/w2/b2 de sempre. */
    val exW1 = Array(nl) { IntArray(maxOf(0, ex - 1)) }
    val exB1 = Array(nl) { IntArray(maxOf(0, ex - 1)) }
    val exW2 = Array(nl) { IntArray(maxOf(0, ex - 1)) }
    val exB2 = Array(nl) { IntArray(maxOf(0, ex - 1)) }

    /** rr[l][e] = começo da linha do roteador do especialista e na camada l (só vale com ex >= 2) */
    val rr = Array(nl) { IntArray(ex) }

    /** Tamanho do layout sem especialistas (= o modelo salvo antes da v9). Os extras ficam no fim. */
    var baseN = 0
        private set

    fun fw1(l: Int, e: Int): Int = if (e == 0) w1[l] else exW1[l][e - 1]
    fun fb1(l: Int, e: Int): Int = if (e == 0) b1[l] else exB1[l][e - 1]
    fun fw2(l: Int, e: Int): Int = if (e == 0) w2[l] else exW2[l][e - 1]
    fun fb2(l: Int, e: Int): Int = if (e == 0) b2[l] else exB2[l][e - 1]

    init {
        for (i in 0 until nl) {
            ln1g[i] = alloc(c); ln1b[i] = alloc(c)
            wqkv[i] = alloc(3 * c * c); bqkv[i] = alloc(3 * c)
            wo[i] = alloc(c * c); bo[i] = alloc(c)
            ln2g[i] = alloc(c); ln2b[i] = alloc(c)
            w1[i] = alloc(4 * c * c); b1[i] = alloc(4 * c)
            w2[i] = alloc(4 * c * c); b2[i] = alloc(c)
        }
        lnfg = alloc(c); lnfb = alloc(c)
        wout = alloc(cfg.v * c); bout = alloc(cfg.v)
        baseN = n
        // Cada especialista novo é um bloco no FIM: os offsets dos antigos nunca mudam quando o modelo cresce.
        for (e in 1 until ex) {
            for (i in 0 until nl) {
                exW1[i][e - 1] = alloc(4 * c * c); exB1[i][e - 1] = alloc(4 * c)
                exW2[i][e - 1] = alloc(4 * c * c); exB2[i][e - 1] = alloc(c)
            }
            for (i in 0 until nl) rr[i][e] = alloc(c)
            if (e == 1) for (i in 0 until nl) rr[i][0] = alloc(c)
        }
    }
}

private const val K0 = 0.7978846f

private fun gelu(x: Float): Float {
    val u = K0 * (x + 0.044715f * x * x * x)
    return 0.5f * x * (1f + tanh(u))
}

private fun geluGrad(x: Float): Float {
    val u = K0 * (x + 0.044715f * x * x * x)
    val th = tanh(u)
    return 0.5f * (1f + th) + 0.5f * x * (1f - th * th) * K0 * (1f + 3f * 0.044715f * x * x)
}

/** y = x W^T + b  (W guardado como [saída][entrada], igual ao PyTorch) */
private fun linFwd(x: FloatArray, nn: Int, inD: Int, w: FloatArray, wOff: Int, bOff: Int, outD: Int, y: FloatArray) {
    for (t in 0 until nn) {
        val xr = t * inD
        val yr = t * outD
        for (o in 0 until outD) {
            var s = w[bOff + o]
            val wr = wOff + o * inD
            for (i in 0 until inD) s += x[xr + i] * w[wr + i]
            y[yr + o] = s
        }
    }
}

/** Acumula gradientes em g e em dx (dx precisa estar zerado antes). */
private fun linBwd(
    x: FloatArray, dy: FloatArray, nn: Int, inD: Int, w: FloatArray,
    wOff: Int, bOff: Int, outD: Int, g: FloatArray, dx: FloatArray
) {
    for (t in 0 until nn) {
        val xr = t * inD
        val yr = t * outD
        for (o in 0 until outD) {
            val d = dy[yr + o]
            if (d == 0f) continue
            g[bOff + o] += d
            val wr = wOff + o * inD
            for (i in 0 until inD) {
                g[wr + i] += d * x[xr + i]
                dx[xr + i] += d * w[wr + i]
            }
        }
    }
}

private fun lnFwd(
    x: FloatArray, nn: Int, c: Int, w: FloatArray, gOff: Int, bOff: Int,
    xhat: FloatArray, rs: FloatArray, y: FloatArray
) {
    for (t in 0 until nn) {
        val r = t * c
        var mean = 0f
        for (i in 0 until c) mean += x[r + i]
        mean /= c
        var vr = 0f
        for (i in 0 until c) {
            val d = x[r + i] - mean
            vr += d * d
        }
        vr /= c
        val rstd = 1f / sqrt(vr + 1e-5f)
        rs[t] = rstd
        for (i in 0 until c) {
            val xh = (x[r + i] - mean) * rstd
            xhat[r + i] = xh
            y[r + i] = xh * w[gOff + i] + w[bOff + i]
        }
    }
}

/** Acumula em dx. */
private fun lnBwd(
    dy: FloatArray, nn: Int, c: Int, w: FloatArray, gOff: Int, bOff: Int,
    xhat: FloatArray, rs: FloatArray, g: FloatArray, dx: FloatArray
) {
    for (t in 0 until nn) {
        val r = t * c
        var m1 = 0f
        var m2 = 0f
        for (i in 0 until c) {
            val d = dy[r + i]
            g[gOff + i] += d * xhat[r + i]
            g[bOff + i] += d
            val dxh = d * w[gOff + i]
            m1 += dxh
            m2 += dxh * xhat[r + i]
        }
        m1 /= c
        m2 /= c
        for (i in 0 until c) {
            val dxh = dy[r + i] * w[gOff + i]
            dx[r + i] += rs[t] * (dxh - m1 - xhat[r + i] * m2)
        }
    }
}

/** Memória de trabalho de uma sequência (uma por thread). */
/** Peso da perda de equilíbrio entre especialistas (Switch Transformer): empurra o roteador a repartir o trabalho. */
private const val AUX_ALPHA = 0.02f

class Work(cfg: MiniGptConfig, private val lay: Layout, private val active: IntArray) {
    private val T = cfg.t
    private val C = cfg.c
    private val NH = cfg.h
    private val L = cfg.l
    private val V = cfg.v
    private val D = cfg.c / cfg.h

    val g = FloatArray(lay.n)

    private val hs = Array(L + 1) { FloatArray(T * C) }
    private val ln1x = Array(L) { FloatArray(T * C) }
    private val ln1rs = Array(L) { FloatArray(T) }
    private val a = Array(L) { FloatArray(T * C) }
    private val qkv = Array(L) { FloatArray(T * 3 * C) }
    private val p = Array(L) { FloatArray(NH * T * T) }
    private val ao = Array(L) { FloatArray(T * C) }
    private val h1 = Array(L) { FloatArray(T * C) }
    private val ln2x = Array(L) { FloatArray(T * C) }
    private val ln2rs = Array(L) { FloatArray(T) }
    private val bn = Array(L) { FloatArray(T * C) }
    private val f1 = Array(L) { FloatArray(T * 4 * C) }
    private val gl = Array(L) { FloatArray(T * 4 * C) }
    private val lnfx = FloatArray(T * C)
    private val lnfrs = FloatArray(T)
    private val z = FloatArray(T * C)
    val logits = FloatArray(T * V)
    private val tmp = FloatArray(T * C)
    private val dz = FloatArray(T * C)
    private val dh = FloatArray(T * C)
    private val dh1 = FloatArray(T * C)
    private val dao = FloatArray(T * C)
    private val dqkv = FloatArray(T * 3 * C)
    private val da = FloatArray(T * C)
    private val dbn = FloatArray(T * C)
    private val dgl = FloatArray(T * 4 * C)
    private val dpTmp = FloatArray(T)
    private val ns = FloatArray(C)

    // especialistas (MoE): escolha, peso (gate), probabilidades do roteador e saída do especialista antes do gate
    private val EX = lay.ex
    private val sel = Array(L) { IntArray(T) }
    private val gate = Array(L) { FloatArray(T) }
    private val rp = Array(L) { FloatArray(T * EX) }
    private val fo = Array(L) { FloatArray(if (EX > 1) T * C else 0) }
    /** quantas vezes cada especialista foi escolhido (para copiar o mais usado ao crescer) */
    val use = IntArray(L * EX)
    private val cntTmp = IntArray(EX)
    private val psTmp = FloatArray(EX)
    private val fTmp = FloatArray(EX)
    private var auxSum = 0f

    /** Feed-forward com k especialistas: o roteador escolhe UM por posição; saída = k * prob * especialista(x). */
    private fun moeFwd(w: FloatArray, l: Int, nn: Int, k: Int) {
        val x = bn[l]
        val f = f1[l]
        val gg = gl[l]
        val o = fo[l]
        val pr = rp[l]
        for (e in 0 until k) {
            cntTmp[e] = 0
            psTmp[e] = 0f
        }
        for (t in 0 until nn) {
            val xo = t * C
            var mx = -1e30f
            for (j in 0 until k) {
                var sc = 0f
                val ro = lay.rr[l][j]
                for (i in 0 until C) sc += x[xo + i] * w[ro + i]
                pr[t * EX + j] = sc
                if (sc > mx) mx = sc
            }
            var sm = 0f
            for (j in 0 until k) {
                val e = exp(pr[t * EX + j] - mx)
                pr[t * EX + j] = e
                sm += e
            }
            var best = 0
            var bp = -1f
            for (j in 0 until k) {
                pr[t * EX + j] /= sm
                if (pr[t * EX + j] > bp) {
                    bp = pr[t * EX + j]
                    best = j
                }
            }
            sel[l][t] = best
            use[l * EX + best]++
            cntTmp[best]++
            for (j in 0 until k) psTmp[j] += pr[t * EX + j]
            val gt = k * bp
            gate[l][t] = gt
            val o1 = lay.fw1(l, best)
            val b1o = lay.fb1(l, best)
            val o2 = lay.fw2(l, best)
            val b2o = lay.fb2(l, best)
            for (h in 0 until 4 * C) {
                var sc = w[b1o + h]
                val wr = o1 + h * C
                for (i in 0 until C) sc += x[xo + i] * w[wr + i]
                f[t * 4 * C + h] = sc
                gg[t * 4 * C + h] = gelu(sc)
            }
            for (oo in 0 until C) {
                var sc = w[b2o + oo]
                val wr = o2 + oo * 4 * C
                for (h in 0 until 4 * C) sc += gg[t * 4 * C + h] * w[wr + h]
                o[xo + oo] = sc
                tmp[xo + oo] = sc * gt
            }
        }
        // perda de equilíbrio: nn * alfa * k * soma(fração de posições do especialista * prob. média dele)
        var sumFP = 0f
        for (e in 0 until k) sumFP += (cntTmp[e].toFloat() / nn) * (psTmp[e] / nn)
        auxSum += nn * AUX_ALPHA * k * sumFP
    }

    /** Gradiente do feed-forward com especialistas. Usa dh; deixa dbn pronto (dbn é zerado aqui). */
    private fun moeBwd(w: FloatArray, l: Int, nn: Int, k: Int, scale: Float) {
        val x = bn[l]
        val f = f1[l]
        val gg = gl[l]
        val o = fo[l]
        val pr = rp[l]
        Arrays.fill(dbn, 0f)
        for (e in 0 until k) fTmp[e] = 0f
        for (t in 0 until nn) fTmp[sel[l][t]] += 1f
        for (e in 0 until k) fTmp[e] = fTmp[e] / nn.toFloat()
        for (t in 0 until nn) {
            val xo = t * C
            val e = sel[l][t]
            val gt = gate[l][t]
            var dgate = 0f
            for (oo in 0 until C) dgate += dh[xo + oo] * o[xo + oo]
            val o1 = lay.fw1(l, e)
            val b1o = lay.fb1(l, e)
            val o2 = lay.fw2(l, e)
            val b2o = lay.fb2(l, e)
            val ho = t * 4 * C
            for (h in 0 until 4 * C) dgl[ho + h] = 0f
            for (oo in 0 until C) {
                val dy = dh[xo + oo] * gt
                g[b2o + oo] += dy
                val wr = o2 + oo * 4 * C
                for (h in 0 until 4 * C) {
                    g[wr + h] += dy * gg[ho + h]
                    dgl[ho + h] += dy * w[wr + h]
                }
            }
            for (h in 0 until 4 * C) {
                val d = dgl[ho + h] * geluGrad(f[ho + h])
                g[b1o + h] += d
                val wr = o1 + h * C
                for (i in 0 until C) {
                    g[wr + i] += d * x[xo + i]
                    dbn[xo + i] += d * w[wr + i]
                }
            }
            val dp = k * dgate
            val ps = pr[t * EX + e]
            var sS = 0f
            for (j in 0 until k) sS += fTmp[j] * pr[t * EX + j]
            for (j in 0 until k) {
                val pj = pr[t * EX + j]
                val dl = dp * ps * ((if (j == e) 1f else 0f) - pj) + scale * AUX_ALPHA * k * (fTmp[j] * pj - pj * sS)
                val ro = lay.rr[l][j]
                for (i in 0 until C) {
                    g[ro + i] += dl * x[xo + i]
                    dbn[xo + i] += dl * w[ro + i]
                }
            }
        }
    }

    /** slots = caminho do nó na árvore: a soma dos vetores de galho condiciona todas as posições. */
    fun forward(w: FloatArray, x: IntArray, nn: Int, slots: IntArray) {
        auxSum = 0f
        Arrays.fill(ns, 0f)
        for (sl in slots) {
            val so = lay.node + sl * C
            for (i in 0 until C) ns[i] += w[so + i]
        }
        for (t in 0 until nn) {
            val ro = lay.tok + x[t] * C
            val po = lay.pos + t * C
            for (i in 0 until C) hs[0][t * C + i] = w[ro + i] + w[po + i] + ns[i]
        }
        val sc = 1f / sqrt(D.toFloat())
        for (l in 0 until L) {
            lnFwd(hs[l], nn, C, w, lay.ln1g[l], lay.ln1b[l], ln1x[l], ln1rs[l], a[l])
            linFwd(a[l], nn, C, w, lay.wqkv[l], lay.bqkv[l], 3 * C, qkv[l])
            val q = qkv[l]
            val pp = p[l]
            val aol = ao[l]
            for (hd in 0 until NH) {
                for (t in 0 until nn) {
                    val qo = t * 3 * C + hd * D
                    val pRow = hd * T * T + t * T
                    var mx = -1e30f
                    for (u in 0..t) {
                        val ko = u * 3 * C + C + hd * D
                        var s = 0f
                        for (i in 0 until D) s += q[qo + i] * q[ko + i]
                        s *= sc
                        pp[pRow + u] = s
                        if (s > mx) mx = s
                    }
                    var sm = 0f
                    for (u in 0..t) {
                        val e = exp(pp[pRow + u] - mx)
                        pp[pRow + u] = e
                        sm += e
                    }
                    val inv = 1f / sm
                    for (u in 0..t) pp[pRow + u] *= inv
                    val oo = t * C + hd * D
                    for (i in 0 until D) aol[oo + i] = 0f
                    for (u in 0..t) {
                        val vo = u * 3 * C + 2 * C + hd * D
                        val pu = pp[pRow + u]
                        for (i in 0 until D) aol[oo + i] += pu * q[vo + i]
                    }
                }
            }
            linFwd(aol, nn, C, w, lay.wo[l], lay.bo[l], C, tmp)
            val h0 = hs[l]
            val h1l = h1[l]
            for (i in 0 until nn * C) h1l[i] = h0[i] + tmp[i]
            lnFwd(h1l, nn, C, w, lay.ln2g[l], lay.ln2b[l], ln2x[l], ln2rs[l], bn[l])
            val kx = active[l]
            if (kx <= 1) {
                linFwd(bn[l], nn, C, w, lay.w1[l], lay.b1[l], 4 * C, f1[l])
                val f = f1[l]
                val gg = gl[l]
                for (i in 0 until nn * 4 * C) gg[i] = gelu(f[i])
                linFwd(gg, nn, 4 * C, w, lay.w2[l], lay.b2[l], C, tmp)
            } else {
                moeFwd(w, l, nn, kx)
            }
            val hn = hs[l + 1]
            for (i in 0 until nn * C) hn[i] = h1l[i] + tmp[i]
        }
        lnFwd(hs[L], nn, C, w, lay.lnfg, lay.lnfb, lnfx, lnfrs, z)
        linFwd(z, nn, C, w, lay.wout, lay.bout, V, logits)
    }

    /** Soma da entropia cruzada dos nn tokens (sem dividir). */
    fun lossOf(w: FloatArray, x: IntArray, y: IntArray, nn: Int, slots: IntArray, withAux: Boolean = false): Float {
        forward(w, x, nn, slots)
        var tot = 0f
        for (t in 0 until nn) {
            val r = t * V
            var mx = logits[r]
            for (j in 1 until V) if (logits[r + j] > mx) mx = logits[r + j]
            var s = 0f
            for (j in 0 until V) s += exp(logits[r + j] - mx)
            tot += -(logits[r + y[t]] - mx - ln(s))
        }
        return if (withAux) tot + auxSum else tot
    }

    /** Forward + backward. Acumula o gradiente (multiplicado por scale) em g. Devolve a soma da perda. */
    fun trainSeq(w: FloatArray, x: IntArray, y: IntArray, nn: Int, scale: Float, slots: IntArray): Float {
        forward(w, x, nn, slots)
        var tot = 0f
        for (t in 0 until nn) {
            val r = t * V
            var mx = logits[r]
            for (j in 1 until V) if (logits[r + j] > mx) mx = logits[r + j]
            var s = 0f
            for (j in 0 until V) {
                val e = exp(logits[r + j] - mx)
                logits[r + j] = e
                s += e
            }
            for (j in 0 until V) logits[r + j] /= s
            tot += -ln(max(logits[r + y[t]], 1e-12f))
            for (j in 0 until V) {
                val dl = logits[r + j] - (if (j == y[t]) 1f else 0f)
                logits[r + j] = dl * scale
            }
        }
        Arrays.fill(dz, 0f)
        linBwd(z, logits, nn, C, w, lay.wout, lay.bout, V, g, dz)
        Arrays.fill(dh, 0f)
        lnBwd(dz, nn, C, w, lay.lnfg, lay.lnfb, lnfx, lnfrs, g, dh)
        val sc = 1f / sqrt(D.toFloat())
        for (l in L - 1 downTo 0) {
            val kx = active[l]
            if (kx <= 1) {
                Arrays.fill(dgl, 0f)
                linBwd(gl[l], dh, nn, 4 * C, w, lay.w2[l], lay.b2[l], C, g, dgl)
                val f = f1[l]
                for (i in 0 until nn * 4 * C) dgl[i] *= geluGrad(f[i])
                Arrays.fill(dbn, 0f)
                linBwd(bn[l], dgl, nn, C, w, lay.w1[l], lay.b1[l], 4 * C, g, dbn)
            } else {
                moeBwd(w, l, nn, kx, scale)
            }
            System.arraycopy(dh, 0, dh1, 0, dh.size)
            lnBwd(dbn, nn, C, w, lay.ln2g[l], lay.ln2b[l], ln2x[l], ln2rs[l], g, dh1)
            Arrays.fill(dao, 0f)
            linBwd(ao[l], dh1, nn, C, w, lay.wo[l], lay.bo[l], C, g, dao)
            Arrays.fill(dqkv, 0f)
            val q = qkv[l]
            val pp = p[l]
            for (hd in 0 until NH) {
                for (t in 0 until nn) {
                    val pRow = hd * T * T + t * T
                    val oo = t * C + hd * D
                    var dot = 0f
                    for (u in 0..t) {
                        val vo = u * 3 * C + 2 * C + hd * D
                        var s = 0f
                        for (i in 0 until D) s += dao[oo + i] * q[vo + i]
                        dpTmp[u] = s
                        dot += pp[pRow + u] * s
                        for (i in 0 until D) dqkv[vo + i] += pp[pRow + u] * dao[oo + i]
                    }
                    for (u in 0..t) {
                        val ds = pp[pRow + u] * (dpTmp[u] - dot) * sc
                        val qo = t * 3 * C + hd * D
                        val ko = u * 3 * C + C + hd * D
                        for (i in 0 until D) {
                            dqkv[qo + i] += ds * q[ko + i]
                            dqkv[ko + i] += ds * q[qo + i]
                        }
                    }
                }
            }
            Arrays.fill(da, 0f)
            linBwd(a[l], dqkv, nn, C, w, lay.wqkv[l], lay.bqkv[l], 3 * C, g, da)
            lnBwd(da, nn, C, w, lay.ln1g[l], lay.ln1b[l], ln1x[l], ln1rs[l], g, dh1)
            System.arraycopy(dh1, 0, dh, 0, dh.size)
        }
        for (t in 0 until nn) {
            val ro = lay.tok + x[t] * C
            val po = lay.pos + t * C
            for (i in 0 until C) {
                g[ro + i] += dh[t * C + i]
                g[po + i] += dh[t * C + i]
            }
        }
        for (i in 0 until C) {
            var sum = 0f
            for (t in 0 until nn) sum += dh[t * C + i]
            for (sl in slots) g[lay.node + sl * C + i] += sum
        }
        return tot
    }
}

class MiniGpt(val cfg: MiniGptConfig) {
    var lay = Layout(cfg)
        private set
    var w = FloatArray(lay.n)
        private set
    private var am = FloatArray(lay.n)
    private var av = FloatArray(lay.n)
    private var adamT = 0

    @Volatile var step = 0
    @Volatile var lastLoss = Float.NaN

    private val nThreads = max(1, min(cfg.maxThreads, Runtime.getRuntime().availableProcessors()))
    private val pool = Executors.newFixedThreadPool(nThreads) { r -> Thread(r).also { it.isDaemon = true } }
    /** especialistas ativos em cada camada (1 = feed-forward comum) */
    val active = IntArray(cfg.l) { 1 }
    private var works = Array(nThreads) { Work(cfg, lay, active) }

    init {
        initWeights(Random(1))
    }

    private fun initWeights(rnd: Random) {
        Arrays.fill(w, 0f)
        Arrays.fill(am, 0f)
        Arrays.fill(av, 0f)
        adamT = 0
        step = 0
        lastLoss = Float.NaN
        val c = cfg.c
        fun normal(off: Int, size: Int, std: Float) {
            for (i in 0 until size) w[off + i] = (rnd.nextGaussian() * std).toFloat()
        }
        fun ones(off: Int) {
            for (i in 0 until c) w[off + i] = 1f
        }
        Arrays.fill(active, 1)
        val deep = 0.02f / sqrt(2f * cfg.l)
        normal(lay.tok, cfg.v * c, 0.02f)
        normal(lay.pos, cfg.t * c, 0.02f)
        for (l in 0 until cfg.l) {
            ones(lay.ln1g[l]); ones(lay.ln2g[l])
            normal(lay.wqkv[l], 3 * c * c, 0.02f)
            normal(lay.wo[l], c * c, deep)
            normal(lay.w1[l], 4 * c * c, 0.02f)
            normal(lay.w2[l], 4 * c * c, deep)
        }
        ones(lay.lnfg)
        normal(lay.wout, cfg.v * c, 0.02f)
        // especialistas extras ficam dormindo (não contam) até o modelo crescer; aqui só têm valores pequenos
        for (l in 0 until cfg.l) {
            for (e in 1 until lay.ex) {
                normal(lay.fw1(l, e), 4 * c * c, 0.02f)
                normal(lay.fw2(l, e), 4 * c * c, deep)
            }
            if (lay.ex > 1) for (e in 0 until lay.ex) normal(lay.rr[l][e], c, 0.02f)
        }
    }

    /** Memória livre estimada (MB) e custo (MB) de chegar a `ex` especialistas por camada. */
    fun growReport(ex: Int): String {
        val nl = Layout(cfg, ex)
        val cost = nl.n.toLong() * 4L * (3 + nThreads) / 1048576L
        val rt = Runtime.getRuntime()
        val free = (rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())) / 1048576L
        return "chegar a " + ex + " especialistas por camada pede ~" + cost + " MB no total; o app tem ~" + free + " MB livres"
    }

    /** O novo tamanho cabe na memória do app, no pior momento da troca (arrays velhos e novos juntos)? */
    private fun fits(newN: Int): Boolean {
        val rt = Runtime.getRuntime()
        val used = rt.totalMemory() - rt.freeMemory()
        val oldWorks = nThreads.toLong() * lay.n * 4L
        val old3 = 3L * lay.n * 4L
        val new3 = 3L * newN * 4L
        val newWorks = nThreads.toLong() * newN * 4L
        val margin = 48L * 1048576L
        val peak1 = used - oldWorks + new3
        val peak2 = used - oldWorks - old3 + new3 + newWorks
        return maxOf(peak1, peak2) + margin < rt.maxMemory()
    }

    /** Recria os arrays para `ex` especialistas por camada, sem copiar nada (usado ao carregar e ao reiniciar). */
    private fun resizeEmpty(ex: Int) {
        works = emptyArray()
        lay = Layout(cfg, ex)
        w = FloatArray(lay.n)
        am = FloatArray(lay.n)
        av = FloatArray(lay.n)
        Arrays.fill(active, ex)
        works = Array(nThreads) { Work(cfg, lay, active) }
    }

    /**
     * Cresce sem teto fixo: tenta DOBRAR os especialistas de cada camada (1 -> 2 -> 4 -> 8...) e, se a
     * memória não der, tenta metade, e assim por diante. Devolve quantos nasceram por camada (0 = nenhum coube).
     * Chame com o modelo travado, entre dois passos.
     */
    fun grow(): Int {
        val k = lay.ex
        var add = max(1, k)
        System.gc()
        while (add >= 1) {
            val newEx = k + add
            if (fits(Layout(cfg, newEx).n) && resizeGrow(newEx)) return add
            add /= 2
        }
        return 0
    }

    /**
     * Cada especialista novo copia (com ruído de 2%) um dos antigos, sorteado com chance proporcional ao
     * uso, e a linha do roteador dele também. Os blocos antigos ficam onde estavam.
     */
    private fun resizeGrow(newEx: Int): Boolean {
        val k = lay.ex
        val c = cfg.c
        val rnd = Random(System.nanoTime())
        val usage = IntArray(cfg.l * k)
        for (l in 0 until cfg.l) {
            for (e in 0 until k) {
                var u = 0
                for (wk in works) u += wk.use[l * k + e]
                usage[l * k + e] = u
            }
        }
        val oldLay = lay
        val oldW = w
        val oldAm = am
        val oldAv = av
        try {
            works = emptyArray() // libera os rascunhos de gradiente antes de alocar os novos arrays
            val nl = Layout(cfg, newEx)
            val nw = FloatArray(nl.n)
            val nam = FloatArray(nl.n)
            val nav = FloatArray(nl.n)
            System.arraycopy(oldW, 0, nw, 0, oldLay.n)
            System.arraycopy(oldAm, 0, nam, 0, oldLay.n)
            System.arraycopy(oldAv, 0, nav, 0, oldLay.n)
            fun noisy(from: Int, to: Int, size: Int, abs: Float) {
                for (i in 0 until size) {
                    val v = nw[from + i]
                    nw[to + i] = v * (1f + 0.02f * rnd.nextGaussian().toFloat()) + abs * rnd.nextGaussian().toFloat()
                }
            }
            if (k == 1) {
                // primeira vez: o roteador ainda não existia; a linha do especialista 0 começa pequena
                for (l in 0 until cfg.l) for (i in 0 until c) nw[nl.rr[l][0] + i] = (rnd.nextGaussian() * 0.02).toFloat()
            }
            for (l in 0 until cfg.l) {
                var tot = 0L
                for (e in 0 until k) tot += usage[l * k + e] + 1
                for (j in k until newEx) {
                    var r = (rnd.nextDouble() * tot).toLong()
                    var src = 0
                    for (e in 0 until k) {
                        r -= usage[l * k + e] + 1
                        if (r < 0) {
                            src = e
                            break
                        }
                    }
                    noisy(nl.fw1(l, src), nl.fw1(l, j), 4 * c * c, 1e-4f)
                    noisy(nl.fb1(l, src), nl.fb1(l, j), 4 * c, 1e-4f)
                    noisy(nl.fw2(l, src), nl.fw2(l, j), 4 * c * c, 1e-4f)
                    noisy(nl.fb2(l, src), nl.fb2(l, j), c, 1e-4f)
                    noisy(nl.rr[l][src], nl.rr[l][j], c, 0.01f)
                }
            }
            lay = nl
            w = nw
            am = nam
            av = nav
            Arrays.fill(active, newEx)
            works = Array(nThreads) { Work(cfg, lay, active) }
            return true
        } catch (e: OutOfMemoryError) {
            lay = oldLay
            w = oldW
            am = oldAm
            av = oldAv
            Arrays.fill(active, oldLay.ex)
            works = Array(nThreads) { Work(cfg, lay, active) }
            return false
        }
    }

    /** Perda (soma) de uma sequência com o modelo atual. Usado no autoteste. */
    fun lossOf(x: IntArray, y: IntArray, slots: IntArray): Float = works[0].lossOf(w, x, y, x.size, slots)

    fun reset() {
        if (lay.ex != cfg.ex) resizeEmpty(cfg.ex)
        initWeights(Random(System.nanoTime()))
    }

    /** Um passo de treino (AdamW, como no mini.py) sobre as sequências dadas. Devolve a perda média por caractere. */
    fun trainBatch(seqs: List<Seq>, lr: Float): Float {
        var tokens = 0
        for (q in seqs) tokens += q.x.size
        val scale = 1f / tokens
        for (wk in works) Arrays.fill(wk.g, 0f)
        val losses = FloatArray(nThreads)
        val tasks = (0 until nThreads).map { ti ->
            Callable<Unit> {
                val wk = works[ti]
                var loss = 0f
                var b = ti
                while (b < seqs.size) {
                    val q = seqs[b]
                    loss += wk.trainSeq(w, q.x, q.y, q.x.size, scale, q.slots)
                    b += nThreads
                }
                losses[ti] = loss
                Unit
            }
        }
        for (f in pool.invokeAll(tasks)) f.get()

        val g = works[0].g
        for (ti in 1 until nThreads) {
            val o = works[ti].g
            for (i in g.indices) g[i] += o[i]
        }
        var sq = 0.0
        for (i in g.indices) sq += (g[i] * g[i]).toDouble()
        val norm = sqrt(sq).toFloat()
        val clip = if (norm > 1f) 1f / norm else 1f

        adamT++
        val b1 = 0.9f
        val b2 = 0.999f
        val c1 = 1f - b1.pow(adamT)
        val c2 = 1f - b2.pow(adamT)
        val decay = 1f - lr * 0.01f
        for (i in w.indices) {
            val gi = g[i] * clip
            am[i] = b1 * am[i] + (1f - b1) * gi
            av[i] = b2 * av[i] + (1f - b2) * gi * gi
            w[i] = w[i] * decay - lr * (am[i] / c1) / (sqrt(av[i] / c2) + 1e-8f)
        }
        var total = 0f
        for (x in losses) total += x
        val mean = total / tokens
        step++
        lastLoss = mean
        return mean
    }

    /**
     * Gera texto. slots = galho da árvore que condiciona a geração (vazio = livre).
     * Começa com "\n" (início de nó) e, com stopAtNewline, para quando o modelo encerra o nó.
     */
    fun generate(slots: IntArray, prompt: String, maxNew: Int, temp: Float, rnd: Random, stopAtNewline: Boolean): String {
        val T = cfg.t
        val V = cfg.v
        val ids = ArrayList<Int>()
        ids.add(0)
        for (x in Vocab.encode(prompt)) ids.add(x)
        val wk = works[0]
        val out = StringBuilder()
        val probs = FloatArray(V)
        for (k in 0 until maxNew) {
            val nn = min(ids.size, T)
            val ctx = IntArray(nn) { ids[ids.size - nn + it] }
            wk.forward(w, ctx, nn, slots)
            val off = (nn - 1) * V
            var mx = wk.logits[off]
            for (j in 1 until V) if (wk.logits[off + j] > mx) mx = wk.logits[off + j]
            var s = 0f
            for (j in 0 until V) {
                val e = if (j == 0 && stopAtNewline && out.length < 8) 0f else exp((wk.logits[off + j] - mx) / temp)
                probs[j] = e
                s += e
            }
            var r = rnd.nextFloat() * s
            var pick = V - 1
            for (j in 0 until V) {
                r -= probs[j]
                if (r <= 0f) {
                    pick = j
                    break
                }
            }
            if (pick == 0 && stopAtNewline) break
            ids.add(pick)
            out.append(Vocab.chars[pick])
        }
        return out.toString()
    }

    private fun meta(): ByteArray {
        val bb = ByteBuffer.allocate(10 * 4)
        bb.putInt(2); bb.putInt(step); bb.putFloat(lastLoss); bb.putInt(lay.n)
        bb.putInt(cfg.v); bb.putInt(cfg.t); bb.putInt(cfg.c); bb.putInt(cfg.h); bb.putInt(cfg.l); bb.putInt(cfg.s)
        return bb.array()
    }

    fun save(db: Db) {
        db.saveModel(meta(), w)
    }

    /**
     * Carrega os pesos salvos no banco; devolve false se não houver ou se o tamanho não bater.
     * Se o modelo salvo tem contexto menor (v6: 64), migra: copia embeddings, vetores de galho e
     * todas as camadas; as posições novas (além das antigas) começam com valores pequenos aleatórios.
     */
    fun load(db: Db): Boolean {
        val mb = db.loadModelMeta() ?: return false
        if (mb.size < 40) return false
        val bb = ByteBuffer.wrap(mb)
        val ver = bb.getInt()
        val st = bb.getInt()
        val ls = bb.getFloat()
        val n = bb.getInt()
        val v = bb.getInt()
        val t = bb.getInt()
        val c = bb.getInt()
        val h = bb.getInt()
        val l = bb.getInt()
        val sl = bb.getInt()
        if (v != cfg.v || c != cfg.c || h != cfg.h || l != cfg.l || sl != cfg.s) return false
        if (t != cfg.t) return if (t < cfg.t) migrate(db, st, ls, n, t) else false
        if (ver < 2 && n != lay.baseN) {
            // v9 guardou especialistas num layout antigo (um bloco por camada): aproveita só a parte comum
            if (n < lay.baseN) return false
            val old = FloatArray(n)
            if (!db.loadModelWeights(old)) return false
            if (lay.ex != 1) resizeEmpty(1)
            System.arraycopy(old, 0, w, 0, lay.baseN)
            step = st
            lastLoss = ls
            return true
        }
        val ex = capacityFor(n) ?: return false
        if (ex != lay.ex) resizeEmpty(ex)
        if (!db.loadModelWeights(w)) {
            resizeEmpty(cfg.ex)
            initWeights(Random(1))
            return false
        }
        Arrays.fill(active, lay.ex)
        step = st
        lastLoss = ls
        return true
    }

    /** Quantos especialistas por camada um modelo salvo com n parâmetros tem (null = tamanho que não bate). */
    private fun capacityFor(n: Int): Int? {
        if (n == lay.baseN) return 1
        val l2 = Layout(cfg, 2).n
        val g = Layout(cfg, 3).n - l2
        val rest = n - l2
        if (rest < 0 || g <= 0 || rest % g != 0) return null
        return 2 + rest / g
    }

    private fun migrate(db: Db, st: Int, ls: Float, n: Int, oldT: Int): Boolean {
        val oldLay = Layout(MiniGptConfig(v = cfg.v, t = oldT, c = cfg.c, h = cfg.h, l = cfg.l, s = cfg.s))
        if (oldLay.n != n) return false
        val old = FloatArray(n)
        if (!db.loadModelWeights(old)) return false
        val c = cfg.c
        System.arraycopy(old, oldLay.tok, w, lay.tok, cfg.v * c)
        System.arraycopy(old, oldLay.pos, w, lay.pos, oldT * c)
        System.arraycopy(old, oldLay.node, w, lay.node, n - oldLay.node)
        step = st
        lastLoss = ls
        return true
    }
}

class Seq(val slots: IntArray, val x: IntArray, val y: IntArray)

class Doc(val id: Long, val slots: IntArray, val ids: IntArray)

/**
 * A árvore DENTRO do transformer: cada nó tem um vetor próprio (vetor de galho). O vetor de um nó
 * é somado aos vetores de todos os seus ancestrais, então a hierarquia da árvore vira parte do modelo.
 * Treino e respostas usam o galho do nó; o texto de cada nó é o que o modelo aprende a reproduzir.
 */
class MiniLab(private val db: Db, val size: Int = 0) {
    val gpt = MiniGpt(Sizes.config(size, Vocab.size))
    val lock = Any()
    private val bg = Executors.newSingleThreadExecutor { r -> Thread(r).also { it.isDaemon = true } }

    init {
        gpt.load(db)
    }

    private var emaLoss = Float.NaN
    private var refLoss = Float.NaN
    private var lastCheck = gpt.step

    /** Especialistas por camada agora (sem teto fixo: o limite é a memória do celular). */
    fun expertsInfo(): String = gpt.lay.ex.toString()

    /**
     * Crescimento autônomo e exponencial: a cada 300 passos compara a média móvel da perda com a da
     * checagem anterior. Se melhorou menos de 2% (platô) e a perda ainda está alta, os especialistas de
     * cada camada DOBRAM (1, 2, 4, 8...), se couber na memória.
     */
    private fun maybeGrow(loss: Float) {
        emaLoss = if (emaLoss.isNaN()) loss else 0.98f * emaLoss + 0.02f * loss
        if (gpt.step - lastCheck < 300) return
        lastCheck = gpt.step
        val before = refLoss
        refLoss = emaLoss
        val plateau = !before.isNaN() && emaLoss > before * 0.98f
        if (plateau && emaLoss > 0.6f && !overfit) growNow()
    }

    // ------------------------------------------------------------------ erro em texto novo (validação)

    private val valBlock = 400

    /** Um bloco de cada 10 do corpus (posições fixas) nunca é treinado: serve para medir se o modelo só decorou. */
    private fun isVal(pos: Int): Boolean = (pos / valBlock) % 10 == 9

    /** Erro médio por caractere em texto que o modelo não treina (NaN = corpus ainda pequeno demais). */
    @Volatile var valLoss = Float.NaN
    @Volatile var bestVal = Float.NaN
    @Volatile var overfit = false
    @Volatile var quality = Float.NaN
    private var badEvals = 0
    private var overfitCorp = 0
    private val qualityLog = ArrayList<String>()

    /** Pares (pergunta, resposta) fixos para medir a qualidade; carregados de assets/avaliacao.txt. */
    @Volatile var evalPairs: List<Pair<String, String>> = emptyList()

    fun validate(): Float {
        val corp = corpusTokens()
        val t = gpt.cfg.t
        if (corp.size < 9 * valBlock + valBlock) return Float.NaN
        val none = IntArray(0)
        var tot = 0f
        var n = 0
        synchronized(lock) {
            var vb = 9 * valBlock
            while (vb + valBlock <= corp.size && n < 40) {
                var off = 0
                while (off + t + 1 <= valBlock && n < 40) {
                    val st = vb + off
                    val x = corp.copyOfRange(st, st + t)
                    val y = corp.copyOfRange(st + 1, st + t + 1)
                    tot += gpt.lossOf(x, y, none)
                    n++
                    off += t
                }
                vb += 10 * valBlock
            }
        }
        return if (n == 0) Float.NaN else tot / (n * t)
    }

    /** Nota de 0 a 100: probabilidade média (geométrica) que o modelo dá a cada letra das respostas certas. */
    fun measureQuality(): Float {
        val pairs = evalPairs
        if (pairs.isEmpty()) return Float.NaN
        val t = gpt.cfg.t
        val sys = sysLine()
        val none = IntArray(0)
        var lossSum = 0f
        var chars = 0
        synchronized(lock) {
            for ((q, a) in pairs) {
                val pre = Vocab.encode(sys + "@u " + q + "\n@a ")
                val ans = Vocab.encode(a + "\n")
                if (pre.isEmpty() || ans.isEmpty()) continue
                val ids = IntArray(1 + pre.size + ans.size)
                System.arraycopy(pre, 0, ids, 1, pre.size)
                System.arraycopy(ans, 0, ids, 1 + pre.size, ans.size)
                if (ids.size - 1 > t) continue
                val full = gpt.lossOf(ids.copyOfRange(0, ids.size - 1), ids.copyOfRange(1, ids.size), none)
                val pl = 1 + pre.size
                val pref = gpt.lossOf(ids.copyOfRange(0, pl - 1), ids.copyOfRange(1, pl), none)
                lossSum += full - pref
                chars += ans.size
            }
        }
        if (chars == 0) return Float.NaN
        val nota = 100f * exp(-lossSum / chars)
        quality = nota
        synchronized(qualityLog) {
            qualityLog.add(gpt.step.toString() + ": " + "%.0f".format(nota))
            if (qualityLog.size > 6) qualityLog.removeAt(0)
        }
        return nota
    }

    /** Mede agora (botão da aba Treino): erro em texto novo e qualidade nas perguntas fixas. */
    fun measureNow() {
        val v = validate()
        if (!v.isNaN()) {
            valLoss = v
            if (bestVal.isNaN() || v < bestVal) bestVal = v
        }
        measureQuality()
    }

    /** A cada 200 passos: mede o erro em texto novo; se piorar 2 vezes seguidas, a osmose para (está decorando). */
    private fun checkProgress() {
        val v = validate()
        if (!v.isNaN()) {
            valLoss = v
            if (bestVal.isNaN() || v < bestVal) {
                bestVal = v
                badEvals = 0
            } else if (v > bestVal * 1.02f) {
                badEvals++
                if (badEvals >= 2 && !overfit) {
                    overfit = true
                    overfitCorp = db.corpusChars()
                }
            } else {
                badEvals = 0
            }
        }
        measureQuality()
    }

    /** true = a osmose deve esperar. Volta sozinho quando chega texto novo (+10% no corpus). */
    fun overfitting(): Boolean {
        if (!overfit) return false
        if (db.corpusChars() > overfitCorp * 1.1f + 500f) {
            overfit = false
            badEvals = 0
            bestVal = Float.NaN
        }
        return overfit
    }

    fun healthInfo(): String {
        val sb = StringBuilder()
        sb.append("Erro em texto novo: ")
        if (valLoss.isNaN()) sb.append("— (precisa de mais texto)") else {
            sb.append("%.2f".format(valLoss))
            if (!bestVal.isNaN()) sb.append(" (melhor ").append("%.2f".format(bestVal)).append(")")
        }
        sb.append("\nQualidade nas perguntas fixas: ")
        if (quality.isNaN()) sb.append("—") else sb.append("%.0f".format(quality)).append("/100")
        val log = synchronized(qualityLog) { ArrayList(qualityLog) }
        if (log.size > 1) sb.append("  [").append(log.joinToString(" | ")).append("]")
        if (overfit) sb.append("\n⚠️ Está decorando: o erro em texto novo subiu. Treino automático em pausa até chegar mais texto.")
        return sb.toString()
    }

    /** Faz nascer especialistas agora (dobra). Devolve uma mensagem para mostrar. */
    fun growNow(): String {
        val before = gpt.lay.ex
        val added = synchronized(lock) {
            val a = gpt.grow()
            if (a > 0 && !frozen) gpt.save(db)
            a
        }
        refLoss = emaLoss
        return if (added > 0) "Os especialistas cresceram: " + before + " -> " + gpt.lay.ex + " por camada."
        else "Sem memória para crescer: " + gpt.growReport(before + maxOf(1, before))
    }

    private fun slotOf(id: Long): Int = (id % gpt.cfg.s).toInt()

    private fun pathSlots(byId: Map<Long, Node>, id: Long): IntArray {
        val out = ArrayList<Int>()
        var cur = byId[id]
        var guard = 0
        while (cur != null && guard < 8) {
            out.add(slotOf(cur.id))
            cur = byId[cur.parentId]
            guard++
        }
        return out.toIntArray()
    }

    private fun docs(): List<Doc> {
        val nodes = db.allNodes()
        val byId = HashMap<Long, Node>()
        for (n in nodes) byId[n.id] = n
        val out = ArrayList<Doc>()
        for (n in nodes) {
            val enc = Vocab.encode(n.content)
            if (enc.isEmpty()) continue
            val ids = IntArray(enc.size + 2)
            System.arraycopy(enc, 0, ids, 1, enc.size)
            out.add(Doc(n.id, pathSlots(byId, n.id), ids))
        }
        return out
    }

    fun corpusSize(): Int {
        var total = db.corpusChars()
        for (d in docs()) total += d.ids.size
        return total
    }

    @Volatile private var corpCache: IntArray? = null
    @Volatile private var lastDialogue = false

    /** Prompt de sistema (até 60 caracteres): vira a linha "@s ..." antes de cada conversa, no treino e na resposta. */
    @Volatile var sysPrompt: String = ""

    fun sysLine(): String {
        val t = sysPrompt.trim()
        return if (t.isEmpty()) "" else "@s " + t.take(60) + "\n"
    }

    fun invalidateCorpus() {
        corpCache = null
    }

    /**
     * Põe a linha "@s ..." antes de cada trecho de diálogo (sequência de linhas "@u"/"@a").
     * É feito só no fluxo de treino; o texto guardado no banco não muda, então trocar o prompt vale para tudo.
     */
    private fun applySys(text: String, start: Boolean): Pair<String, Boolean> {
        val sp = sysLine()
        val sb = StringBuilder(text.length + 256)
        var prev = start
        var i = 0
        val n = text.length
        while (i < n) {
            var e = text.indexOf('\n', i)
            if (e < 0) e = n
            val dlg = text.startsWith("@u ", i) || text.startsWith("@a ", i)
            if (dlg && !prev && sp.isNotEmpty()) sb.append(sp)
            sb.append(text, i, if (e < n) e + 1 else n)
            prev = dlg
            i = e + 1
        }
        return sb.toString() to prev
    }

    /** Texto do treino em massa: um documento por linha; "\n" (id 0) separa os documentos. */
    @Synchronized
    private fun corpusTokens(): IntArray {
        val c = corpCache
        if (c != null) return c
        val sys = applySys(db.corpusText(), false)
        lastDialogue = sys.second
        val enc = Vocab.encode(sys.first)
        val out = if (enc.isNotEmpty() && enc[0] != 0) {
            val a = IntArray(enc.size + 1)
            System.arraycopy(enc, 0, a, 1, enc.size)
            a
        } else enc
        corpCache = out
        return out
    }

    fun corpusChars(): Int = db.corpusChars()

    /** Acrescenta linhas ao corpus (cada bloco vira uma linha). Mantém o cache em memória sem reler tudo. */
    @Synchronized
    fun addCorpus(blocks: List<String>): Int {
        val sb = StringBuilder()
        for (b in blocks) sb.append(b).append('\n')
        val text = sb.toString()
        val n = db.addCorpus(text, 8_000_000)
        val c = corpCache
        if (n != text.length) {
            corpCache = null
        } else if (c != null) {
            val sys = applySys(text, lastDialogue)
            lastDialogue = sys.second
            val enc = Vocab.encode(sys.first)
            val out = c.copyOf(c.size + enc.size)
            System.arraycopy(enc, 0, out, c.size, enc.size)
            corpCache = out
        }
        return n
    }

    @Volatile var frozen = false

    fun saveNow() {
        if (frozen) return
        synchronized(lock) { if (!frozen) gpt.save(db) }
    }

    fun clearCorpus() {
        db.clearCorpus()
        corpCache = null
    }

    private fun sample(docs: List<Doc>, corp: IntArray, focus: Doc?, batch: Int, rnd: Random): List<Seq> {
        val t = gpt.cfg.t
        val hasCorp = corp.size > t + 2
        val out = ArrayList<Seq>()
        for (b in 0 until batch) {
            if ((focus == null || b % 2 == 1) && hasCorp && (docs.isEmpty() || rnd.nextBoolean())) {
                var start = -1
                var last = 0
                for (tries in 0 until 8) {
                    var s0 = rnd.nextInt(corp.size - t - 1)
                    // alinha ao começo de uma linha (id 0 = nova linha), para o diálogo começar em "@u" ou "@a"
                    var k = s0
                    val lim = minOf(s0 + 120, corp.size - t - 2)
                    while (k < lim && corp[k] != 0) k++
                    if (k < lim) s0 = k + 1
                    last = s0
                    // blocos de validação não entram no treino
                    if (!isVal(s0) && !isVal(s0 + t + 1)) {
                        start = s0
                        break
                    }
                }
                if (start < 0 && docs.isEmpty()) start = last
                if (start >= 0) {
                    out.add(Seq(IntArray(0), corp.copyOfRange(start, start + t), corp.copyOfRange(start + 1, start + t + 1)))
                    continue
                }
            }
            val d = if (focus != null && b % 2 == 0) focus else docs[rnd.nextInt(docs.size)]
            val len = d.ids.size
            val nn = min(t, len - 1)
            val start = rnd.nextInt(len - nn)
            out.add(Seq(d.slots, d.ids.copyOfRange(start, start + nn), d.ids.copyOfRange(start + 1, start + nn + 1)))
        }
        return out
    }

    /** Treina `steps` passos (nós da árvore + corpus). O modelo só fica travado durante cada passo. */
    fun train(steps: Int, batch: Int, focusId: Long, stop: AtomicBoolean?, saveAtEnd: Boolean = true, onStep: (Int, Float) -> Unit): Int {
        val ds = docs()
        val corp = corpusTokens()
        if (ds.isEmpty() && corp.size <= gpt.cfg.t + 2) return 0
        val focus = ds.firstOrNull { it.id == focusId }
        val rnd = Random(System.nanoTime())
        var i = 0
        while (i < steps && (stop == null || !stop.get())) {
            val seqs = sample(ds, corp, focus, batch, rnd)
            val lr = gpt.cfg.lr * minOf(1f, (gpt.step + 1) / 100f)
            val loss = synchronized(lock) { gpt.trainBatch(seqs, lr) }
            i++
            onStep(i, loss)
            maybeGrow(loss)
            if (gpt.step % 200 == 0) {
                try {
                    checkProgress()
                } catch (e: Exception) {
                }
            }
            if (i % 100 == 0 && !frozen) synchronized(lock) { gpt.save(db) }
        }
        if (saveAtEnd && !frozen) synchronized(lock) { gpt.save(db) }
        return i
    }

    /** Aprendizado em tempo real: logo depois de um nó novo, treina alguns passos em segundo plano. */
    fun learnInBackground(nodeId: Long, steps: Int = 8) {
        bg.execute {
            try {
                train(steps, 4, nodeId, null) { _, _ -> }
            } catch (e: Exception) {
            }
        }
    }

    /** Resposta de conversa: o prompt é o diálogo recente ("@u ...\n@a "); o galho da árvore condiciona o modelo. */
    fun generateChat(nodeId: Long, prompt: String, maxNew: Int = 90): String {
        val slots = if (nodeId > 0) {
            val byId = HashMap<Long, Node>()
            for (n in db.allNodes()) byId[n.id] = n
            pathSlots(byId, nodeId)
        } else IntArray(0)
        return synchronized(lock) { gpt.generate(slots, prompt, maxNew, 0.7f, Random(), true) }.trim()
    }

    fun generateFree(prompt: String, maxNew: Int): String =
        synchronized(lock) { gpt.generate(IntArray(0), prompt, maxNew, 0.8f, Random(), false) }

    fun reset() {
        synchronized(lock) {
            gpt.reset()
            db.clearModel()
        }
        valLoss = Float.NaN
        bestVal = Float.NaN
        overfit = false
        badEvals = 0
        quality = Float.NaN
        synchronized(qualityLog) { qualityLog.clear() }
    }
}

/** Gradiente calculado à mão contra gradiente numérico num modelo minúsculo. Devolve o pior erro relativo. */
private fun gradCheck(cfg: MiniGptConfig, active: IntArray, seed: Long): Float {
    val lay = Layout(cfg)
    val w = FloatArray(lay.n)
    val rnd = Random(seed)
    for (i in w.indices) w[i] = (rnd.nextGaussian() * 0.3).toFloat()
    val wk = Work(cfg, lay, active)
    val slots = intArrayOf(1, 3)
    var worst = 0f
    for (len in intArrayOf(6, 4)) {
        val x = IntArray(6) { rnd.nextInt(7) }
        val y = IntArray(6) { rnd.nextInt(7) }
        Arrays.fill(wk.g, 0f)
        wk.trainSeq(w, x, y, len, 1f, slots)
        val ana = wk.g.copyOf()
        for (k in 0 until 80) {
            val i = when {
                k < 20 -> lay.node + rnd.nextInt(cfg.s * cfg.c)
                k < 45 && cfg.ex > 1 -> lay.rr[rnd.nextInt(cfg.l)][rnd.nextInt(cfg.ex)] + rnd.nextInt(cfg.c)
                else -> rnd.nextInt(w.size)
            }
            val old = w[i]
            val eps = 1e-2f
            w[i] = old + eps
            val lp = wk.lossOf(w, x, y, len, slots, true)
            w[i] = old - eps
            val lm = wk.lossOf(w, x, y, len, slots, true)
            w[i] = old
            val num = (lp - lm) / (2f * eps)
            val err = abs(num - ana[i]) / max(0.05f, abs(num) + abs(ana[i]))
            if (err > worst) worst = err
        }
    }
    return worst
}

/** Cresce um modelo pequeno de 1 para 2 especialistas e mede quanto a perda muda (deve mudar pouco). */
private fun growCheck(): Float {
    val cfg = MiniGptConfig(v = 7, t = 8, c = 16, h = 2, l = 2, s = 5, maxThreads = 1)
    val g = MiniGpt(cfg)
    val rnd = Random(9)
    val w = g.w
    for (i in w.indices) w[i] = (rnd.nextGaussian() * 0.3).toFloat()
    val x = IntArray(8) { rnd.nextInt(7) }
    val y = IntArray(8) { rnd.nextInt(7) }
    val slots = intArrayOf(1, 3)
    val before = g.lossOf(x, y, slots)
    if (g.grow() != 1 || g.lay.ex != 2) return 1f
    val after = g.lossOf(x, y, slots)
    return abs(after - before) / max(0.05f, before)
}

/** Autoteste: modelo comum, modelo com especialistas (roteador incluído) e crescimento. */
fun miniSelfTest(): String {
    val a = gradCheck(MiniGptConfig(v = 7, t = 6, c = 8, h = 2, l = 2, s = 5), intArrayOf(1, 1), 3L)
    val b = gradCheck(MiniGptConfig(v = 7, t = 6, c = 8, h = 2, l = 2, s = 5, ex = 3), intArrayOf(2, 3), 5L)
    val gr = growCheck()
    val pa = "%.2f".format(a * 100f)
    val pb = "%.2f".format(b * 100f)
    val pg = "%.1f".format(gr * 100f)
    return if (a < 0.02f && b < 0.02f && gr < 0.15f)
        "Autoteste OK: gradiente $pa% (comum), $pb% (com especialistas); ao crescer a perda mudou $pg%."
    else "Autoteste FALHOU: gradiente $pa% (comum), $pb% (especialistas); crescimento mudou a perda $pg%. Mande este resultado ao Claude."
}
