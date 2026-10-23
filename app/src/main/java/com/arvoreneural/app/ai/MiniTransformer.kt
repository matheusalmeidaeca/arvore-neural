package com.arvoreneural.app.ai

import com.arvoreneural.app.data.Db
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
 * Tamanho igual ao mini.py: T=64, C=128, H=4, L=4. Os pesos aprendidos ficam no banco de dados.
 */

class MiniGptConfig(val v: Int, val t: Int = 64, val c: Int = 128, val h: Int = 4, val l: Int = 4)

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

class Layout(cfg: MiniGptConfig) {
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
class Work(cfg: MiniGptConfig, private val lay: Layout) {
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

    fun forward(w: FloatArray, x: IntArray, nn: Int) {
        for (t in 0 until nn) {
            val ro = lay.tok + x[t] * C
            val po = lay.pos + t * C
            for (i in 0 until C) hs[0][t * C + i] = w[ro + i] + w[po + i]
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
            linFwd(bn[l], nn, C, w, lay.w1[l], lay.b1[l], 4 * C, f1[l])
            val f = f1[l]
            val gg = gl[l]
            for (i in 0 until nn * 4 * C) gg[i] = gelu(f[i])
            linFwd(gg, nn, 4 * C, w, lay.w2[l], lay.b2[l], C, tmp)
            val hn = hs[l + 1]
            for (i in 0 until nn * C) hn[i] = h1l[i] + tmp[i]
        }
        lnFwd(hs[L], nn, C, w, lay.lnfg, lay.lnfb, lnfx, lnfrs, z)
        linFwd(z, nn, C, w, lay.wout, lay.bout, V, logits)
    }

    /** Soma da entropia cruzada dos nn tokens (sem dividir). */
    fun lossOf(w: FloatArray, x: IntArray, y: IntArray, nn: Int): Float {
        forward(w, x, nn)
        var tot = 0f
        for (t in 0 until nn) {
            val r = t * V
            var mx = logits[r]
            for (j in 1 until V) if (logits[r + j] > mx) mx = logits[r + j]
            var s = 0f
            for (j in 0 until V) s += exp(logits[r + j] - mx)
            tot += -(logits[r + y[t]] - mx - ln(s))
        }
        return tot
    }

    /** Forward + backward. Acumula o gradiente (multiplicado por scale) em g. Devolve a soma da perda. */
    fun trainSeq(w: FloatArray, x: IntArray, y: IntArray, nn: Int, scale: Float): Float {
        forward(w, x, nn)
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
            Arrays.fill(dgl, 0f)
            linBwd(gl[l], dh, nn, 4 * C, w, lay.w2[l], lay.b2[l], C, g, dgl)
            val f = f1[l]
            for (i in 0 until nn * 4 * C) dgl[i] *= geluGrad(f[i])
            Arrays.fill(dbn, 0f)
            linBwd(bn[l], dgl, nn, C, w, lay.w1[l], lay.b1[l], 4 * C, g, dbn)
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
        return tot
    }
}

class MiniGpt(val cfg: MiniGptConfig) {
    val lay = Layout(cfg)
    val w = FloatArray(lay.n)
    private val am = FloatArray(lay.n)
    private val av = FloatArray(lay.n)
    private var adamT = 0

    @Volatile var step = 0
    @Volatile var lastLoss = Float.NaN

    private val nThreads = max(1, min(4, Runtime.getRuntime().availableProcessors()))
    private val pool = Executors.newFixedThreadPool(nThreads) { r -> Thread(r).also { it.isDaemon = true } }
    private val works = Array(nThreads) { Work(cfg, lay) }

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
    }

    fun reset() = initWeights(Random(System.nanoTime()))

    /** Um passo de treino (AdamW, como no mini.py). Devolve a perda média por caractere. */
    fun trainStep(data: IntArray, batch: Int, lr: Float, rnd: Random): Float {
        val T = cfg.t
        require(data.size > T + 1) { "texto curto demais" }
        val starts = IntArray(batch) { rnd.nextInt(data.size - T - 1) }
        for (wk in works) Arrays.fill(wk.g, 0f)
        val losses = FloatArray(nThreads)
        val scale = 1f / (T * batch)
        val tasks = (0 until nThreads).map { ti ->
            Callable<Unit> {
                val wk = works[ti]
                var loss = 0f
                var b = ti
                while (b < batch) {
                    val s = starts[b]
                    val x = data.copyOfRange(s, s + T)
                    val y = data.copyOfRange(s + 1, s + T + 1)
                    loss += wk.trainSeq(w, x, y, T, scale)
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
        val mean = total / (T * batch)
        step++
        lastLoss = mean
        return mean
    }

    fun generate(prompt: String, maxNew: Int, temp: Float, rnd: Random): String {
        val T = cfg.t
        val V = cfg.v
        val ids = ArrayList<Int>()
        val p = Vocab.encode(prompt)
        if (p.isEmpty()) {
            ids.add(0)
        } else {
            for (x in p) ids.add(x)
        }
        val wk = works[0]
        val out = StringBuilder()
        val probs = FloatArray(V)
        for (k in 0 until maxNew) {
            val nn = min(ids.size, T)
            val ctx = IntArray(nn) { ids[ids.size - nn + it] }
            wk.forward(w, ctx, nn)
            val off = (nn - 1) * V
            var mx = wk.logits[off]
            for (j in 1 until V) if (wk.logits[off + j] > mx) mx = wk.logits[off + j]
            var s = 0f
            for (j in 0 until V) {
                val e = exp((wk.logits[off + j] - mx) / temp)
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
            ids.add(pick)
            out.append(Vocab.chars[pick])
        }
        return out.toString()
    }

    private fun meta(): ByteArray {
        val bb = ByteBuffer.allocate(9 * 4)
        bb.putInt(1); bb.putInt(step); bb.putFloat(lastLoss); bb.putInt(lay.n)
        bb.putInt(cfg.v); bb.putInt(cfg.t); bb.putInt(cfg.c); bb.putInt(cfg.h); bb.putInt(cfg.l)
        return bb.array()
    }

    fun save(db: Db) {
        db.saveModel(meta(), w)
    }

    /** Carrega os pesos salvos no banco; devolve false se não houver ou se o tamanho não bater. */
    fun load(db: Db): Boolean {
        val mb = db.loadModelMeta() ?: return false
        if (mb.size < 36) return false
        val bb = ByteBuffer.wrap(mb)
        bb.getInt()
        val st = bb.getInt()
        val ls = bb.getFloat()
        val n = bb.getInt()
        val v = bb.getInt()
        val t = bb.getInt()
        val c = bb.getInt()
        val h = bb.getInt()
        val l = bb.getInt()
        if (n != lay.n || v != cfg.v || t != cfg.t || c != cfg.c || h != cfg.h || l != cfg.l) return false
        if (!db.loadModelWeights(w)) {
            initWeights(Random(1))
            return false
        }
        step = st
        lastLoss = ls
        return true
    }
}

/** Liga o transformer ao banco: texto de treino = tudo que a árvore aprendeu. */
class MiniLab(private val db: Db) {
    val gpt = MiniGpt(MiniGptConfig(v = Vocab.size))

    init {
        gpt.load(db)
    }

    fun corpus(): IntArray = Vocab.encode(db.allNodes().joinToString("\n") { it.content })

    fun save() = gpt.save(db)

    fun reset() {
        gpt.reset()
        db.clearModel()
    }
}

/** Compara o gradiente calculado à mão com o gradiente numérico, num modelo minúsculo. */
fun miniSelfTest(): String {
    val cfg = MiniGptConfig(v = 7, t = 6, c = 8, h = 2, l = 2)
    val lay = Layout(cfg)
    val w = FloatArray(lay.n)
    val rnd = Random(3)
    for (i in w.indices) w[i] = (rnd.nextGaussian() * 0.3).toFloat()
    val wk = Work(cfg, lay)
    var worst = 0f
    for (len in intArrayOf(6, 4)) {
        val x = IntArray(6) { rnd.nextInt(7) }
        val y = IntArray(6) { rnd.nextInt(7) }
        Arrays.fill(wk.g, 0f)
        wk.trainSeq(w, x, y, len, 1f)
        val ana = wk.g.copyOf()
        for (k in 0 until 40) {
            val i = rnd.nextInt(w.size)
            val old = w[i]
            val eps = 1e-2f
            w[i] = old + eps
            val lp = wk.lossOf(w, x, y, len)
            w[i] = old - eps
            val lm = wk.lossOf(w, x, y, len)
            w[i] = old
            val num = (lp - lm) / (2f * eps)
            val err = abs(num - ana[i]) / max(0.05f, abs(num) + abs(ana[i]))
            if (err > worst) worst = err
        }
    }
    val pct = "%.2f".format(worst * 100f)
    return if (worst < 0.02f) "Autoteste OK: erro máximo do gradiente $pct% (modelo mini de ${lay.n} parâmetros)."
    else "Autoteste FALHOU: erro máximo $pct%. Mande este resultado ao Claude."
}
