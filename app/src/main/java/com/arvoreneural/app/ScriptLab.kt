package com.arvoreneural.app

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import org.mozilla.javascript.ScriptableObject
import org.mozilla.javascript.Context as Rhino

/** Ponte entre o JavaScript e o log do app (precisa ser pública para o Rhino enxergar os métodos). */
class JsLog(private val sb: StringBuilder) {
    fun log(m: String) {
        sb.append("☕ [KOTLIN LOG] ").append(m).append('\n')
    }
}

/**
 * Laboratório de scripts: Kotlin entrega números ao Python (NumPy, via Chaquopy), o Python diz se o
 * "neurônio" disparou, e o JavaScript (Rhino) transforma isso numa mensagem. Os dois scripts são texto
 * editável na aba Treino: mudou, tocou em Rodar, já vale (sem recompilar o app).
 */
object ScriptLab {
    val DEFAULT_PY = """# Entradas (nesta ordem): sol, fim de semana, dinheiro. O script enxerga np e entradas.
pesos = np.array([0.7, 0.5, 0.2])
vies = -0.5
soma_ponderada = float(np.dot(entradas, pesos) + vies)
disparou = bool(soma_ponderada > 0)
print("soma ponderada:", round(soma_ponderada, 3))"""

    val DEFAULT_JS = """// O Python entrega resultadoIA (true/false). A última linha é o retorno.
var mensagem = resultadoIA
    ? "🏖️ Partiu praia! O neurônio disparou."
    : "🏠 Melhor ficar em casa. O neurônio barrou.";
logDoKotlin("Processamento JS finalizado.");
mensagem;"""

    const val DEFAULT_IN = "1, 0, 1"

    fun parseInputs(s: String): DoubleArray =
        s.split(Regex("[,;\\s]+")).filter { it.isNotBlank() }.mapNotNull { it.toDoubleOrNull() }.toDoubleArray()

    /** Roda Python e depois JavaScript. Nunca lança: erros viram texto. */
    fun run(ctx: Context, pyCode: String, jsCode: String, inputs: DoubleArray): String {
        val out = StringBuilder()
        var disparou: Boolean
        try {
            if (!Python.isStarted()) Python.start(AndroidPlatform(ctx))
            val raw = Python.getInstance().getModule("runner").callAttr("rodar", pyCode, inputs).toString()
            val j = JSONObject(raw)
            if (!j.getBoolean("ok")) return "❌ Python:\n" + j.getString("erro")
            disparou = j.getBoolean("disparou")
            out.append("🐍 Python: disparou = ").append(disparou).append('\n')
            val so = j.getString("saida")
            if (so.isNotBlank()) out.append(so)
        } catch (e: Throwable) {
            return "❌ Não consegui rodar o Python: " + (e.message ?: e.javaClass.simpleName)
        }
        val logs = StringBuilder()
        val cx = Rhino.enter()
        try {
            cx.optimizationLevel = -1 // no Android não dá para gerar bytecode Java: modo interpretado
            cx.languageVersion = Rhino.VERSION_ES6
            val scope = cx.initStandardObjects()
            ScriptableObject.putProperty(scope, "resultadoIA", disparou)
            ScriptableObject.putProperty(scope, "kotlin", Rhino.javaToJS(JsLog(logs), scope))
            cx.evaluateString(scope, "function logDoKotlin(m) { kotlin.log(String(m)); }", "prelude", 1, null)
            val res = cx.evaluateString(scope, jsCode, "regras.js", 1, null)
            if (logs.isNotEmpty()) out.append(logs)
            out.append("🟨 JavaScript: ").append(Rhino.toString(res))
        } catch (e: Throwable) {
            out.append("❌ JavaScript: ").append(e.message ?: e.javaClass.simpleName)
        } finally {
            Rhino.exit()
        }
        return out.toString()
    }
}
