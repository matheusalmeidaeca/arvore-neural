package com.arvoreneural.app

import android.content.Context
import com.arvoreneural.app.data.Db
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/**
 * v11.2: o Engine ficou mínimo. O mini transformer (Sayoto), a osmose e o treino foram removidos
 * para não competir por memória com o Cérebro (GGUF/llama.cpp). Sobrou o banco (aba Prever)
 * e o escopo de corrotinas usado pelo Cérebro.
 */
object Engine {
    lateinit var db: Db
    @Volatile var ready = false
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    @Synchronized
    fun init(ctx: Context) {
        if (ready) return
        db = Db(ctx.applicationContext)
        ready = true
    }
}
