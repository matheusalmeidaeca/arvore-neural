package com.arvoreneural.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import androidx.core.content.ContextCompat
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Surface
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.ui.Alignment
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.platform.LocalView
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import java.util.concurrent.atomic.AtomicBoolean
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.arvoreneural.app.ai.Forecaster
import com.arvoreneural.app.ai.MiniLab
import com.arvoreneural.app.ai.Osmose
import com.arvoreneural.app.ai.Sizes
import com.arvoreneural.app.ai.miniSelfTest
import com.arvoreneural.app.data.Db
import com.arvoreneural.app.data.Node
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    private val notifPerm = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Engine.init(applicationContext)
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            notifPerm.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        Engine.ensureService(applicationContext)
        setContent { MaterialTheme { App() } }
        if (savedInstanceState == null) handleShare(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleShare(intent)
    }

    override fun onResume() {
        super.onResume()
        Engine.osmose.setForeground(true)
        Engine.ensureService(applicationContext)
    }

    override fun onPause() {
        super.onPause()
        Engine.osmose.setForeground(false)
    }

    /** Texto ou arquivo .txt compartilhado de outro app (WhatsApp, navegador…): o app absorve. */
    private fun handleShare(i: Intent?) {
        if (i == null || i.action != Intent.ACTION_SEND) return
        val text = i.getStringExtra(Intent.EXTRA_TEXT)
        @Suppress("DEPRECATION")
        val uri = i.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
        if (text == null && uri == null) return
        i.action = null
        val app = applicationContext
        Thread {
            val raw = text ?: try {
                app.contentResolver.openInputStream(uri!!)?.bufferedReader()?.readText() ?: ""
            } catch (e: Exception) {
                ""
            }
            val msg = Engine.osmose.absorb(raw)
            Engine.version++
            runOnUiThread { Toast.makeText(app, msg, Toast.LENGTH_LONG).show() }
        }.start()
    }
}

@Composable
fun App() {
    var tab by remember { mutableIntStateOf(0) }
    val labels = listOf("Conversa", "Treino", "Prever")
    val icons = listOf("💬", "🌊", "📈")
    Scaffold(
        bottomBar = {
            NavigationBar {
                labels.forEachIndexed { i, l ->
                    NavigationBarItem(
                        selected = tab == i,
                        onClick = { tab = i },
                        icon = { Text(icons[i]) },
                        label = { Text(l) }
                    )
                }
            }
        }
    ) { pad ->
        Box(Modifier.padding(pad).padding(16.dp).fillMaxSize()) {
            when (tab) {
                0 -> ChatScreen()
                1 -> TrainScreen { Engine.version++ }
                else -> ForecastScreen(Engine.db)
            }
        }
    }
}

class Msg(val fromUser: Boolean, val text: String)

@Composable
fun ChatScreen() {
    val msgs = Engine.msgs
    val listState = rememberLazyListState()
    LaunchedEffect(msgs.size, Engine.pending) {
        val n = msgs.size + (if (Engine.pending > 0) 1 else 0)
        if (n > 0) listState.animateScrollToItem(n - 1)
    }
    Column(Modifier.fillMaxSize()) {
        if (msgs.isNotEmpty()) {
            TextButton(onClick = { Engine.clearChat() }) { Text("Limpar conversa") }
        }
        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            state = listState,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (msgs.isEmpty()) {
                item {
                    Text("Converse comigo! Eu respondo só com o que está nos meus pesos (não existe arquivo de textos por trás). Aprendo com tudo que você escreve (osmose). Toque em 👍 nas respostas boas e em ✏️ para corrigir: a correção treina os pesos na hora e eu mostro quanto a chance da resposta certa subiu.")
                }
            }
            items(msgs.size) { idx ->
                val m = msgs[idx]
                Column(
                    Modifier.fillMaxWidth(),
                    horizontalAlignment = if (m.fromUser) Alignment.End else Alignment.Start
                ) {
                    Surface(
                        color = if (m.fromUser) MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.surfaceVariant,
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.widthIn(max = 300.dp)
                    ) { Text(m.text, Modifier.padding(10.dp)) }
                    if (!m.fromUser && idx > 0) {
                        Row {
                            TextButton(onClick = { Engine.like(idx) }) {
                                Text(if (idx in Engine.liked) "✓ guardado" else "👍 gostei")
                            }
                            TextButton(onClick = {
                                Engine.correctText = ""
                                Engine.correcting = idx
                            }) { Text("✏️ corrigir") }
                        }
                    }
                }
            }
            if (Engine.pending > 0) {
                item { Text("✍️ pensando…", style = MaterialTheme.typography.bodySmall) }
            }
        }
        if (Engine.correcting >= 0) {
            AlertDialog(
                onDismissRequest = { Engine.correcting = -1 },
                title = { Text("Qual seria a resposta certa?") },
                text = {
                    OutlinedTextField(
                        value = Engine.correctText,
                        onValueChange = { Engine.correctText = it },
                        placeholder = { Text("Resposta certa") }
                    )
                },
                confirmButton = {
                    TextButton(onClick = {
                        val t = Engine.correctText.trim()
                        if (t.isNotEmpty()) Engine.correct(Engine.correcting, t)
                        Engine.correcting = -1
                    }) { Text("Ensinar") }
                },
                dismissButton = { TextButton(onClick = { Engine.correcting = -1 }) { Text("Cancelar") } }
            )
        }
        if (Engine.notice.isNotEmpty()) {
            Surface(
                color = MaterialTheme.colorScheme.tertiaryContainer,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
            ) { Text(Engine.notice, Modifier.padding(10.dp), style = MaterialTheme.typography.bodySmall) }
        }
        Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = Engine.chatInput,
                onValueChange = { Engine.chatInput = it },
                placeholder = { Text("Mensagem") },
                modifier = Modifier.weight(1f)
            )
            Spacer(Modifier.width(8.dp))
            Button(onClick = {
                val t = Engine.chatInput.trim()
                if (t.isNotEmpty()) {
                    Engine.chatInput = ""
                    Engine.send(t)
                }
            }) { Text("Enviar") }
        }
    }
}

@Composable
fun TrainScreen(onChange: () -> Unit) {
    val ts = Engine.ts
    val lab = Engine.lab
    var confirmClear by remember { mutableStateOf(false) }
    var confirmReset by remember { mutableStateOf(false) }
    var confirmCorpus by remember { mutableStateOf(false) }
    var confirmPrune by remember { mutableStateOf(false) }
    val view = LocalView.current
    val training = ts.training
    DisposableEffect(training) {
        view.keepScreenOn = training
        onDispose { view.keepScreenOn = false }
    }
    Column(
        modifier = Modifier.verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            "Modelo: " + lab.gpt.lay.n + " parâmetros (" + lab.gpt.cfg.l + " camadas, " + lab.gpt.cfg.c + " dims, contexto " +
                lab.gpt.cfg.t + ", especialistas por camada: " + lab.expertsInfo() + "). O feed-forward de cada camada é uma mistura de especialistas que dobra de número quando o aprendizado empaca, sem teto fixo (o limite é a memória do celular). A osmose treina sozinha; aqui você alimenta e acelera.",
            style = MaterialTheme.typography.bodyMedium
        )
        OsmoseCard(Engine.osmose, onChange)

        var health by remember { mutableStateOf("") }
        LaunchedEffect(Unit) {
            while (true) {
                health = lab.healthInfo()
                delay(2000)
            }
        }
        Text("📊 Saúde do modelo", style = MaterialTheme.typography.titleMedium)
        Text(health, style = MaterialTheme.typography.bodySmall)
        if (Engine.selfTestMsg.isNotEmpty()) Text("🧪 " + Engine.selfTestMsg, style = MaterialTheme.typography.bodySmall)
        Button(enabled = !ts.busy, onClick = {
            Engine.scope.launch(Dispatchers.Default) {
                lab.measureNow()
                health = lab.healthInfo()
            }
        }) { Text("📏 Medir agora") }

        OutlinedTextField(
            value = Engine.learnText,
            onValueChange = { Engine.learnText = it },
            label = { Text("Colar um texto para a osmose absorver") },
            modifier = Modifier.fillMaxWidth().heightIn(min = 120.dp)
        )
        Button(onClick = {
            val t = Engine.learnText
            Engine.learnText = ""
            Engine.osmoseMsg = "Absorvendo…"
            Engine.scope.launch(Dispatchers.IO) {
                Engine.osmoseMsg = try {
                    Engine.osmose.absorb(t)
                } catch (e: Throwable) {
                    "Erro ao absorver."
                }
                onChange()
            }
        }) { Text("Absorver texto") }

        Text("🏋️ Treino manual", style = MaterialTheme.typography.titleMedium)
        Text(ts.status, style = MaterialTheme.typography.titleSmall)
        Text(ts.corpusInfo, style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = !ts.busy, onClick = { Engine.startTrain(300) }) { Text("Treinar 300") }
            Button(enabled = !ts.busy, onClick = { Engine.startTrain(2000) }) { Text("Treinar 2000") }
            Button(enabled = ts.training, onClick = { ts.stop.set(true) }) { Text("Parar") }
        }
        Button(enabled = !ts.busy, onClick = {
            Engine.scope.launch(Dispatchers.Default) { ts.status = lab.growNow() }
        }) { Text("🌱 Crescer agora (dobra os especialistas)") }


        SizeCard()
        BackupCard()

        Text("🐍 Scripts: Python + JavaScript", style = MaterialTheme.typography.titleMedium)
        Text(
            "O Kotlin entrega os números ao Python (NumPy); o Python diz se o neurônio disparou (variável disparou); " +
                "o JavaScript recebe resultadoIA e devolve a mensagem. Edite, toque em Rodar e vale na hora.",
            style = MaterialTheme.typography.bodySmall
        )
        OutlinedTextField(
            value = Engine.scriptIn,
            onValueChange = { Engine.scriptIn = it },
            label = { Text("Entradas (números separados por vírgula)") },
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedTextField(
            value = Engine.scriptPy,
            onValueChange = { Engine.scriptPy = it },
            label = { Text("Python (NumPy)") },
            modifier = Modifier.fillMaxWidth().heightIn(min = 140.dp)
        )
        OutlinedTextField(
            value = Engine.scriptJs,
            onValueChange = { Engine.scriptJs = it },
            label = { Text("JavaScript") },
            modifier = Modifier.fillMaxWidth().heightIn(min = 120.dp)
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = !Engine.scriptBusy, onClick = { Engine.runScripts() }) { Text("▶ Rodar") }
            TextButton(enabled = !Engine.scriptBusy, onClick = {
                Engine.scriptPy = ScriptLab.DEFAULT_PY
                Engine.scriptJs = ScriptLab.DEFAULT_JS
                Engine.scriptIn = ScriptLab.DEFAULT_IN
                Engine.saveScripts()
            }) { Text("Restaurar exemplo") }
        }
        if (Engine.scriptOut.isNotEmpty()) Text(Engine.scriptOut, style = MaterialTheme.typography.bodySmall)

        Text("🧪 Avançado", style = MaterialTheme.typography.titleMedium)
        OutlinedTextField(
            value = ts.prompt,
            onValueChange = { ts.prompt = it },
            label = { Text("Começo do texto (opcional)") },
            modifier = Modifier.fillMaxWidth()
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = !ts.busy, onClick = {
                val p = ts.prompt
                ts.busy = true
                ts.output = "Gerando…"
                Engine.scope.launch(Dispatchers.Default) {
                    ts.output = try {
                        p + lab.generateFree(p, 200)
                    } catch (e: Throwable) {
                        "Erro ao gerar: " + (e.message ?: "")
                    }
                    ts.busy = false
                }
            }) { Text("Gerar texto") }
            Button(enabled = !ts.busy, onClick = {
                ts.busy = true
                ts.output = "Testando…"
                Engine.scope.launch(Dispatchers.Default) {
                    ts.output = try {
                        miniSelfTest()
                    } catch (e: Throwable) {
                        "Erro no autoteste: " + (e.message ?: "")
                    }
                    ts.busy = false
                }
            }) { Text("Autoteste") }
        }
        Text(ts.output)
        TextButton(enabled = !ts.busy, onClick = {
            if (!confirmCorpus) {
                confirmCorpus = true
            } else {
                confirmCorpus = false
                Engine.scope.launch(Dispatchers.IO) {
                    lab.clearCorpus()
                    ts.corpusInfo = "Corpus de treino: 0 caracteres"
                }
            }
        }) { Text(if (confirmCorpus) "Toque de novo para confirmar" else "Apagar corpus de treino") }
        TextButton(enabled = !ts.busy, onClick = {
            if (!confirmReset) {
                confirmReset = true
            } else {
                confirmReset = false
                Engine.scope.launch(Dispatchers.IO) {
                    lab.reset()
                    ts.status = "Modelo reiniciado do zero."
                    ts.output = ""
                }
            }
        }) { Text(if (confirmReset) "Toque de novo para confirmar" else "Reiniciar modelo (pesos)") }
        TextButton(onClick = {
            if (!confirmClear) {
                confirmClear = true
            } else {
                confirmClear = false
                Engine.scope.launch(Dispatchers.IO) {
                    Engine.db.clearNodes()
                    Engine.osmoseMsg = "Nós antigos apagados"
                    onChange()
                }
            }
        }) { Text(if (confirmClear) "Toque de novo para confirmar" else "Apagar nós antigos (a árvore não é mais usada)") }
    }
}

@Composable
fun OsmoseCard(osmose: Osmose, onChange: () -> Unit) {
    val ctx = LocalContext.current
    var on by remember { mutableStateOf(osmose.enabled) }
    var name by remember { mutableStateOf(osmose.myName) }
    var sysText by remember { mutableStateOf(osmose.systemPrompt) }
    var imita by remember { mutableStateOf(osmose.aiPlaysMe) }
    var detail by remember { mutableStateOf(osmose.detail) }
    var status by remember { mutableStateOf("") }
    LaunchedEffect(Unit) {
        while (true) {
            status = withContext(Dispatchers.IO) { osmose.status() }
            on = osmose.enabled
            delay(2000)
        }
    }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            val app = ctx.applicationContext
            Engine.osmoseMsg = "Absorvendo…"
            Engine.scope.launch(Dispatchers.IO) {
                Engine.osmoseMsg = try {
                    val raw = app.contentResolver.openInputStream(uri)?.bufferedReader()?.readText() ?: ""
                    osmose.absorb(raw)
                } catch (e: Throwable) {
                    "Erro ao ler o arquivo."
                }
                onChange()
            }
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("🌊 Osmose", style = MaterialTheme.typography.titleMedium)
        Text(status, style = MaterialTheme.typography.bodySmall)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(checked = on, onCheckedChange = {
                on = it
                osmose.enabled = it
                if (it) Engine.ensureService(ctx.applicationContext)
            })
            Spacer(Modifier.width(8.dp))
            Text("Aprender sozinho (também em segundo plano, com notificação fixa)")
        }
        OutlinedTextField(
            value = name,
            onValueChange = { name = it; osmose.myName = it },
            label = { Text("Seu nome nas conversas do WhatsApp") },
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedTextField(
            value = sysText,
            onValueChange = { sysText = it.take(60) },
            label = { Text("Prompt de sistema (até 60 letras)") },
            modifier = Modifier.fillMaxWidth()
        )
        Button(onClick = {
            osmose.systemPrompt = sysText
            Engine.osmoseMsg = "Prompt salvo. Vale para o treino daqui para frente e para as respostas."
        }) { Text("Salvar prompt") }
        Text(
            "O prompt entra antes de cada conversa, no treino e na resposta. Um modelo pequeno não obedece ordens: ele aprende o estilo dos diálogos que viu com esse prompt. Troque só de vez em quando; vazio desliga.",
            style = MaterialTheme.typography.bodySmall
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(checked = imita, onCheckedChange = { imita = it; osmose.aiPlaysMe = it })
            Spacer(Modifier.width(8.dp))
            Text("A IA fala como eu (em vez da outra pessoa). Vale para as próximas importações.")
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(checked = detail, onCheckedChange = { detail = it; osmose.detail = it })
            Spacer(Modifier.width(8.dp))
            Text("Mostrar detalhes nas respostas")
        }
        Button(onClick = { picker.launch(arrayOf("text/*")) }) { Text("Absorver conversa ou texto") }
        Text(Engine.osmoseMsg)
        Button(onClick = {
            val i = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:" + ctx.packageName))
            ctx.startActivity(i)
        }) { Text("Abrir ajustes do app (Bateria › Sem restrições)") }
        Text(
            "Dica: no WhatsApp, abra uma conversa › Exportar conversa › Árvore Neural. Qualquer texto compartilhado com o app também é absorvido. Para treinar com a tela apagada, deixe a bateria do app em Sem restrições.",
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
fun BackupCard() {
    val ctx = LocalContext.current
    var confirmRestore by remember { mutableStateOf(false) }
    val save = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri: Uri? ->
        if (uri != null) {
            Engine.backupMsg = "Salvando…"
            Engine.scope.launch(Dispatchers.IO) { Engine.backupMsg = Engine.backupTo(uri) }
        }
    }
    val open = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            Engine.backupMsg = "Restaurando…"
            Engine.scope.launch(Dispatchers.IO) {
                val r = Engine.restoreFrom(uri)
                Engine.backupMsg = r.second
                if (r.first) {
                    delay(2500)
                    android.os.Process.killProcess(android.os.Process.myPid())
                }
            }
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("💾 Backup", style = MaterialTheme.typography.titleMedium)
        Text(
            "Guarda num arquivo tudo que o app aprendeu (pesos, corpus, conversa). Faça antes de desinstalar ou trocar de celular.",
            style = MaterialTheme.typography.bodySmall
        )
        Button(onClick = { save.launch("arvore-backup.db") }) { Text("Salvar backup") }
        Button(onClick = {
            if (!confirmRestore) {
                confirmRestore = true
                Engine.backupMsg = "Restaurar substitui TUDO que há no app agora. Toque de novo para escolher o arquivo."
            } else {
                confirmRestore = false
                open.launch(arrayOf("*/*"))
            }
        }) { Text(if (confirmRestore) "Toque de novo para confirmar" else "Restaurar backup") }
        Text(Engine.backupMsg)
    }
}

@Composable
fun ForecastScreen(db: Db) {
    var name by remember { mutableStateOf("gastos") }
    var value by remember { mutableStateOf("") }
    var points by remember { mutableStateOf(listOf<Double>()) }
    var tick by remember { mutableIntStateOf(0) }
    val scope = rememberCoroutineScope()
    LaunchedEffect(name, tick) {
        points = withContext(Dispatchers.IO) { db.points(name.trim()).map { it.value } }
    }
    val pred = Forecaster.predict(points)
    var moe by remember { mutableStateOf<Pair<Double, Int>?>(null) }
    LaunchedEffect(points) {
        moe = withContext(Dispatchers.Default) { Forecaster.predictMoe(points) }
    }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        OutlinedTextField(
            value = name,
            onValueChange = { name = it },
            label = { Text("Série (ex.: gastos)") },
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedTextField(
            value = value,
            onValueChange = { value = it },
            label = { Text("Novo valor") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
            modifier = Modifier.fillMaxWidth()
        )
        Button(onClick = {
            val v = value.replace(",", ".").toDoubleOrNull()
            if (v != null) {
                val s = name.trim()
                scope.launch {
                    withContext(Dispatchers.IO) { db.insertPoint(s, v) }
                    value = ""
                    tick++
                }
            }
        }) { Text("Adicionar e reaprender") }
        Text("${points.size} valores: " + points.takeLast(6).joinToString(", "))
        Text(
            if (pred != null) "Previsão do próximo: " + "%.2f".format(pred)
            else "Preciso de ao menos 2 valores para prever.",
            style = MaterialTheme.typography.titleMedium
        )
        val m = moe
        Text(
            if (m != null) "🌱 Segunda opinião (especialistas que crescem, experimental): " + "%.2f".format(m.first) +
                " com " + m.second + " especialistas. Compare com a tendência acima."
            else "🌱 A segunda opinião (experimental) precisa de 14 valores positivos.",
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
fun SizeCard() {
    var pick by remember { mutableStateOf(-1) }
    var msg by remember { mutableStateOf("") }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("📏 Tamanho do modelo", style = MaterialTheme.typography.titleMedium)
        Text(
            "Trocar o tamanho recomeça o treino do modelo (corpus e conversa ficam). Modelos maiores treinam bem mais devagar e precisam de mais texto. Faça um backup antes (logo abaixo).",
            style = MaterialTheme.typography.bodySmall
        )
        for (i in Sizes.names.indices) {
            val cur = i == Engine.size
            Button(enabled = !cur, onClick = {
                if (pick != i) {
                    pick = i
                } else {
                    pick = -1
                    msg = "Trocando…"
                    Engine.scope.launch(Dispatchers.IO) {
                        msg = Engine.changeSize(i)
                        delay(2500)
                        android.os.Process.killProcess(android.os.Process.myPid())
                    }
                }
            }) {
                Text(Sizes.names[i] + (if (cur) " (atual)" else if (pick == i) ": toque de novo para confirmar" else ""))
            }
            Text(Sizes.info[i], style = MaterialTheme.typography.bodySmall)
        }
        if (msg.isNotEmpty()) Text(msg)
    }
}
