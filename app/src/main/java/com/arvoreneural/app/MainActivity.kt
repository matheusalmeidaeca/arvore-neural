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
import com.arvoreneural.app.ai.HashEmbedder
import com.arvoreneural.app.ai.MiniLab
import com.arvoreneural.app.ai.Osmose
import com.arvoreneural.app.ai.miniSelfTest
import com.arvoreneural.app.ai.Tree
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
    val labels = listOf("Conversa", "Árvore", "Aprender", "Prever", "Transformer")
    val icons = listOf("💬", "🌳", "✍️", "📈", "🧠")
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
                1 -> TreeScreen(Engine.db, Engine.tree, Engine.version) { Engine.version++ }
                2 -> LearnScreen(Engine.tree, Engine.db, Engine.osmose) { Engine.version++ }
                3 -> ForecastScreen(Engine.db)
                else -> TransformerScreen(Engine.lab, Engine.tree, Engine.osmose)
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
                    Text("Converse comigo! Eu respondo com o que já aprendi e continuo aprendendo com tudo que você escreve (osmose). Toque em 👍 nas respostas boas.")
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
                        TextButton(onClick = { Engine.like(idx) }) {
                            Text(if (idx in Engine.liked) "✓ guardado" else "👍 gostei")
                        }
                    }
                }
            }
            if (Engine.pending > 0) {
                item { Text("✍️ pensando…", style = MaterialTheme.typography.bodySmall) }
            }
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

fun flatten(nodes: List<Node>): List<Pair<Node, Int>> {
    val byParent = nodes.groupBy { it.parentId }
    val out = mutableListOf<Pair<Node, Int>>()
    fun walk(pid: Long, d: Int) {
        byParent[pid]?.forEach {
            out.add(it to d)
            walk(it.id, d + 1)
        }
    }
    walk(0L, 0)
    return out
}

@Composable
fun TreeScreen(db: Db, tree: Tree, version: Int, onChange: () -> Unit) {
    val ctx = LocalContext.current
    var nodes by remember { mutableStateOf(listOf<Node>()) }
    var q by remember { mutableStateOf("") }
    var answers by remember { mutableStateOf(listOf<Pair<Node, Float>>()) }
    var editing by remember { mutableStateOf<Node?>(null) }
    var editText by remember { mutableStateOf("") }
    var addParent by remember { mutableStateOf(-1L) }
    var addText by remember { mutableStateOf("") }
    var showImport by remember { mutableStateOf(false) }
    var impText by remember { mutableStateOf("") }
    var impPaths by remember { mutableStateOf(true) }
    val scope = rememberCoroutineScope()
    LaunchedEffect(version) { nodes = withContext(Dispatchers.IO) { db.allNodes() } }
    val flat = remember(nodes) { flatten(nodes) }
    val filePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            val app = ctx.applicationContext
            Engine.scope.launch(Dispatchers.IO) {
                impText = try {
                    app.contentResolver.openInputStream(uri)?.bufferedReader()?.readText() ?: ""
                } catch (e: Throwable) {
                    ""
                }
            }
        }
    }

    val ed = editing
    if (ed != null) {
        AlertDialog(
            onDismissRequest = { editing = null },
            title = { Text("Editar nó") },
            text = {
                OutlinedTextField(
                    value = editText,
                    onValueChange = { editText = it },
                    modifier = Modifier.fillMaxWidth().heightIn(min = 100.dp)
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    val id = ed.id
                    val t = editText
                    editing = null
                    Engine.scope.launch(Dispatchers.IO) {
                        tree.editNode(id, t)
                        Engine.treeMsg = "Nó editado."
                        onChange()
                    }
                }) { Text("Salvar") }
            },
            dismissButton = { TextButton(onClick = { editing = null }) { Text("Cancelar") } }
        )
    }
    if (addParent >= 0L) {
        AlertDialog(
            onDismissRequest = { addParent = -1L },
            title = { Text(if (addParent == 0L) "Novo nó raiz" else "Novo nó filho") },
            text = {
                OutlinedTextField(
                    value = addText,
                    onValueChange = { addText = it },
                    modifier = Modifier.fillMaxWidth().heightIn(min = 100.dp)
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    val pid = addParent
                    val t = addText
                    addParent = -1L
                    addText = ""
                    Engine.scope.launch(Dispatchers.IO) {
                        val id = tree.addNode(pid, t)
                        Engine.treeMsg = if (id > 0) "Nó criado." else "Escreva algo para criar o nó."
                        onChange()
                    }
                }) { Text("Criar") }
            },
            dismissButton = { TextButton(onClick = { addParent = -1L }) { Text("Cancelar") } }
        )
    }
    if (showImport) {
        AlertDialog(
            onDismissRequest = { showImport = false },
            title = { Text("Importar lista indentada") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "Um nó por linha. Cada _ no começo é um nível: B, _Br, __Bra. Cole aqui ou escolha um .txt.",
                        style = MaterialTheme.typography.bodySmall
                    )
                    OutlinedTextField(
                        value = impText,
                        onValueChange = { impText = it },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 160.dp)
                    )
                    TextButton(onClick = { filePicker.launch(arrayOf("text/*")) }) { Text("Escolher arquivo .txt") }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Switch(checked = impPaths, onCheckedChange = { impPaths = it })
                        Spacer(Modifier.width(8.dp))
                        Text("Treinar também com os caminhos (B > Br > Bra)", style = MaterialTheme.typography.bodySmall)
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    val raw = impText
                    val withPaths = impPaths
                    showImport = false
                    impText = ""
                    Engine.treeMsg = "Importando…"
                    Engine.scope.launch(Dispatchers.IO) {
                        Engine.treeMsg = try {
                            tree.importOutline(raw, withPaths)
                        } catch (e: Throwable) {
                            "Erro ao importar: " + (e.message ?: "")
                        }
                        onChange()
                    }
                }) { Text("Importar") }
            },
            dismissButton = { TextButton(onClick = { showImport = false }) { Text("Cancelar") } }
        )
    }

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = q,
            onValueChange = { q = it },
            label = { Text("Pergunte à árvore") },
            modifier = Modifier.fillMaxWidth()
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                scope.launch { answers = withContext(Dispatchers.IO) { tree.ask(q) } }
            }) { Text("Buscar") }
            Button(onClick = { addText = ""; addParent = 0L }) { Text("+ Raiz") }
            Button(onClick = { showImport = true }) { Text("Importar lista") }
        }
        if (Engine.treeMsg.isNotEmpty()) Text(Engine.treeMsg, style = MaterialTheme.typography.bodySmall)
        if (answers.isNotEmpty()) {
            Text("Mais parecidos:", style = MaterialTheme.typography.titleSmall)
            answers.forEach { (n, s) -> Text("• ${n.content.take(140)} (${(s * 100).toInt()}%)") }
        }
        Text("${nodes.size} nós na árvore. Toque no texto para editar, + para filho, ✕ para apagar.", style = MaterialTheme.typography.titleSmall)
        LazyColumn {
            items(flat) { (n, d) ->
                Row(
                    Modifier.fillMaxWidth().padding(start = (minOf(d, 10) * 16).dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(
                        onClick = { editText = n.content; editing = n },
                        modifier = Modifier.weight(1f)
                    ) { Text(n.title, modifier = Modifier.fillMaxWidth()) }
                    TextButton(onClick = { addText = ""; addParent = n.id }) { Text("+") }
                    TextButton(onClick = {
                        scope.launch {
                            withContext(Dispatchers.IO) { db.deleteNode(n.id) }
                            onChange()
                        }
                    }) { Text("✕") }
                }
            }
        }
    }
}

@Composable
fun LearnScreen(tree: Tree, db: Db, osmose: Osmose, onChange: () -> Unit) {
    val ctx = LocalContext.current
    var confirmClear by remember { mutableStateOf(false) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            val from = Engine.learnFrom
            val app = ctx.applicationContext
            Engine.learnMsg = "Aprendendo…"
            Engine.scope.launch(Dispatchers.IO) {
                val n = try {
                    val raw = app.contentResolver.openInputStream(uri)?.bufferedReader()?.readText() ?: ""
                    tree.learnFile(raw, from)
                } catch (e: Throwable) {
                    0
                }
                Engine.learnMsg = "$n trechos aprendidos"
                onChange()
            }
        }
    }
    Column(
        modifier = Modifier.verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        OsmoseCard(osmose, onChange)
        BackupCard()
        OutlinedTextField(
            value = Engine.learnText,
            onValueChange = { Engine.learnText = it },
            label = { Text("Ensine algo à árvore") },
            modifier = Modifier.fillMaxWidth().heightIn(min = 140.dp)
        )
        Button(onClick = {
            val t = Engine.learnText
            Engine.learnText = ""
            Engine.learnMsg = "Aprendendo…"
            Engine.scope.launch(Dispatchers.IO) {
                try {
                    tree.learn(t)
                    Engine.learnMsg = "Aprendido!"
                } catch (e: Throwable) {
                    Engine.learnMsg = "Erro ao aprender."
                }
                onChange()
            }
        }) { Text("Aprender") }
        OutlinedTextField(
            value = Engine.learnFrom,
            onValueChange = { Engine.learnFrom = it },
            label = { Text("WhatsApp: só mensagens de (nome, opcional)") },
            modifier = Modifier.fillMaxWidth()
        )
        Button(onClick = { picker.launch(arrayOf("text/*")) }) { Text("Importar arquivo .txt") }
        Text(Engine.learnMsg)
        TextButton(onClick = {
            if (!confirmClear) {
                confirmClear = true
            } else {
                confirmClear = false
                Engine.scope.launch(Dispatchers.IO) {
                    db.clearNodes()
                    Engine.learnMsg = "Árvore apagada"
                    onChange()
                }
            }
        }) { Text(if (confirmClear) "Toque de novo para confirmar" else "Apagar toda a árvore") }
    }
}

@Composable
fun OsmoseCard(osmose: Osmose, onChange: () -> Unit) {
    val ctx = LocalContext.current
    var on by remember { mutableStateOf(osmose.enabled) }
    var name by remember { mutableStateOf(osmose.myName) }
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
            "Guarda num arquivo tudo que o app aprendeu (pesos, corpus, árvore, conversa). Faça antes de desinstalar ou trocar de celular.",
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
    }
}

@Composable
fun TransformerScreen(lab: MiniLab, tree: Tree, osmose: Osmose) {
    val ts = Engine.ts
    var confirmReset by remember { mutableStateOf(false) }
    var confirmCorpus by remember { mutableStateOf(false) }
    val view = LocalView.current
    val training = ts.training
    DisposableEffect(training) {
        view.keepScreenOn = training
        onDispose { view.keepScreenOn = false }
    }
    val ctx = LocalContext.current
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            val from = ts.from
            val app = ctx.applicationContext
            ts.busy = true
            ts.status = "Lendo e preparando o arquivo…"
            Engine.scope.launch(Dispatchers.IO) {
                val msg = try {
                    val raw = app.contentResolver.openInputStream(uri)?.bufferedReader()?.readText() ?: ""
                    tree.importMass(raw, from)
                } catch (e: Throwable) {
                    "Erro ao importar: " + (e.message ?: "")
                }
                ts.corpusInfo = "Corpus de treino em massa: " + lab.corpusChars() + " caracteres"
                ts.status = msg
                ts.busy = false
            }
        }
    }

    Column(
        modifier = Modifier.verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            "Seu mini-transformer em Kotlin: " + lab.gpt.lay.n + " parâmetros (4 camadas, 128 dims, 4 cabeças, contexto 192). Cada nó da árvore tem um vetor próprio dentro do transformer, somado aos dos seus ancestrais: a árvore faz parte do modelo. Ele aprende sozinho (osmose, aba Aprender); aqui você treina mais fundo à mão. O treino continua se você trocar de aba ou sair do app (aparece uma notificação).",
            style = MaterialTheme.typography.bodyMedium
        )
        Text(ts.status, style = MaterialTheme.typography.titleSmall)
        Text(ts.corpusInfo, style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = !ts.busy, onClick = { Engine.startTrain(300) }) { Text("Treinar 300") }
            Button(enabled = !ts.busy, onClick = { Engine.startTrain(2000) }) { Text("Treinar 2000") }
            Button(enabled = ts.training, onClick = { ts.stop.set(true) }) { Text("Parar") }
        }
        OutlinedTextField(
            value = ts.from,
            onValueChange = { ts.from = it },
            label = { Text("WhatsApp: só mensagens de (opcional)") },
            modifier = Modifier.fillMaxWidth()
        )
        Button(enabled = !ts.busy, onClick = { picker.launch(arrayOf("text/*")) }) {
            Text("Importar texto para treino em massa")
        }
        TextButton(enabled = !ts.busy, onClick = {
            if (!confirmCorpus) {
                confirmCorpus = true
            } else {
                confirmCorpus = false
                Engine.scope.launch(Dispatchers.IO) {
                    lab.clearCorpus()
                    ts.corpusInfo = "Corpus de treino em massa: 0 caracteres"
                }
            }
        }) { Text(if (confirmCorpus) "Toque de novo para confirmar" else "Apagar corpus de treino") }
        OutlinedTextField(
            value = ts.prompt,
            onValueChange = { ts.prompt = it },
            label = { Text("Começo do texto (opcional)") },
            modifier = Modifier.fillMaxWidth()
        )
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
        Text(ts.output)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
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
            TextButton(enabled = !ts.busy, onClick = {
                if (!confirmReset) {
                    confirmReset = true
                } else {
                    confirmReset = false
                    Engine.scope.launch(Dispatchers.IO) {
                        lab.reset()
                        ts.status = "Transformer reiniciado do zero."
                        ts.output = ""
                    }
                }
            }) { Text(if (confirmReset) "Toque de novo para confirmar" else "Reiniciar transformer") }
        }
    }
}
