package com.arvoreneural.app

import android.net.Uri
import android.os.Bundle
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
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
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
import com.arvoreneural.app.ai.miniSelfTest
import com.arvoreneural.app.ai.Tree
import com.arvoreneural.app.data.Db
import com.arvoreneural.app.data.Node
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val db = Db(applicationContext)
        val tree = Tree(db, HashEmbedder())
        val lab = MiniLab(db)
        tree.lab = lab
        setContent { MaterialTheme { App(db, tree, lab) } }
    }
}

@Composable
fun App(db: Db, tree: Tree, lab: MiniLab) {
    var tab by remember { mutableIntStateOf(0) }
    var version by remember { mutableIntStateOf(0) }
    val msgs = remember { mutableStateListOf<Msg>() }
    LaunchedEffect(Unit) {
        val h = withContext(Dispatchers.IO) { db.chatHistory() }
        msgs.addAll(0, h.map { Msg(it.first, it.second) })
    }
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
                0 -> ChatScreen(db, tree, msgs) { version++ }
                1 -> TreeScreen(db, tree, version) { version++ }
                2 -> LearnScreen(tree, db) { version++ }
                3 -> ForecastScreen(db)
                else -> TransformerScreen(lab, tree)
            }
        }
    }
}

class Msg(val fromUser: Boolean, val text: String)

@Composable
fun ChatScreen(db: Db, tree: Tree, msgs: MutableList<Msg>, onChange: () -> Unit) {
    var input by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()
    LaunchedEffect(msgs.size) {
        if (msgs.isNotEmpty()) listState.animateScrollToItem(msgs.size - 1)
    }
    Column(Modifier.fillMaxSize()) {
        if (msgs.isNotEmpty()) {
            TextButton(onClick = {
                msgs.clear()
                scope.launch { withContext(Dispatchers.IO) { db.clearChat() } }
            }) { Text("Limpar conversa") }
        }
        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            state = listState,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (msgs.isEmpty()) {
                item {
                    Text("Converse comigo! Frases afirmativas eu aprendo. Perguntas (com ?) eu respondo com o que já sei.")
                }
            }
            items(msgs) { m ->
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = if (m.fromUser) Arrangement.End else Arrangement.Start
                ) {
                    Surface(
                        color = if (m.fromUser) MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.surfaceVariant,
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.widthIn(max = 300.dp)
                    ) { Text(m.text, Modifier.padding(10.dp)) }
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                placeholder = { Text("Mensagem") },
                modifier = Modifier.weight(1f)
            )
            Spacer(Modifier.width(8.dp))
            Button(onClick = {
                val t = input.trim()
                if (t.isNotEmpty()) {
                    input = ""
                    msgs.add(Msg(true, t))
                    scope.launch {
                        val r = withContext(Dispatchers.IO) {
                            db.addChat(true, t)
                            val a = tree.chat(t)
                            db.addChat(false, a)
                            a
                        }
                        msgs.add(Msg(false, r))
                        onChange()
                    }
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
    var nodes by remember { mutableStateOf(listOf<Node>()) }
    var q by remember { mutableStateOf("") }
    var answers by remember { mutableStateOf(listOf<Pair<Node, Float>>()) }
    val scope = rememberCoroutineScope()
    LaunchedEffect(version) { nodes = withContext(Dispatchers.IO) { db.allNodes() } }
    val flat = remember(nodes) { flatten(nodes) }

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = q,
            onValueChange = { q = it },
            label = { Text("Pergunte à árvore") },
            modifier = Modifier.fillMaxWidth()
        )
        Button(onClick = {
            scope.launch { answers = withContext(Dispatchers.IO) { tree.ask(q) } }
        }) { Text("Buscar") }
        if (answers.isNotEmpty()) {
            Text("Mais parecidos:", style = MaterialTheme.typography.titleSmall)
            answers.forEach { (n, s) -> Text("• ${n.content.take(140)} (${(s * 100).toInt()}%)") }
        }
        Text("${nodes.size} nós na árvore", style = MaterialTheme.typography.titleSmall)
        LazyColumn {
            items(flat) { (n, d) ->
                Row(Modifier.fillMaxWidth().padding(start = (d * 16).dp)) {
                    Text(n.title, modifier = Modifier.weight(1f).padding(top = 12.dp))
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
fun LearnScreen(tree: Tree, db: Db, onChange: () -> Unit) {
    val ctx = LocalContext.current
    var text by remember { mutableStateOf("") }
    var msg by remember { mutableStateOf("") }
    var onlyFrom by remember { mutableStateOf("") }
    var confirmClear by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            scope.launch {
                val n = withContext(Dispatchers.IO) {
                    val raw = ctx.contentResolver.openInputStream(uri)?.bufferedReader()?.readText() ?: ""
                    tree.learnFile(raw, onlyFrom)
                }
                msg = "$n trechos aprendidos"
                onChange()
            }
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Ensine algo à árvore") },
            modifier = Modifier.fillMaxWidth().heightIn(min = 140.dp)
        )
        Button(onClick = {
            val t = text
            scope.launch {
                withContext(Dispatchers.IO) { tree.learn(t) }
                text = ""
                msg = "Aprendido!"
                onChange()
            }
        }) { Text("Aprender") }
        OutlinedTextField(
            value = onlyFrom,
            onValueChange = { onlyFrom = it },
            label = { Text("WhatsApp: só mensagens de (nome, opcional)") },
            modifier = Modifier.fillMaxWidth()
        )
        Button(onClick = { picker.launch(arrayOf("text/*")) }) { Text("Importar arquivo .txt") }
        Text(msg)
        TextButton(onClick = {
            if (!confirmClear) {
                confirmClear = true
            } else {
                confirmClear = false
                scope.launch {
                    withContext(Dispatchers.IO) { db.clearNodes() }
                    msg = "Árvore apagada"
                    onChange()
                }
            }
        }) { Text(if (confirmClear) "Toque de novo para confirmar" else "Apagar toda a árvore") }
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
fun TransformerScreen(lab: MiniLab, tree: Tree) {
    var status by remember { mutableStateOf("Passos treinados: " + lab.gpt.step) }
    var training by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var prompt by remember { mutableStateOf("") }
    var output by remember { mutableStateOf("") }
    var confirmReset by remember { mutableStateOf(false) }
    val stop = remember { AtomicBoolean(false) }
    val scope = rememberCoroutineScope()
    val view = LocalView.current
    DisposableEffect(training) {
        view.keepScreenOn = training
        onDispose { view.keepScreenOn = false }
    }
    val ctx = LocalContext.current
    var onlyFrom by remember { mutableStateOf("") }
    var corpusInfo by remember { mutableStateOf("") }
    var confirmCorpus by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        val n = withContext(Dispatchers.IO) { lab.corpusChars() }
        corpusInfo = "Corpus de treino em massa: " + n + " caracteres"
    }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            scope.launch {
                busy = true
                status = "Lendo e preparando o arquivo…"
                val msg = withContext(Dispatchers.IO) {
                    val raw = ctx.contentResolver.openInputStream(uri)?.bufferedReader()?.readText() ?: ""
                    tree.importMass(raw, onlyFrom)
                }
                val n = withContext(Dispatchers.IO) { lab.corpusChars() }
                corpusInfo = "Corpus de treino em massa: " + n + " caracteres"
                status = msg
                busy = false
            }
        }
    }
    val startTrain = { total: Int ->
        scope.launch {
            busy = true
            training = true
            stop.set(false)
            val msg = withContext(Dispatchers.Default) {
                if (lab.corpusSize() < 80) {
                    return@withContext "Ensine mais texto (ou importe um arquivo) antes de treinar."
                }
                val t0 = System.currentTimeMillis()
                lab.train(total, 8, -1L, stop) { i, loss ->
                    val sec = (System.currentTimeMillis() - t0) / 1000
                    val ls = "%.3f".format(loss)
                    status = "passo " + lab.gpt.step + " | perda " + ls + " | " + i + "/" + total + " | " + sec + "s"
                }
                val fin = "%.3f".format(lab.gpt.lastLoss)
                "Treino concluído: " + lab.gpt.step + " passos no total, perda " + fin + "."
            }
            status = msg
            training = false
            busy = false
        }
    }

    Column(
        modifier = Modifier.verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            "Seu mini-transformer em Kotlin: " + lab.gpt.lay.n + " parâmetros (4 camadas, 128 dims, 4 cabeças, contexto 64). Cada nó da árvore tem um vetor próprio dentro do transformer, somado aos dos seus ancestrais: a árvore faz parte do modelo. Aprende sozinho a cada frase que você ensina na Conversa; aqui você treina mais fundo. Deixe o app aberto durante o treino.",
            style = MaterialTheme.typography.bodyMedium
        )
        Text(status, style = MaterialTheme.typography.titleSmall)
        Text(corpusInfo, style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = !busy, onClick = { startTrain(300) }) { Text("Treinar 300") }
            Button(enabled = !busy, onClick = { startTrain(2000) }) { Text("Treinar 2000") }
            Button(enabled = training, onClick = { stop.set(true) }) { Text("Parar") }
        }
        OutlinedTextField(
            value = onlyFrom,
            onValueChange = { onlyFrom = it },
            label = { Text("WhatsApp: só mensagens de (opcional)") },
            modifier = Modifier.fillMaxWidth()
        )
        Button(enabled = !busy, onClick = { picker.launch(arrayOf("text/*")) }) {
            Text("Importar texto para treino em massa")
        }
        TextButton(enabled = !busy, onClick = {
            if (!confirmCorpus) {
                confirmCorpus = true
            } else {
                confirmCorpus = false
                scope.launch {
                    withContext(Dispatchers.IO) { lab.clearCorpus() }
                    corpusInfo = "Corpus de treino em massa: 0 caracteres"
                }
            }
        }) { Text(if (confirmCorpus) "Toque de novo para confirmar" else "Apagar corpus de treino") }
        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            label = { Text("Começo do texto (opcional)") },
            modifier = Modifier.fillMaxWidth()
        )
        Button(enabled = !busy, onClick = {
            val p = prompt
            scope.launch {
                busy = true
                output = "Gerando…"
                output = withContext(Dispatchers.Default) {
                    p + lab.generateFree(p, 200)
                }
                busy = false
            }
        }) { Text("Gerar texto") }
        Text(output)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = !busy, onClick = {
                scope.launch {
                    busy = true
                    output = "Testando…"
                    output = withContext(Dispatchers.Default) { miniSelfTest() }
                    busy = false
                }
            }) { Text("Autoteste") }
            TextButton(enabled = !busy, onClick = {
                if (!confirmReset) {
                    confirmReset = true
                } else {
                    confirmReset = false
                    scope.launch {
                        withContext(Dispatchers.IO) { lab.reset() }
                        status = "Transformer reiniciado do zero."
                        output = ""
                    }
                }
            }) { Text(if (confirmReset) "Toque de novo para confirmar" else "Reiniciar transformer") }
        }
    }
}
