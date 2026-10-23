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
        setContent { MaterialTheme { App(db, tree) } }
    }
}

@Composable
fun App(db: Db, tree: Tree) {
    var tab by remember { mutableIntStateOf(0) }
    var version by remember { mutableIntStateOf(0) }
    val labels = listOf("Árvore", "Aprender", "Prever")
    val icons = listOf("🌳", "✍️", "📈")
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
                0 -> TreeScreen(db, tree, version) { version++ }
                1 -> LearnScreen(tree) { version++ }
                else -> ForecastScreen(db)
            }
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
fun LearnScreen(tree: Tree, onChange: () -> Unit) {
    val ctx = LocalContext.current
    var text by remember { mutableStateOf("") }
    var msg by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            scope.launch {
                val n = withContext(Dispatchers.IO) {
                    val raw = ctx.contentResolver.openInputStream(uri)?.bufferedReader()?.readText() ?: ""
                    tree.learnFile(raw)
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
        Button(onClick = { picker.launch(arrayOf("text/*")) }) { Text("Importar arquivo .txt") }
        Text(msg)
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
