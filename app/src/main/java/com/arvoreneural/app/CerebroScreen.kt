package com.arvoreneural.app

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

@Composable
fun CerebroScreen() {
    val ctx = LocalContext.current.applicationContext
    val msgs = CerebroState.msgs
    val listState = rememberLazyListState()
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) CerebroState.importModel(ctx, uri)
    }
    LaunchedEffect(Unit) { CerebroState.refresh(ctx) }
    LaunchedEffect(msgs.size, CerebroState.stats) {
        if (msgs.isNotEmpty()) listState.animateScrollToItem(msgs.size - 1)
    }
    Column(Modifier.fillMaxSize()) {
        Text("🧠 Cérebro (llama.cpp dentro do app)", style = MaterialTheme.typography.titleMedium)
        Text(CerebroState.status, style = MaterialTheme.typography.bodySmall)
        if (CerebroState.modelInfo.isNotEmpty()) {
            Text(CerebroState.modelInfo, style = MaterialTheme.typography.bodySmall)
        }
        Row(Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                enabled = !CerebroState.working && !CerebroState.generating,
                onClick = { picker.launch(arrayOf("*/*")) }
            ) { Text("Escolher GGUF") }
            if (!CerebroState.loaded) {
                Button(
                    enabled = CerebroState.hasFile && !CerebroState.working,
                    onClick = { CerebroState.load(ctx) }
                ) { Text("Carregar") }
            } else {
                TextButton(
                    enabled = !CerebroState.generating,
                    onClick = { CerebroState.unload() }
                ) { Text("Liberar memória") }
            }
        }
        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth().padding(top = 8.dp),
            state = listState,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (msgs.isEmpty()) {
                item {
                    Text(
                        "Aqui quem responde é o modelo GGUF que você escolher (por exemplo um Qwen), rodando no próprio celular. " +
                            "Escolha o arquivo uma vez; ele é copiado para dentro do app. Depois toque em Carregar e converse."
                    )
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
                    ) { Text(if (m.text.isEmpty()) "✍️ pensando…" else m.text, Modifier.padding(10.dp)) }
                }
            }
        }
        if (CerebroState.stats.isNotEmpty()) {
            Text(CerebroState.stats, style = MaterialTheme.typography.bodySmall)
        }
        Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = CerebroState.input,
                onValueChange = { CerebroState.input = it },
                placeholder = { Text(if (CerebroState.loaded) "Mensagem" else "Carregue o modelo primeiro") },
                enabled = CerebroState.loaded,
                modifier = Modifier.weight(1f)
            )
            Spacer(Modifier.width(8.dp))
            if (CerebroState.generating) {
                Button(onClick = { CerebroState.stop() }) { Text("Parar") }
            } else {
                Button(
                    enabled = CerebroState.loaded,
                    onClick = {
                        val t = CerebroState.input.trim()
                        if (t.isNotEmpty()) {
                            CerebroState.input = ""
                            CerebroState.send(t)
                        }
                    }
                ) { Text("Enviar") }
            }
        }
        if (msgs.isNotEmpty() && !CerebroState.generating) {
            TextButton(onClick = { CerebroState.clear() }) { Text("Limpar conversa") }
        }
    }
}
