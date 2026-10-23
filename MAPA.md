# MAPA — Árvore Neural (APK nativo via GitHub)

> Como retomar com o Claude: envie este arquivo no início de uma conversa nova e diga em qual etapa você está.

## Objetivo
App Android nativo que (1) organiza conhecimento numa árvore, (2) usa modelo de linguagem leve no celular e (3) analisa/prevê dados (ex.: finanças). Aprende em tempo real pela interação e por arquivos de texto. APK compilado pelo GitHub Actions.

## Decisões
- Kotlin + Jetpack Compose. Banco: SQLite puro (SQLiteOpenHelper), sem Room, para reduzir risco de erro de build.
- Sem Gradle wrapper no repositório: o GitHub Actions instala o Gradle 8.7 (AGP 8.5.2, Kotlin 1.9.24, Java 17).
- "Aprender" = a árvore cresce e se ajusta, não é treino de transformer do zero (inviável no celular).
- Embeddings: HashEmbedder (offline, sem arquivo de modelo). A interface `Embedder` permite trocar por transformer ONNX.

## Etapas
- [x] 0. Preparação: Termux, repositório e token no GitHub (ver PASSO_A_PASSO_TERMUX.md)
- [x] 1. Projeto base + workflow que gera o APK (`.github/workflows/build.yml`)
- [x] 2. Banco de dados SQLite + árvore de nós (`data/Db.kt`)
- [x] 3. Importar .txt e fatiar em trechos (`ai/Tree.kt` → `learnFile`, máx. 300 trechos)
- [x] 4. Aprendizado em tempo real: texto novo é encaixado sob o nó mais parecido (similaridade ≥ 0,30) e o nó pai é ajustado (`Tree.learn`)
- [x] 5. Prever/analisar números: tendência dos últimos 12 valores, reajustada a cada novo dado (`ai/Forecaster.kt`)
- [x] 5.1 Conversa: aba "Conversa" para trocar mensagens com a árvore (`Tree.chat`). Pergunta (com `?`) busca na árvore; frase afirmativa é aprendida. Respostas são recuperação de trechos, não texto gerado.
- [x] 6. Transformer próprio: port do mini.py (GPT de caracteres, T=64, C=128, H=4, L=4, AdamW) em Kotlin puro, com backpropagation escrito à mão (`ai/MiniTransformer.kt`). Treina no celular com o texto da árvore; pesos salvos no banco (versão 3). Aba "Transformer": treinar, gerar texto, autoteste do gradiente.
- [x] 6.1 Árvore dentro do transformer: cada nó tem um vetor de galho no modelo (somado aos dos ancestrais); treino e respostas usam o galho do nó; frases novas na Conversa treinam o modelo em segundo plano; resposta da Conversa = 🧠 geração do transformer + 🌳 texto do nó
- [x] 6.3 Treino em massa: importa um arquivo inteiro (até 8 milhões de caracteres) direto para o corpus do transformer, sem criar um nó por trecho; a árvore recebe só ~40 nós representativos. Os passos de treino misturam corpus e nós. Banco versão 4.
- [ ] 6.2 Usar os vetores internos do transformer para escolher/encaixar nós (no lugar do HashEmbedder) e cache de atenção (KV) para gerar mais rápido
- [x] 5.2 Importador de WhatsApp (tira data/hora/nome, junta mensagens seguidas, filtro por nome), respostas limpas (`tidy`), histórico da conversa salvo (banco versão 2) e botão "Apagar toda a árvore"
- [x] 7. v7 — Resposta única + osmose (`ai/Osmose.kt`):
  - Contexto do transformer: 64 -> 192 caracteres. Os pesos da v6 são migrados ao abrir (embeddings, vetores de galho e camadas são copiados; só as posições 64..191 começam aleatórias).
  - Formato de diálogo no corpus: cada fala é uma linha `@u texto` (usuário) ou `@a texto` (IA); linha vazia separa conversas/dias. Importar WhatsApp agora gera diálogo (`Importer.dialogue`); precisa do nome do usuário (aba Aprender › Osmose).
  - Resposta da Conversa: UMA só. O prompt é o diálogo recente + `@a `; a árvore escolhe o galho (vetor de galho do transformer). Se o modelo ainda está imaturo (menos de 300 passos ou perda >= 1,9), a resposta vem da memória da árvore ("Lembro disto: …") com aviso.
  - Osmose: (a) tudo que o usuário escreve na Conversa entra no corpus; afirmações viram nós; (b) laço de treino em segundo plano enquanto o app está aberto (10 passos, lote 4, descansa o dobro do tempo; pausa com bateria < 40% sem tomada ou celular >= 41 °C); (c) compartilhar texto/arquivo .txt de outro app com o Árvore Neural absorve (corpus + até 15–20 nós); (d) botão 👍 nas respostas guarda o trecho do diálogo (3 cópias).
  - Pesos só são gravados no banco a cada ~200 passos, ao pausar o app ou ao parar o treino manual (evita desgastar o armazenamento).
- [x] 7.1 v7.1 — segundo plano, backup, assinatura fixa:
  - `Engine.kt` (object): db, árvore, lab e osmose, mensagens da conversa, treino manual, rascunhos e mensagens das telas. Vive no processo, não na tela: trocar de aba não cancela resposta, treino nem importação (usam `Engine.scope`).
  - `TrainService.kt`: serviço em primeiro plano (tipo specialUse) com notificação fixa e botão "Desligar osmose". Sobe quando a osmose está ligada, há resposta pendente ou treino manual; some sozinho quando não há trabalho. A osmose agora treina também com o app em segundo plano (mantém as travas: bateria < 40% sem tomada, >= 41 °C). WAKE_LOCK parcial só enquanto há trabalho.
  - Backup: Aprender › Backup salva/restaura o arquivo `arvore.db` inteiro (pesos, corpus, árvore, conversa). Restaurar fecha o app (`frozen` impede gravar por cima).
  - Assinatura: `app/build.gradle.kts` usa `arvore.jks` na raiz se existir; o workflow grava o arquivo a partir dos Secrets KS_BASE64, KS_PASS, KS_ALIAS, KS_KEYPASS e roda `assembleRelease`. Sem Secrets, cai em `assembleDebug`. `*.jks` está no .gitignore.
  - A Activity não é mais recriada ao girar a tela (`configChanges`).
- [ ] 8. Polimento: ícone, gráfico da série, busca por ramo, cache de atenção (KV) para responder mais rápido

## Arquivos
- `app/src/main/java/com/arvoreneural/app/MainActivity.kt` — telas: Conversa, Árvore, Aprender, Prever
- `.../data/Db.kt` — tabelas `nodes` e `points`
- `.../ai/Embedder.kt` — interface + HashEmbedder
- `.../ai/Tree.kt` — lógica da árvore e conversa (`chat`)
- `.../ai/MiniTransformer.kt` — transformer próprio (modelo, treino, geração, salvar/carregar)
- `.../ai/Importer.kt` (inclui `massBlocks` para o treino em massa) — leitura de conversas exportadas do WhatsApp
- `.../Engine.kt` — estado do app fora das telas, treino manual, conversa, backup
- `.../TrainService.kt` — serviço em primeiro plano
- `.../ai/Osmose.kt` — aprendizado contínuo, absorção de textos/conversas, resposta única da Conversa, 👍
- `.../ai/Forecaster.kt` — previsão
- `.github/workflows/build.yml` — build do APK

## O que isto é (e não é)
- A árvore é uma memória com busca por similaridade (HashEmbedder: contagem de palavras).
- O transformer é um GPT de caracteres de verdade, escrito à mão em Kotlin, treinado do zero no celular. É pequeno (~0,85 milhão de parâmetros) e começa gerando texto sem sentido; melhora devagar com mais texto e mais passos.
- Dentro do modelo, a árvore aparece como vetores de galho: nó filho = vetor próprio + vetores dos ancestrais.
- Os pesos são criados pelo treino e guardados no banco do app. O que vai no APK é só o código.

## Limites conhecidos
- Este código foi escrito sem compilar no ambiente do Claude: o primeiro build pode apontar erros. Se falhar, copie o log do passo "Compilar APK" e envie ao Claude.
- HashEmbedder entende palavras parecidas, não significado. O item 6.1 melhora isso.
- O treino em Kotlin puro é bem mais lento que o PyTorch do Termux. Use batch pequeno (8) e deixe o app aberto. Vocabulário fixo (ASCII + acentos do português): outros caracteres são ignorados.
- O gradiente escrito à mão foi conferido com gradiente numérico (erro ~1e-5) numa versão de referência em Python; o botão "Autoteste" refaz essa conferência no próprio código Kotlin.

## Como a resposta é feita (v7)
1. Você escreve. A mensagem vira uma linha `@u …` no corpus (osmose). Se for afirmação (sem `?`) e não repetir um nó existente, vira um nó novo e o transformer treina alguns passos nele.
2. A árvore (HashEmbedder: contagem de palavras) acha o nó mais parecido. Se a similaridade for >= 0,30, o caminho desse nó até a raiz vira o "galho" que condiciona o transformer.
3. O transformer recebe as últimas falas no mesmo formato do treino, termina em `@a `, e escreve a resposta até a quebra de linha (máx. 90 caracteres). Em modo detalhe aparece o galho escolhido e os passos/perda.
4. Se o modelo ainda não amadureceu, a resposta é o texto do nó mais parecido (pergunta) ou "Anotado." (afirmação), com aviso.

## Limites conhecidos da v7
- Código escrito sem compilar no ambiente do Claude (sem Kotlin/Android SDK lá). A lógica nova de diálogo, prompt e migração de pesos foi conferida num porte em Python; o resto precisa do primeiro build. Se falhar, mande o log do passo "Compilar APK".
- Contexto 192 deixa cada passo de treino e cada resposta mais lentos que na v6 (a geração ainda recalcula tudo a cada caractere; falta o cache KV).
- O modelo continua pequeno (~1 milhão de parâmetros): aprende o jeito de falar, não raciocina.
- O Android não deixa o app ler mensagens de outros apps sozinho: o caminho seguro é compartilhar/exportar.
- A osmose treina também em segundo plano (serviço com notificação). Alguns fabricantes ainda matam o serviço se a bateria do app não estiver em "Sem restrições".
- Mudar "A IA fala como eu" só vale para importações novas; o que já está no corpus mantém os papéis antigos.
