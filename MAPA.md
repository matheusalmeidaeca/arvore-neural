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
- [ ] 6.2 Usar os vetores internos do transformer para escolher/encaixar nós (no lugar do HashEmbedder) e cache de atenção (KV) para gerar mais rápido
- [x] 5.2 Importador de WhatsApp (tira data/hora/nome, junta mensagens seguidas, filtro por nome), respostas limpas (`tidy`), histórico da conversa salvo (banco versão 2) e botão "Apagar toda a árvore"
- [ ] 7. Polimento: ícone, gráfico da série, exportar/importar backup do banco, busca por ramo, APK assinado (release)

## Arquivos
- `app/src/main/java/com/arvoreneural/app/MainActivity.kt` — telas: Conversa, Árvore, Aprender, Prever
- `.../data/Db.kt` — tabelas `nodes` e `points`
- `.../ai/Embedder.kt` — interface + HashEmbedder
- `.../ai/Tree.kt` — lógica da árvore e conversa (`chat`)
- `.../ai/MiniTransformer.kt` — transformer próprio (modelo, treino, geração, salvar/carregar)
- `.../ai/Importer.kt` — leitura de conversas exportadas do WhatsApp
- `.../ai/Forecaster.kt` — previsão
- `.github/workflows/build.yml` — build do APK

## O que isto é (e não é)
- A árvore é uma memória com busca por similaridade (HashEmbedder: contagem de palavras).
- O transformer é um GPT de caracteres de verdade, escrito à mão em Kotlin, treinado do zero no celular. É pequeno (~0,85 milhão de parâmetros) e começa gerando texto sem sentido; melhora devagar com mais texto e mais passos.
- Dentro do modelo, a árvore aparece como vetores de galho: nó filho = vetor próprio + vetores dos ancestrais.
- Os pesos são criados pelo treino e guardados no banco do app. O que vai no APK é só o código.

## Limites conhecidos
- Este código foi escrito sem compilar no ambiente do Claude: o primeiro build pode apontar erros. Se falhar, copie o log do passo "Compilar APK debug" e envie ao Claude.
- HashEmbedder entende palavras parecidas, não significado. O item 6.1 melhora isso.
- O treino em Kotlin puro é bem mais lento que o PyTorch do Termux. Use batch pequeno (8) e deixe o app aberto. Vocabulário fixo (ASCII + acentos do português): outros caracteres são ignorados.
- O gradiente escrito à mão foi conferido com gradiente numérico (erro ~1e-5) numa versão de referência em Python; o botão "Autoteste" refaz essa conferência no próprio código Kotlin.
