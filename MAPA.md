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
- [x] 7.2 v7.2 — editar a árvore à mão:
  - Aba Árvore: tocar no texto de um nó edita; "+" cria filho; "+ Raiz" cria nó raiz; "Importar lista" aceita lista indentada (cada `_` no começo = um nível; tab ou 2 espaços também valem). Reimportar reaproveita nós iguais (mesmo pai e texto).
  - `Tree.addNode`, `Tree.editNode`, `Tree.importOutline`; `Db.updateNode`, `Db.batch` (transação). Com "treinar com os caminhos", cada nó novo gera uma linha no corpus como `B > Br > Bra`.
  - O texto de cada nó é o que o transformer aprende a reproduzir, condicionado ao galho (vetores dos ancestrais somados, até 8 níveis).
  - `exemplo_abecedario.txt` na raiz do zip: lista de exemplo para importar.
- [x] 7.3 v7.3 — tamanhos de modelo e prompt de sistema:
  - `Sizes` (MiniTransformer.kt): Mini ≈1M (c128, L4), Médio ≈5M (c256, L6, 4 threads), Grande ≈15M (c384, L8, 2 threads); 50M foi removido na v8. Cabeças de 32 dims; lr menor nos maiores (1e-3, 6e-4, 4e-4, 3e-4) com aquecimento de 100 passos. Escolha na aba Transformer (SizeCard); trocar grava `model_size` em `arvore_prefs`, fecha o app e recomeça os pesos (corpus, árvore e conversa ficam).
  - Trava contra ciclo de falhas: `size_trying` marca a tentativa; se o app morrer por falta de memória, a próxima abertura volta ao Mini. OutOfMemoryError na construção também volta ao Mini. Manifest com `largeHeap`.
  - Memória estimada (floats x4 x (3 + threads) + atividades): Mini ~70 MB, Médio ~270 MB, Grande ~420 MB, 50M ~930 MB (provavelmente não cabe). FLOPs por passo (batch 4 x 192): 4,6 / 24 / 68 / 231 GFLOP.
  - Prompt de sistema: `Osmose.systemPrompt` (até 60 letras, padrão "Você é a Árvore..."), linha `@s ...` inserida no fluxo de treino antes de cada trecho de diálogo (`MiniLab.applySys`, o corpus guardado não muda) e no começo do prompt da resposta (`buildPrompt`). Trocar o prompt invalida o cache do corpus.
  - Status do treino manual mostra s/passo.
- [x] 8.0 v8 — tudo junto, menos peças soltas:
  - Abas: Conversa, Árvore, **Treino** (junta as antigas Aprender e Transformer), Prever. A aba Crescer da v7.4 saiu.
  - Uma só porta de entrada de texto: `Osmose.absorb` (arquivo, texto colado ou compartilhado de outro app). Removidos `Tree.learnFile`, `Tree.importMass`, `Tree.chat`, `MiniLab.generateFor` e o campo "só mensagens de".
  - Semente: na primeira abertura a osmose recebe `assets/semente.txt` (~90 pequenos diálogos @u/@a em português) para já ter o que treinar. Só uma vez (`seeded` em `arvore_prefs`).
  - Mistura de especialistas (`ai/DynamicSuperAI.kt`: `Expert`, `Router`, `DynamicSuperAI`) agora serve a aba Prever como segunda opinião (`Forecaster.predictMoe`, janelas de 6 valores, exige 14 valores positivos). Em testes com séries sintéticas ela NÃO supera a tendência linear em séries simples; fica como comparação experimental.
  - Tamanho 50M removido (não cabe na memória do celular). Tamanhos: Mini, Médio, Grande.
- [x] 8.1 Conversas no formato `Nome: fala` (linha vazia entre conversas): `Importer.isSpeakerChat/speakerDialogue/speakerBlocks` tiram os nomes; em cada conversa quem fala primeiro vira `@u` e os outros `@a`. Os nós da árvore usam só o texto das falas.
- [x] 9.0 v9 — especialistas DENTRO do transformer (MoE no feed-forward):
  - Cada camada tem até `ex` especialistas (Mini 4, Médio 2, Grande 1 = sem MoE). O especialista 0 é o feed-forward de sempre; os extras ficam no fim do vetor de pesos (`Layout.exW1/exB1/exW2/exB2`) e o roteador em `Layout.wr`. Começa com 1 ativo por camada (`MiniGpt.active`): o modelo é exatamente o da v8 e os pesos treinados são carregados sem perder nada (`load`, ramo `baseN`).
  - Com k >= 2 ativos: o roteador escolhe UM especialista por posição (top-1); saída = k * prob * especialista(x), assim com probabilidades iguais o gate vale 1. Gradiente do roteador escrito à mão (`Work.moeFwd/moeBwd`); conferido contra gradiente numérico em Python (erro ~1e-7). O botão Autoteste agora testa também o modelo com especialistas.
  - Crescer: `MiniGpt.grow` ativa +1 especialista em todas as camadas, copiando (com ruído de 2%) o mais usado e a linha do roteador dele. Automático em `MiniLab.maybeGrow`: a cada 1500 passos, se a perda (média móvel) melhorou menos de 2% e está acima de 0,8. Também pelo botão "Crescer agora" na aba Treino. Metadado do modelo guarda quantos especialistas estão ativos.
  - Sem perda de balanceamento: um especialista pode acabar dominando.
- [x] 9.1 Especialistas sem teto fixo e crescimento exponencial:
  - O modelo nasce com 1 especialista por camada (igual à v8, mesmo uso de memória). `MiniGpt.grow` DOBRA (1, 2, 4, 8...); se a memória não der, tenta metade, e assim por diante. Cada bloco novo vai no FIM do vetor de pesos (`Layout(cfg, ex)`), então nada do que já existe muda de lugar; o arquivo salvo diz quantos especialistas tem pelo tamanho (`capacityFor`), meta versão 2.
  - Gatilho (`MiniLab.maybeGrow`): a cada 300 passos, platô (< 2% de melhora) com perda > 0,6. Botão "Crescer agora" dobra na hora.
  - O único limite é a memória do app: antes de crescer `fits()` confere o pior momento da troca; se não couber, o app avisa quanto custa e quanto sobra. Custo no Mini: ~15 MB por especialista novo em cada camada.
  - Salvos da v9.0 (layout antigo com extras): só a parte comum é aproveitada.
  - `DynamicSuperAI` (aba Prever) também sem teto.
  - Autoteste: gradiente comum, gradiente com especialistas e "crescer não muda a saída".
- [x] 10.0 v10 — lista de melhorias (itens do "o que falta aprimorar"):
  - 1 Testar: o Autoteste agora roda sozinho uma vez a cada versão nova e o resultado aparece na aba Treino (🧪). NÃO foi compilado nem rodado pelo Claude; o primeiro build e esse resultado são a verificação real.
  - 2 Parar na hora certa: 1 bloco de cada 10 (400 caracteres) do corpus nunca é treinado (`MiniLab.isVal`); a cada 200 passos mede o erro neles (`validate`). Se piorar 2 vezes seguidas (>2% acima do melhor) a osmose pausa (`overfitting`) e só volta com +10% de texto novo. O crescimento por platô também não dispara nesse estado.
  - 3 Árvore primeiro: pergunta com semelhança >= 0,5 a um nó é respondida direto por ele (`Osmose.DIRECT_SIM`), sem esperar treino.
  - 4 Mais texto: `semente2.txt` (~100 diálogos novos), entra uma vez, mesmo em quem já tinha a primeira semente. Continua sendo pouco texto.
  - 5 Equilíbrio dos especialistas: perda auxiliar do Switch Transformer (`AUX_ALPHA` 0,02) no roteador; gradiente conferido em Python (erro ~1e-7) e incluído no Autoteste.
  - 6 Árvore: `Tree.consolidate` junta nós >= 93% parecidos (o antigo fica, herda filhos e usos); a coluna `hits` (Db v5) conta usos. Apagar nós nunca usados só vale com 60+ dias de contagem e 60+ dias de idade; duas confirmações.
  - 7 Medidor: `assets/avaliacao.txt` (30 pares pergunta|resposta). Nota 0-100 = probabilidade média de cada letra certa; medida a cada 200 passos e no botão "Medir agora". Mede o que está nas sementes (memória), não generalização; para isso vale o erro em texto novo.
  - 8 Especialistas no disco: NÃO feito. Exige mudar o vetor único de pesos (`w`) e o Adam para páginas em arquivo; sem compilador é arriscado demais. Plano: blocos de especialista em arquivo mapeado, LRU de uns poucos em memória, Adam só nos carregados.
- [x] 10.1 Scripts dentro do app (o equivalente real do txt do GraalVM, que não roda no Android):
  - Python 3.11 + NumPy pelo Chaquopy 16.0.0 (AGP 7.0-8.8, minSdk 24; só arm64-v8a), `app/src/main/python/runner.py`. JavaScript pelo Rhino 1.7.14 (Java puro, modo interpretado; `javax.script` NÃO existe no Android). `ScriptLab.kt` liga os dois; editor na aba Treino (Scripts), salvo em `arvore_prefs`.
  - O build agora precisa de python3 na máquina (o workflow instala o 3.11) e de internet para baixar o NumPy do repositório do Chaquopy.
  - C/Rust (LLVM) NÃO entrou: pede NDK + CMake + JNI, e nada no app precisa disso hoje.
  - Se o build quebrar por causa do Chaquopy, a v10.0 continua válida; é só voltar o zip anterior.
- [ ] 8. Polimento: ícone, gráfico da série, busca por ramo, cache de atenção (KV) para responder mais rápido

## Arquivos
- `app/src/main/java/com/arvoreneural/app/MainActivity.kt` — telas: Conversa, Árvore, Treino, Prever
- `.../data/Db.kt` — tabelas `nodes` e `points`
- `.../ai/Embedder.kt` — interface + HashEmbedder
- `.../ai/Tree.kt` — lógica da árvore e conversa (`chat`)
- `.../ai/MiniTransformer.kt` — transformer próprio (modelo, treino, geração, salvar/carregar)
- `.../ai/Importer.kt` (inclui `massBlocks` para o treino em massa) — leitura de conversas exportadas do WhatsApp
- `.../Engine.kt` — estado do app fora das telas, treino manual, conversa, backup
- `.../TrainService.kt` — serviço em primeiro plano
- `.../ai/Osmose.kt` — aprendizado contínuo, absorção de textos/conversas, resposta única da Conversa, 👍
- `.../ai/Forecaster.kt` — previsão (tendência + segunda opinião)
- `.../ai/DynamicSuperAI.kt` — especialistas que crescem
- `app/src/main/assets/semente.txt` — diálogos iniciais da osmose
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
