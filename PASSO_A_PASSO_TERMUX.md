# Passo a passo: Termux → GitHub → APK

1. No Termux (de preferência da F-Droid), rode:
   pkg update -y && pkg install -y git unzip
   termux-setup-storage        (aceite a permissão)

2. No GitHub (navegador): crie um repositório vazio chamado arvore-neural (sem README).

3. Crie um token: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token. Marque os escopos `repo` e `workflow`. Copie o token.

4. Descompacte o projeto (o zip está em Downloads):
   mkdir -p ~/arvore-neural && cd ~/arvore-neural
   unzip -o ~/storage/downloads/ArvoreNeural.zip

5. Envie ao GitHub:
   git config --global user.name "SEU NOME"
   git config --global user.email "SEU@EMAIL"
   git config --global credential.helper store
   git init -b main
   git add .
   git commit -m "Projeto inicial"
   git remote add origin https://github.com/SEU_USUARIO/arvore-neural.git
   git push -u origin main
   (usuário = seu usuário do GitHub; senha = o TOKEN)

6. No GitHub: aba Actions → "Build APK" → aguarde ~5 min → Artifacts → arvore-neural-apk. Baixe, extraia o zip e instale o app-debug.apk (permita "fontes desconhecidas").

Para atualizar depois: git add . && git commit -m "msg" && git push

## Atualizar da v6 para a v7
1. No Termux, dentro da pasta do projeto, apague os arquivos antigos do código e descompacte a v7 por cima:
   cd ~/arvore-neural && unzip -o ~/storage/downloads/ArvoreNeural_v7.zip
2. git add . && git commit -m "v7: osmose" && git push
3. Aguarde o GitHub Actions e instale o novo APK por cima do antigo (os dados e o transformer treinado ficam).

## Usar a osmose
- Aba Aprender › Osmose: digite seu nome como aparece no WhatsApp.
- No WhatsApp: conversa › ⋮ › Mais › Exportar conversa › Sem mídia › Árvore Neural.
- Deixe o app aberto (e, de preferência, na tomada): ele treina sozinho em pequenos passos.

## v7.1: treino em segundo plano, backup e assinatura fixa

### A) Criar a chave fixa (uma vez só)
Faça FORA da pasta do projeto, para a chave nunca ir ao GitHub:
1. pkg install -y openjdk-17
2. cd ~ && keytool -genkeypair -v -keystore arvore.jks -alias arvore -keyalg RSA -keysize 2048 -validity 10000
   (use a MESMA senha para o arquivo e para a chave; anote)
3. base64 -w0 arvore.jks   (copie a saída inteira)
4. GitHub › repositório › Settings › Secrets and variables › Actions › New repository secret. Crie 4:
   KS_BASE64 = a saída do passo 3
   KS_PASS = a senha
   KS_ALIAS = arvore
   KS_KEYPASS = a mesma senha
5. Guarde ~/arvore.jks e a senha em outro lugar também (nuvem, e-mail para você). Se perder, não dá mais para atualizar o app por cima.

### B) Instalar a v7.1
- ATENÇÃO, só nesta primeira vez: o app antigo foi assinado com outra chave, então o Android não deixa instalar por cima. É preciso desinstalar o antigo, e isso APAGA tudo que ele aprendeu. O app antigo não tem backup. Depois desta instalação, use Aprender › Backup antes de qualquer desinstalação.
- cd ~/arvore-neural && unzip -o ~/storage/downloads/ArvoreNeural_v7_1.zip
- git add . && git commit -m "v7.1" && git push
- Actions › Build APK › Artifacts › arvore-neural-apk. O APK assinado vem como app-release.apk.
- Os próximos APKs instalam por cima, sem desinstalar.

### C) No celular, para o treino não parar
- Aceite a permissão de notificações quando o app pedir.
- Ajustes › Apps › Árvore Neural › Bateria › Sem restrições (há um botão na aba Aprender que abre os ajustes do app).
