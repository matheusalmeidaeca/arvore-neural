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
