import io
import json
import sys
import time
import contextlib
import traceback

import numpy as np

LIMITE_SEGUNDOS = 10


def _com_prazo(prazo):
    """Função de rastreio: se o script passar do prazo, levanta TimeoutError (pega laços infinitos)."""
    def local(frame, evento, arg):
        if time.monotonic() > prazo:
            raise TimeoutError("O script passou de %d segundos e foi interrompido." % LIMITE_SEGUNDOS)
        return local

    def globalf(frame, evento, arg):
        if time.monotonic() > prazo:
            raise TimeoutError("O script passou de %d segundos e foi interrompido." % LIMITE_SEGUNDOS)
        return local
    return globalf


def rodar(codigo, entradas):
    """Roda o script do usuário. O script enxerga `np` (NumPy) e `entradas` (array de floats)
    e deve deixar o resultado na variável `disparou`. Devolve um JSON com ok/disparou/saida/erro."""
    saida = io.StringIO()
    ns = {"np": np, "entradas": np.array(list(entradas), dtype=float)}
    antigo = sys.gettrace()
    try:
        compilado = compile(codigo, "neuronio.py", "exec")
        sys.settrace(_com_prazo(time.monotonic() + LIMITE_SEGUNDOS))
        with contextlib.redirect_stdout(saida):
            exec(compilado, ns)
        sys.settrace(antigo)
        if "disparou" not in ns:
            return json.dumps({"ok": False, "erro": "O script precisa definir a variável disparou (True ou False)."})
        return json.dumps({"ok": True, "disparou": bool(ns["disparou"]), "saida": saida.getvalue()})
    except BaseException:  # inclui SystemExit (sys.exit) e KeyboardInterrupt, que não são Exception
        sys.settrace(antigo)
        return json.dumps({"ok": False, "erro": traceback.format_exc(limit=3)})
    finally:
        sys.settrace(antigo)
