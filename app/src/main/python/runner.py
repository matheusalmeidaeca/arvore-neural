import io
import json
import contextlib
import traceback

import numpy as np


def rodar(codigo, entradas):
    """Roda o script do usuário. O script enxerga `np` (NumPy) e `entradas` (array de floats)
    e deve deixar o resultado na variável `disparou`. Devolve um JSON com ok/disparou/saida/erro."""
    saida = io.StringIO()
    ns = {"np": np, "entradas": np.array(list(entradas), dtype=float)}
    try:
        with contextlib.redirect_stdout(saida):
            exec(compile(codigo, "neuronio.py", "exec"), ns)
        if "disparou" not in ns:
            return json.dumps({"ok": False, "erro": "O script precisa definir a variável disparou (True ou False)."})
        return json.dumps({"ok": True, "disparou": bool(ns["disparou"]), "saida": saida.getvalue()})
    except Exception:
        return json.dumps({"ok": False, "erro": traceback.format_exc(limit=3)})
