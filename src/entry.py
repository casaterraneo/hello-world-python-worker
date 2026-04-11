import json
import base64
import io
from urllib.parse import urlparse
import numpy as np
from workers import WorkerEntrypoint, Response

HARDCODED_SECRET = "test-secret-1234"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_resp(data, status=200):
    return Response(
        json.dumps(data, separators=(",", ":")),
        status=status,
        headers={"Content-Type": "application/json"},
    )

def _error(msg, status=400):
    return _json_resp({"detail": msg}, status=status)

# ---------------------------------------------------------------------------
# Tris — inferenza NumPy (pesi estratti da model.pt via export_weights.py)
# ---------------------------------------------------------------------------

_tris_weights = None

def _get_tris_weights():
    global _tris_weights
    if _tris_weights is None:
        from tris_weights import WEIGHTS_B64
        buf = io.BytesIO(base64.b64decode(WEIGHTS_B64))
        _tris_weights = np.load(buf)
    return _tris_weights

def _tris_forward(board):
    w = _get_tris_weights()
    x = np.array(board, dtype=np.float32)
    x = np.maximum(0.0, x @ w["w0"].T + w["b0"])
    x = np.maximum(0.0, x @ w["w2"].T + w["b2"])
    x = np.maximum(0.0, x @ w["w4"].T + w["b4"])
    q = x @ w["w6"].T + w["b6"]
    legal = [i for i, v in enumerate(board) if v == 0]
    if not legal:
        raise ValueError("Nessuna mossa legale disponibile")
    return int(max(legal, key=lambda i: float(q[i])))

# ---------------------------------------------------------------------------
# Worker entrypoint — routing nativo senza FastAPI
# ---------------------------------------------------------------------------

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        path = urlparse(str(request.url)).path
        method = str(request.method)

        # GET /
        if method == "GET" and path == "/":
            await self.env.KV_BINDING.put("bar", "baz")
            bar = await self.env.KV_BINDING.get("bar")
            message = f"Hello world TEST! Version: {self.env.APP_VERSION} {bar}"
            return _json_resp({"message": message})

        # GET /test-depends-async
        if method == "GET" and path == "/test-depends-async":
            key = request.headers.get("x-internal-secret")
            if key != HARDCODED_SECRET:
                return _error("Unauthorized", status=401)
            return _json_resp({"auth": "ok", "method": "Depends asincrona"})

        # POST /tris/move
        if method == "POST" and path == "/tris/move":
            try:
                body = json.loads(await request.text())
            except Exception:
                return _error("JSON non valido")
            board = body.get("board") if isinstance(body, dict) else None
            if not isinstance(board, list) or len(board) != 9:
                return _error("board deve avere esattamente 9 valori")
            for v in board:
                if v not in (0, 1, -1, 0.0, 1.0, -1.0):
                    return _error("I valori del board devono essere 0, 1 o -1")
            move = _tris_forward(board)
            return _json_resp({"move": move})

        return _error("Not found", status=404)
