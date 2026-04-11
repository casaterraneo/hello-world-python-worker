import json
import base64
import struct
from urllib.parse import urlparse
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
# Tris — inferenza pure Python (pesi estratti da model.pt via export_weights.py)
# Nessuna dipendenza da numpy: usa solo struct (stdlib)
# ---------------------------------------------------------------------------

_tris_weights = None

# Shapes: w0(256x9) b0(256) w2(256x256) b2(256) w4(256x256) b4(256) w6(9x256) b6(9)
_W_SHAPES = [(256, 9), (256,), (256, 256), (256,), (256, 256), (256,), (9, 256), (9,)]
_W_NAMES  = ["w0", "b0", "w2", "b2", "w4", "b4", "w6", "b6"]

def _get_tris_weights():
    global _tris_weights
    if _tris_weights is None:
        from tris_weights import WEIGHTS_B64
        raw = base64.b64decode(WEIGHTS_B64)
        all_floats = struct.unpack(f"{len(raw)//4}f", raw)
        offset = 0
        weights = {}
        for name, shape in zip(_W_NAMES, _W_SHAPES):
            n = 1
            for s in shape:
                n *= s
            weights[name] = list(all_floats[offset:offset + n])
            offset += n
        _tris_weights = weights
    return _tris_weights

def _linear_relu(W, b, x, out_f, in_f):
    return [max(0.0, sum(W[i * in_f + j] * x[j] for j in range(in_f)) + b[i])
            for i in range(out_f)]

def _linear(W, b, x, out_f, in_f):
    return [sum(W[i * in_f + j] * x[j] for j in range(in_f)) + b[i]
            for i in range(out_f)]

def _tris_forward(board):
    w = _get_tris_weights()
    x = [float(v) for v in board]
    x = _linear_relu(w["w0"], w["b0"], x, 256, 9)
    x = _linear_relu(w["w2"], w["b2"], x, 256, 256)
    x = _linear_relu(w["w4"], w["b4"], x, 256, 256)
    q = _linear(w["w6"], w["b6"], x, 9, 256)
    legal = [i for i, v in enumerate(board) if v == 0]
    if not legal:
        raise ValueError("Nessuna mossa legale disponibile")
    return max(legal, key=lambda i: q[i])

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
