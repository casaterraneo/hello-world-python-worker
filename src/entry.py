from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from workers import WorkerEntrypoint, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import numpy as np
import base64
import io

HARDCODED_SECRET = "test-secret-1234"

app = FastAPI(title="Hello World Python Worker")

@app.get("/")
async def root(request: Request):
    env = request.scope["env"]
    await env.KV_BINDING.put("bar", "baz")
    bar = await env.KV_BINDING.get("bar")
    # return Response(f"Hello world TEST! Version: {env.APP_VERSION} {bar}")
    message = f"Hello world TEST! Version: {env.APP_VERSION} {bar}"
    return {"message": message}

api_key_header = APIKeyHeader(name="X-Internal-Secret", auto_error=False)

# Depends ASYNC
async def verify_secret_async(key: str = Depends(api_key_header)):
    if key != HARDCODED_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Unauthorized", 
            headers={"WWW-Authenticate": "API Key"}
        ) 
    return key

@app.get("/test-depends-async")
async def test_depends_async(key: str = Depends(verify_secret_async)):
    return {"auth": "ok", "method": "Depends asincrona"}

@app.get("/test-ai")
async def test_ai(request: Request):
    env = request.scope["env"]
    response = await env.AI.run(
        "@cf/meta/llama-3.1-8b-instruct", 
        {
            "prompt": "What is the origin of the phrase Hello, World"
        },
    )
    result = response.to_py()
    return {"output": result}


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

def _tris_forward(board: list) -> int:
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

class TrisRequest(BaseModel):
    board: list[float]

@app.post("/tris/move")
async def tris_move(body: TrisRequest):
    if len(body.board) != 9:
        raise HTTPException(status_code=400, detail="board deve avere esattamente 9 valori")
    for v in body.board:
        if v not in (0.0, 1.0, -1.0):
            raise HTTPException(status_code=400, detail="I valori del board devono essere 0, 1 o -1")
    move = _tris_forward(body.board)
    return {"move": move}
    
class Default(WorkerEntrypoint):
    async def fetch(self, request):
        import asgi
        return await asgi.fetch(app, request.js_object, self.env)
