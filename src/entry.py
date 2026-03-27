from fastapi import FastAPI, Request
from workers import WorkerEntrypoint, Response

HARDCODED_SECRET = "test-secret-1234"

app = FastAPI()

@app.get("/")
async def root(request: Request):
    env = request.scope["env"]        
    await env.KV_BINDING.put("bar", "baz")
    bar = await env.KV_BINDING.get("bar")
    # return Response(f"Hello world TEST! Version: {env.APP_VERSION} {bar}")
    message = f"Hello world TEST! Version: {env.APP_VERSION} {bar}"
    return {"message": message}

@app.get("/test")
async def test_auth(request: Request):
    incoming = request.headers.get("X-Internal-Secret", "")

    if incoming != HARDCODED_SECRET:
        return {"error": "Unauthorized", "received": incoming or "nessun header"}

    return {
        "auth": "ok",
        "header_ricevuto": incoming,
        "note": "Depends NON usato — confronto diretto nell'handler"
    }

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        import asgi
        return await asgi.fetch(app, request.js_object, self.env)
