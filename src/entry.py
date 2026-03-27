from fastapi import FastAPI, Request, Depends, HTTPException, status
from workers import WorkerEntrypoint, Response
from fastapi.security import APIKeyHeader

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
        "@cf/openai/gpt-oss-120b",
        {
            "instructions": "You are a concise assistant.",
            "input": "What is the origin of the phrase 'The King is dead, long live the King!'?",
        },
    )
    return {"output": response.output}    
    
class Default(WorkerEntrypoint):
    async def fetch(self, request):
        import asgi
        return await asgi.fetch(app, request.js_object, self.env)
