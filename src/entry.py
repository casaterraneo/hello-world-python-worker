from fastapi import FastAPI, Request
from workers import WorkerEntrypoint, Response

app = FastAPI()

@app.get("/")
async def root():
    await self.env.KV_BINDING.put("bar", "baz")
    bar = await self.env.KV_BINDING.get("bar")
    # return Response(f"Hello world TEST! Version: {self.env.APP_VERSION} {bar}")
    message = f"Hello world TEST! Version: {self.env.APP_VERSION} {bar}"
    return {message}

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        import asgi
        return await asgi.fetch(app, request.js_object, self.env)
