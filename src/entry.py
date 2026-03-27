from workers import WorkerEntrypoint, Response


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        await self.env.KV_BINDING.put("bar", "baz")
        bar = await self.env.KV_BINDING.get("bar")
        return Response(f"Hello world TEST! Version: {self.env.APP_VERSION} {bar}")
