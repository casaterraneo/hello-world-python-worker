from workers import WorkerEntrypoint, Response


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        return Response(f"Hello world TEST! Version: {self.env.APP_VERSION}")
