import asyncio
from aiohttp import web

async def handle(request):
    if request.method == 'POST':
        data = await request.post()
        message = data.get('message', 'No message received')
        print(f"Received message: {message}")
        return web.Response(text=f"Message received: {message}")
    else:
        return web.Response(text="Please send a POST request.")

async def init_app():
    app = web.Application()
    app.router.add_post('/', handle)
    return app

async def main():
    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    print("Server started at http://localhost:8080")
    await asyncio.Event().wait()

if __name__ == '__main__':
    asyncio.run(main())
