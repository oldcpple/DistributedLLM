import asyncio
import aiohttp

class Http_client:
    def __init__(self):
        pass

    async def send_message(self, session, url, message):
        async with session.post(url, data=message) as response:
            return await response.text()

    async def main(self, url, prompt):
        message = prompt
        async with aiohttp.ClientSession() as session:
            response_text = await self.send_message(session, url, message)
            print(response_text)
    
    def client_run(self, url, prompt):
        asyncio.run(self.main(url, prompt))


def generate_text(prompt: str):
    client = Http_client()
    url = 'http://localhost:8080/'
    client.client_run(url, prompt)


if __name__ == '__main__':
    prompt = {'text': 'how old are you?'}
    generate_text(prompt)
