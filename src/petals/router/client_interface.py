import asyncio
import aiohttp
from hivemind.utils.logging import get_logger
from typing import List, Optional, Union

from src.petals.router.sampling_params import SamplingParams

logger = get_logger(__name__)

class Http_client:
    def __init__(self):
        pass

    async def send_message(self, session, url, message):
        async with session.post(url, data=message) as response:
            return await response.text()

    async def main(self, url, prompt, sampling_params):
        message = prompt
        async with aiohttp.ClientSession() as session:
            response_text = await self.send_message(session, url, message, sampling_params)
            print(response_text)
    
    def client_run(self, url, prompt, sampling_params):
        asyncio.run(self.main(url, prompt, sampling_params))


def generate_text(
        prompt: str,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
):
    client = Http_client()
    url = 'http://localhost:8080/'
    if sampling_params is None:
        # Use default sampling params.
        sampling_params = SamplingParams()
    client.client_run(url, prompt, sampling_params)


if __name__ == '__main__':
    prompt = {'text': 'how old are you?'}
    generate_text(prompt)
