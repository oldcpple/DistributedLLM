import asyncio
from aiohttp import web
from scheduler import Scheduler
import torch
import time
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


    # It is a http server that collects requests(prompts) 
    # from clients and then sends to the scheduler
    # it's also responsible for sending the final results back to clients

    # it will start a http server at localhost:8080
    
class Http_server:

    def __init__(self, model_name):
        self.initial_peers = ['/ip4/192.168.130.241/tcp/46617/p2p/12D3KooWA94ramnXLtGMHovogPWUQGLPZ3k6Ds5iDfwFTRUWZwa5']
        self.scheduler = None
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
        self.model = AutoDistributedModelForCausalLM.from_pretrained(model_name, initial_peers = self.initial_peers,\
                                                                      torch_dtype=torch.float32)

    def encode(self, prompt):
        pass

    def decode(self, prompt):
        pass

    def collect_prompt(self, prompt, scheduler: Scheduler):
        inputs = self.encode(prompt)
        timestamp = int(time.time())
        scheduler.add_request(inputs, timestamp)

    async def handle(self, request):
        data = await request.json()
        text = data.get('text')
        queue_id = id(request)

        wrapped_data = {'text': text}

        self.scheduler.response_queues[queue_id] = asyncio.Queue()
        # route to scheduler
        #await self.scheduler.data_queue.put((wrapped_data, queue_id))
        await self.scheduler.collect_prompts_new(wrapped_data, queue_id)

        result = await self.scheduler.response_queues[queue_id].get()
        result_text, result_length = result
        print(result_text)
        return web.json_response({'result': result_text, 'length': result_length})

    async def init_app(self):
        app = web.Application()
        app.router.add_post('/', self.handle)
        app.router.add_get('/', lambda request: web.Response(text="Please send a POST request."))
        return app

    async def main(self):
        app = await self.init_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        print("Server started at http://0.0.0.0:8080")
        await asyncio.Event().wait()
    
    def server_start(self):
        self.scheduler = Scheduler(self.model, self.tokenizer, None)
        loop = asyncio.get_event_loop()
        #loop.create_task(self.scheduler.collect_prompts())
        loop.create_task(self.scheduler.scheduler_main())
        loop.run_until_complete(self.main())

if __name__ == '__main__':
    server = Http_server('bigscience/bloom-7b1')
    server.server_start()

    
    