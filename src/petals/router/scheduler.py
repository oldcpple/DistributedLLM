import queue
import asyncio
import time
import torch

class RequestDataStructure:
    def __init__(self, content, queue_id: id, priority, phase = 0, max_new_token = 20):
        # content can be str (at prefilling) or tensors (at decoding)
        self.content = content
        self.queue_id = queue_id
        self.priority = priority
        # phase = 0: prefilling,  phase = 1: decoding
        self.phase = phase
        self.max_new_token = max_new_token
        self.token_idx = 0

class Scheduler:

    def __init__(self, model, tokenizer, session) -> None:
        self.requests_pool = asyncio.PriorityQueue()
        self.max_batch_size = 10
        self.data_queue = asyncio.Queue()
        self.response_queues = {}
        self.inputs_queue = asyncio.Queue()
        self.priority_base = 1721191179
        self.tokenizer = tokenizer
        self.model = model
        self.session = session
    
    async def collect_prompts(self):
        while True:
            data, queue_id = await self.data_queue.get()
            await self.add_request(data, queue_id)
            print(data, queue_id)

    async def add_request(self, data, queue_id):
        # FCFS
        timestamp = time.time()
        content = data['text']
        # to make sure that requests at prefilling stage
        # are always have higher priority than those at decoding
        priority = timestamp - self.priority_base
        request = RequestDataStructure(content, queue_id, priority)
        await self.requests_pool.put((priority, request))

    # main func of Scheduler, selecting requests from requests_pool
    # and batching them, sending to remote server for inference
    async def scheduler_main(self):
        while True:
            current_batch = []
            # the rest of selected requests should be at
            # the same phase as the first request
            first_priority, first_request = await self.requests_pool.get()
            batch_phase = first_request.phase
            current_batch.append(request)
            for i in range(self.max_batch_size - 1):
                if self.requests_pool.empty():
                    break
                priority, request = await self.requests_pool.get()
                if request.phase != batch_phase:
                    await self.requests_pool.put(priority, request)
                    break
                current_batch.append(request)

            # if it's at prefilling stage, prefilling of each
            # request should compute serially without batching
            if batch_phase == 0:
                for request in current_batch:
                    content = request.content
                    queue_id = request.queue_id
                    priority = request.priority + self.priority_base
                    inputs = self.tokenizer(content, return_tensors='pt')["input_ids"]
                    await self.inputs_queue.put(inputs)
                    result = await self.model_step()
                    request.content = result
                    request.token_idx += 1
                    await self.requests_pool.put((priority, request))
                    # for testing
                    output = self.tokenizer.decode(result[0])
                    await self.response_queues[queue_id].put(output)           

            # if it's at decoding stage, batch all requests
            if batch_phase == 1:
                content_list = []
                for request in current_batch:
                    content_list.append(request.content)
                await self.inputs_queue.put(content_list)
                results = await self.model_step()
                # results should be a list
                idx = 0
                for result in results:
                    request = current_batch[idx]
                    queue_id = request.queue_id
                    priority = request.priority
                    request.token_idx += 1
                    # this request has finished
                    if request.token_idx >= request.max_new_token:
                        output = self.tokenizer.decode(result)
                        await self.response_queues[queue_id].put(output)
                        del request
                    else:
                        request.content = result
                        await self.requests_pool.put((priority, request))

            # post process after an iteration
            

    # send prompts to remote servers for inference
    async def model_step(self):
        inputs = await self.inputs_queue.get()
        outputs = self.model.generate(inputs, max_new_tokens=20, session = self.session)
        return outputs



    def schedule(self):
        pass

    def scheduler_start(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.collect_prompts())