import queue
import asyncio
import time
import torch

'''
data structure:
iteration_info = {
        'batch_size': #interger: batch size of current batch,
        'input_length_table': #list: length of each request prompts in the batch,
        'iter_token_num_table': #list: num of input tokens in the following iteration of each request prompts in the batch,
        'prefix_length_table': #list: prefix length of each request prompts in the batch,,
        'request_id_table': #list: request id of each requests in the batch,
        'stage': #str: 'prefill' or 'decode'
    }
'''

class RequestDataStructure:
    def __init__(self, content, queue_id: id, priority, phase = 'prefill', max_new_tokens = 20):
        # content can be str (at prefilling) or tensors (at decoding)
        self.content = content
        self.queue_id = queue_id
        self.priority = priority
        self.phase = 'prefill'
        self.max_new_tokens = max_new_tokens
        # for testing
        self.max_new_tokens = 20
        self.token_idx = 0
        self.new_tokens = 0

class Scheduler:

    def __init__(self, model, tokenizer, session) -> None:
        self.requests_pool = asyncio.PriorityQueue()
        self.max_batch_size = 10
        self.data_queue = asyncio.Queue()
        self.response_queues = {}
        self.inputs_queue = asyncio.Queue()
        self.finished_queue = asyncio.Queue()
        self.priority_base = 1721191179
        self.tokenizer = tokenizer
        self.model = model
        self.session = session
        self.last_batch_size = 0
    
    async def collect_prompts(self):
        while True:
            data, queue_id = await self.data_queue.get()
            await self.add_request(data, queue_id)

    async def collect_prompts_new(self, data, queue_id):
        await self.add_request(data, queue_id)

    async def add_request(self, data, queue_id):
        # FCFS
        timestamp = time.time()
        content = data['text']
        # to make sure that requests at prefilling stage
        # are always have higher priority than those at decoding
        content = self.tokenizer(content, return_tensors='pt')["input_ids"]
        priority = timestamp - self.priority_base
        request = RequestDataStructure(content, queue_id, priority)
        await self.requests_pool.put((priority, request))

    # main func of Scheduler, selecting requests from requests_pool
    # and batching them, sending to remote server for inference
    async def scheduler_main(self):
        while True:
            await asyncio.sleep(0.01)
            
            finished_request_id_table = []
            for i in range(self.last_batch_size):
                if self.finished_queue.empty():
                    break
                finished_request_id = await self.finished_queue.get()
                finished_request_id_table.append(finished_request_id)
            
            current_batch = []
            # this should select the request with highest priority
            # and requests at PREFILL should always have higher priority
            # than those at DECODE
            first_priority, first_request = await self.requests_pool.get()
            # the rest of selected requests should be at the same
            # phase as the first request, either prefill or decode
            batch_phase = first_request.phase
            current_batch.append(first_request)
            for i in range(self.max_batch_size):
                if self.requests_pool.empty():
                    break
                priority, request = await self.requests_pool.get()
                # to keep all requests in a batch to be at the same phase
                if request.phase != batch_phase:
                    await self.requests_pool.put((priority, request))
                    break
                current_batch.append(request)

            self.last_batch_size = len(current_batch)

            # tokenize and batch all request prompts
            input_ids_list = []
            input_length_table = []
            iter_token_num_table = []
            prefix_length_table = []
            request_id_table = []
            max_new_tokens_table = []
            new_tokens_table = []

            # for batch at prefill
            if batch_phase == 'prefill':
                for request in current_batch:
                    input_ids = request.content
                    queue_id = request.queue_id
                    input_ids_list.append(input_ids)
                    input_length = input_ids.shape[1]
                    input_length_table.append(input_length)
                    # at prefill, all sequence tokens will be used for computing
                    iter_token_num_table.append(input_length)
                    prefix_length_table.append(0)
                    request_id_table.append(queue_id)
                    max_new_tokens_table.append(request.max_new_tokens)
                    new_tokens_table.append(request.new_tokens)
                # batch all request prompts at dimension 1
                batched_input = torch.concat(input_ids_list, dim=1)
                batch_size = len(current_batch)
                # create a itereation_info data structure
                iteration_info = {
                    'batch_size': batch_size,
                    'input_length_table': input_length_table,
                    'iter_token_num_table': iter_token_num_table,
                    'prefix_length_table': prefix_length_table,
                    'request_id_table': request_id_table,
                    'max_new_tokens_table': max_new_tokens_table,
                    'new_tokens_table': new_tokens_table,
                    'finished_request_id_table': finished_request_id_table,
                    'timestamp': time.time(),
                    'stage': 'prefill'
                }
                # send to LLM
                result_this_iter, info = self.model.generate(batched_input, iteration_info = iteration_info, max_new_tokens=20)
                returned_length_table = info[0]
                finished_table = info[1]
                assert len(returned_length_table) == batch_size
                assert len(finished_table) == batch_size
                outputs_list = self.decoding_batched_inputs(result_this_iter, returned_length_table)
                assert len(outputs_list) == batch_size
                # update requests in the batch
                idx = 0
                for request in current_batch:
                    request.content = outputs_list[idx]
                    # varifying if the request has finished
                    if finished_table[idx] == True:
                        final_length = request.content.shape[1]
                        output_text = self.tokenizer.decode(request.content[0])
                        await self.response_queues[request.queue_id].put((output_text, final_length))
                        await self.finished_queue.put(request.queue_id)
                        del request
                    else:
                        # from now current request is at decode stage
                        # thus it's priority changes
                        request.priority += self.priority_base
                        request.phase = 'decode'
                        request.new_tokens += 1
                        await self.requests_pool.put((request.priority, request))
                    idx += 1

            # for batch at decode
            elif batch_phase == 'decode':
                for request in current_batch:
                    input_ids = request.content
                    queue_id = request.queue_id
                    input_ids_list.append(input_ids)
                    input_length = input_ids.shape[1]
                    input_length_table.append(input_length)
                    # at decode, only new token from last iteration 
                    # will be used for computing
                    iter_token_num_table.append(1)
                    prefix_length_table.append(input_length - 1)
                    request_id_table.append(queue_id)
                    max_new_tokens_table.append(request.max_new_tokens)
                    new_tokens_table.append(request.new_tokens)
                # batch all request prompts at dimension 1
                batched_input = torch.concat(input_ids_list, dim=1)
                batch_size = len(current_batch)
                # create a itereation_info data structure
                iteration_info = {
                    'batch_size': batch_size,
                    'input_length_table': input_length_table,
                    'iter_token_num_table': iter_token_num_table,
                    'prefix_length_table': prefix_length_table,
                    'request_id_table': request_id_table,
                    'max_new_tokens_table': max_new_tokens_table,
                    'new_tokens_table': new_tokens_table,
                    'finished_request_id_table': finished_request_id_table,
                    'timestamp': time.time(),
                    'stage': 'decode'
                }
                # send to LLM
                ts = time.time()
                result_this_iter, info = self.model.generate(batched_input, iteration_info = iteration_info, max_new_tokens=20)
                te = time.time()
                print('end to end time span: {}'.format(te - ts))
                returned_length_table = info[0]
                finished_table = info[1]
                assert len(returned_length_table) == batch_size
                assert len(finished_table) == batch_size
                outputs_list = self.decoding_batched_inputs(result_this_iter, returned_length_table)
                assert len(outputs_list) == batch_size
                # update requests in the batch
                idx = 0
                for request in current_batch:
                    request.content = outputs_list[idx]
                    # varifying if the request has finished
                    if finished_table[idx] == True:
                        final_length = request.content.shape[1]
                        output_text = self.tokenizer.decode(request.content[0])
                        await self.response_queues[request.queue_id].put((output_text, final_length))
                        await self.finished_queue.put(request.queue_id)
                        del request
                    else:
                        request.new_tokens += 1
                        await self.requests_pool.put((request.priority, request))
                    idx += 1
   
    
    def decoding_batched_inputs(self, batched_outputs, length_table):
        length_idx = 0
        outputs_list = []
        for length in length_table:
            req_inputs = batched_outputs[:, length_idx:length_idx+length]
            length_idx += length
            outputs_list.append(req_inputs)
        return outputs_list

    # send prompts to remote servers for inference
    async def model_step(self):
        input_tokens, queue_id = await self.inputs_queue.get()
        outputs = self.model.generate(input_tokens, request_id = queue_id, max_new_tokens=20, session = self.session)
        return outputs



    def schedule(self):
        pass

    def scheduler_start(self):
        loop = asyncio.get_event_loop()
        loop.create_task(self.collect_prompts())