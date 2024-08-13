from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import PretrainedConfig

from petals.data_structures import InferenceMetadata
from petals.server.memory_cache import MemoryCache
from petals.server.task_pool import PrioritizedTaskPool
from petals.utils.misc import get_size_in_bytes, is_dummy
from petals.server.req_tensor_descr import NewTensorDescriptor

logger = get_logger(__name__)

class AttentionAllocInfo:
    def __init__(self, key_handle, value_handle) -> None:
        self.key_handle = key_handle
        self.value_handle = value_handle

class TransformerBackend(ModuleBackend):
    """A wrapper for a transformer block that can process requests for forward, backward and inference"""

    _peft_module = None

    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        memory_cache: MemoryCache,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        import petals.utils.peft as _peft_module

        self._peft_module = _peft_module

        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.memory_cache = memory_cache
        self.max_chunk_size_bytes = max_chunk_size_bytes

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )  # note: inference_pools may be merged later, see merge_inference_pools_inplace
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        for shard in self.module.module_shards:
            for submodule in shard.modules():
                if isinstance(submodule, config.attn_class):
                    self.shard_num_heads.append(submodule.num_heads)
        assert len(self.shard_num_heads) == len(self.module.devices)
        assert sum(self.shard_num_heads) == config.num_attention_heads

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * get_size_in_bytes(descr.dtype)
        
        self.attention_alloc_table:Dict[id, AttentionAllocInfo] = {}


    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache_tensors = []
        for device, num_heads in zip(self.module.devices, self.shard_num_heads):
            num_heads //= self.config.num_key_value_groups
            if hasattr(self.config, "num_key_value_heads"):
                num_heads = self.config.num_key_value_heads
            keys = TensorDescriptor((batch_size, num_heads, head_dim, max_length), dtype=self.dtype, device=device)
            values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.extend((keys, values))
        return cache_tensors
    
    def get_inference_cache_descriptors_req(self, cached_request, iteration_info, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache_tensors = []
        for device, num_heads in zip(self.module.devices, self.shard_num_heads):
            num_heads //= self.config.num_key_value_groups
            if hasattr(self.config, "num_key_value_heads"):
                num_heads = self.config.num_key_value_heads
            # return a list of NewTensorDescriptor, each is corresponding to
            # a request or namely a single prompt
            batch_size = iteration_info['batch_size']
            request_id_table = iteration_info['request_id_table']
            max_length = 20
            keys = []
            values = []
            for i in range(batch_size):
                if request_id_table[i] in cached_request:
                    continue
                keys.append(NewTensorDescriptor(request_id_table[i], (num_heads, head_dim, max_length), dtype=self.dtype, device=device))
                values.append(NewTensorDescriptor(request_id_table[i], (num_heads, max_length, head_dim), dtype=self.dtype, device=device))
            assert len(keys) == len(values)
            #keys = [NewTensorDescriptor(request_id_table[i], (num_heads, head_dim, max_length), dtype=self.dtype, device=device) for i in range(batch_size)]
            #values = [NewTensorDescriptor(request_id_table[i], (num_heads, max_length, head_dim), dtype=self.dtype, device=device) for i in range(batch_size)]
            cache_tensors.extend((keys, values))
        return cache_tensors

    def forward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().forward(*inputs)

    def backward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().backward(*inputs)

    def update_alloc_table(self, descriptor, cache_handles):
        key_descriptors = descriptor[0]
        req_key_handles = cache_handles[0]
        req_value_handles = cache_handles[1]
        handle_idx = 0
        for req_key_handle, req_value_handle in zip(req_key_handles, req_value_handles):
            info = AttentionAllocInfo(req_key_handle, req_value_handle)
            request_id = key_descriptors[handle_idx].request_id
            self.attention_alloc_table.update({request_id:info})
            handle_idx += 1

    @torch.inference_mode()
    def inference_step(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_info: InferenceMetadata,
        iteration_info
    ) -> Tuple[torch.Tensor, ...]:
        """
        This is a modified version of inference_step()
        """
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"
        seq_len = hidden_states.shape[1]

        input_length_table = iteration_info['input_length_table']

        finished_request_id_table = iteration_info['finished_request_id_table']
        if len(finished_request_id_table) > 0:
            handles_to_reclaim = []
            for finished_request_id in finished_request_id_table:
                info = self.attention_alloc_table[finished_request_id]
                handles_to_reclaim.append(info.key_handle)
                handles_to_reclaim.append(info.value_handle)
                del self.attention_alloc_table[finished_request_id]
            self.memory_cache.reclaim_cache_for_finished_requests(handles_to_reclaim)
        self.update_alloc_table(inference_info.descriptor, inference_info.cache_handles)
        self.memory_cache.alloc_cache_for_new_requests()
        key_handles = []
        value_handles = []
        for id in iteration_info['request_id_table']:
            alloc_info = self.attention_alloc_table[id]
            key_handles.append(alloc_info.key_handle)
            value_handles.append(alloc_info.value_handle)
        
        # the structure of cache_handlers:
        # it's a list of two tuples: key_handles and value_handles,
        # each element in key_/value_handles is corresponding to a request,
        # have the same order as requests in the batch
        cache_handles = [tuple(key_handles), tuple(value_handles)]

        with self.memory_cache.use_cache_new(
            cache_handles
        ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter):
            # here cache_tensors is a tuple of two list
            # the first list contains key cache for each request
            # the second list contains value cache for each request

            #self._reorder_cache_inplace_new(cache_tensors, hypo_ids)

            # make a change here, for each request prompt
            # we would like to ban the chunk method temporarily
            # since we will only test with short sequences
            # it will be reserved for later version
            
            # it's a vLLM-like implementation
            prefix_length_table = iteration_info['prefix_length_table']
            layer_past = self._select_layer_past_new(cache_tensors, prefix_length_table)

            output_hidden_states, new_kvs = self.module.forward(
                    hidden_states, layer_past=layer_past, use_cache=True, iteration_info = iteration_info
                )


            self._update_cache_inplace_new(cache_tensors, new_kvs, prefix_length_table)
            # re-concat output hidden states of each request at dimention 1 (seq length)
            # note that ouput hidden states will have the same shape as input hidden states
            return (output_hidden_states,)
        

    @torch.inference_mode()
    def inference_step_serial(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_info: InferenceMetadata,
        iteration_info
    ) -> Tuple[torch.Tensor, ...]:
        """
        This is a modified version of inference_step()
        """
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"
        seq_len = hidden_states.shape[1]
            
        input_length_table = iteration_info['input_length_table']

        self.update_alloc_table(inference_info.descriptor, inference_info.cache_handles)
        key_handles = []
        value_handles = []
        for id in iteration_info['request_id_table']:
            alloc_info = self.attention_alloc_table[id]
            key_handles.append(alloc_info.key_handle)
            value_handles.append(alloc_info.value_handle)
        
        # the structure of cache_handlers:
        # it's a list of two tuples: key_handles and value_handles,
        # each element in key_/value_handles is corresponding to a request,
        # have the same order as requests in the batch
        cache_handles = [tuple(key_handles), tuple(value_handles)]
        with self.memory_cache.use_cache_new(
            cache_handles
        ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter):
            # here cache_tensors is a tuple of two list
            # the first list contains key cache for each request
            # the second list contains value cache for each request

            #self._reorder_cache_inplace_new(cache_tensors, hypo_ids)

            # make a change here, for each request prompt
            # we would like to ban the chunk method temporarily
            # since we will only test with short sequences
            # it will be reserved for later version
            
            # it's a simple serial implementation, running 
            # self.module.forward() with a SINGLE request at a time.
            # it's easy to achieve since no inside modification to
            # module.forward() is needed. But it might not support
            # static batching
            prefix_length_table = iteration_info['prefix_length_table']
            layer_past = self._select_layer_past_new(cache_tensors, prefix_length_table)
            key_layer_past = layer_past[0]
            value_layer_past = layer_past[1]
            index = 0
            output_hidden_states_list = []
            new_key_list = []
            new_value_list = []
            batch_size = iteration_info['batch_size']
            #length_table = iteration_info['input_length_table']
            token_num_table = iteration_info['iter_token_num_table']
            token_num_idx = 0
            for i in range(batch_size):
                req_hidden_states = hidden_states[:, token_num_idx:token_num_idx+token_num_table[i], :]
                token_num_idx += token_num_table[i]
                req_layer_past = tuple([key_layer_past[i], value_layer_past[i]])
                req_output_hidden_states, req_new_kvs = self.module.forward(
                    req_hidden_states, layer_past=req_layer_past, use_cache=True, iteration_info = iteration_info
                )
                output_hidden_states_list.append(req_output_hidden_states)

                req_layer_past = req_new_kvs
                new_key_list.append(req_new_kvs[0])
                new_value_list.append(req_new_kvs[1])

            new_kvs = tuple([new_key_list, new_value_list])
            
            self._update_cache_inplace_new(cache_tensors, new_kvs, prefix_length_table)
            # re-concat output hidden states of each request at dimention 1 (seq length)
            # note that ouput hidden states will have the same shape as input hidden states
            output_hidden_states = torch.concat(output_hidden_states_list, dim = 1)
            return (output_hidden_states,)

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory, given that
        # the model uses multi-query attention
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids

    def _reorder_cache_inplace_new(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """This is a modified version of _reorder_cache_inplace()"""
        key_cache_tensors = cache_tensors[0]
        value_cache_tensors = cache_tensors[1]
        if not is_dummy(hypo_ids):
            for cache_tensor in key_cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids
            for cache_tensor in value_cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids

    def _select_layer_past(self, cache_tensors: Sequence[torch.Tensor], prefix_length: int) -> Sequence[torch.Tensor]:
        """Extract first {prefix_length} tokens and reshape them such that they can be used as layer_past"""

        key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])
        for i in range(len(key_cache)):
            key_cache[i] = key_cache[i].flatten(0, 1)[:, :, :prefix_length]
            # shape: [batch * num_kv_heads, head_dim, kv_length]
            value_cache[i] = value_cache[i].flatten(0, 1)[:, :prefix_length]
            # shape: [batch * num_kv_heads, kv_length, head_dim]
        layer_past = tuple(chain(*zip(key_cache, value_cache)))
        return PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past
    
    def _select_layer_past_new(self, cache_tensors: Sequence[torch.Tensor], prefix_length) -> Sequence[torch.Tensor]:
        """This is a modified version of _select_layer_past()"""
        # different from the original _select_layer_past()
        # each element in cache_tensors here is a list of tensors
        key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])
        layer_past_key = []
        layer_past_value = []
        # prefix_length is actually input_length_table
        # where the i-th element is the prefix length of the i-th request
        for i in range(len(key_cache)):
            idx = 0
            for req_key_cache in key_cache[i]:
                req_key_cache = req_key_cache[:, :, :prefix_length[idx]]
                layer_past_key.append(req_key_cache)
                idx += 1
            # shape: [num_kv_heads, head_dim, kv_length]
            idx = 0
            for req_value_cache in value_cache[i]:
                req_value_cache = req_value_cache[:, :prefix_length[idx]]
                layer_past_value.append(req_value_cache)
                idx += 1
            # shape: [num_kv_heads, kv_length, head_dim]
        layer_past = tuple([layer_past_key, layer_past_value])
        return PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past

    def _update_cache_inplace(
        self, cache_tensors: Sequence[torch.Tensor], new_kvs: Sequence[torch.Tensor], prefix_length
    ):
        """Writes new key/value tensors back into cache, works in-place"""
        _batch_size_times_num_kv_heads, head_dim, new_length = new_kvs[0].shape
        for cache_key, new_key in zip(cache_tensors[0::2], new_kvs[0::2]):
            new_key = new_key.view(*cache_key.shape[:3], new_length)
            cache_key[:, :, :, prefix_length:new_length] = new_key[:, :, :, prefix_length:new_length]
        for cache_value, new_value in zip(cache_tensors[1::2], new_kvs[1::2]):
            new_value = new_value.view(*cache_value.shape[:2], new_length, head_dim)
            cache_value[:, :, prefix_length:new_length, :] = new_value[:, :, prefix_length:new_length, :]

    def _update_cache_inplace_new(
        self, cache_tensors: Sequence[torch.Tensor], new_kvs: Sequence[torch.Tensor], prefix_length
    ):
        """Writes new key/value tensors back into cache, works in-place"""
        for cache_key, new_key in zip(cache_tensors[0::2], new_kvs[0::2]):
            idx = 0
            for req_cache_key, req_new_key in zip(cache_key, new_key):
                '''
                num_kv_heads, head_dim, new_length = req_new_key.shape
                req_new_key = req_new_key.view(req_cache_key.shape[:2], new_length)
                req_cache_key[:, :, prefix_length:new_length] = new_key[:, :, prefix_length:new_length]
                '''
                original_length = req_cache_key.shape[2]
                new_length = req_new_key.shape[2]
                temp_tensor = req_new_key[:, :, original_length:new_length]
                req_cache_key[:, :, prefix_length[idx]:new_length] = req_new_key[:, :, prefix_length[idx]:new_length]
                idx += 1
        for cache_value, new_value in zip(cache_tensors[1::2], new_kvs[1::2]):
            idx = 0
            for req_cache_value, req_new_value in zip(cache_value, new_value):
                '''
                num_kv_heads, head_dim, new_length = req_new_value.shape
                req_new_value = req_new_value.view(req_cache_value.shape[:1], new_length, head_dim)
                cache_value[:, prefix_length:new_length, :] = new_value[:, prefix_length:new_length, :]
                '''
                original_length = req_cache_value.shape[1]
                new_length = req_new_value.shape[1]
                temp_tensor = req_new_value[:, original_length:new_length, :]
                #req_cache_value = torch.concat((req_cache_value, temp_tensor), dim = 1)
                req_cache_value[:, prefix_length[idx]:new_length, :] = req_new_value[:, prefix_length[idx]:new_length, :]
                idx += 1

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy


def merge_inference_pools_inplace(backends: Dict[ExpertUID, TransformerBackend]):
    """Replace each backend's rpc_inference pools with a combined pool runs multiple blocks in one call"""
    assert len(backends) != 0 and all(isinstance(b, TransformerBackend) for b in backends.values())
    first_pool = next(iter(backends.values())).inference_pool
    merged_pool = PrioritizedTaskPool(
        _MergedInferenceStep(backends),
        max_batch_size=first_pool.max_batch_size,
        device=first_pool.device,
        name=f"merged_inference",
    )
    for backend in backends.values():
        assert not backend.inference_pool.is_alive()
        backend.inference_pool = merged_pool


class _MergedInferenceStep:
    def __init__(self, backends: Dict[ExpertUID, TransformerBackend]):
        self.backends = backends

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_infos: Sequence[InferenceMetadata],
        iteration_info,
        *optional_prompts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        assert len(inference_infos) == len(
            optional_prompts
        ), f"found {len(inference_infos)} blocks but {len(optional_prompts)} prompts"
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            (hidden_states,) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info, iteration_info)
        return (hidden_states,)
