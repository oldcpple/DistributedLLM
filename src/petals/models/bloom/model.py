from typing import Optional

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from petals.models.bloom.modeling_bloom import BloomForCausalLM, BloomForSequenceClassification, BloomModel, BloomPreTrainedModel

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.bloom.config import DistributedBloomConfig
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from typing import Optional, Tuple, Union
import time

logger = get_logger(__name__)


class DistributedBloomModel(FromPretrainedMixin, PTuneMixin, BloomModel):
    """BloomModel, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^h\."]

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig, *, dht: Optional[hivemind.DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.h) == 0
        config.num_hidden_layers = n_layer

        self.h = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        iteration_info = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # The causal mask will be added on the server-side
        assert (
            attention_mask is None or (attention_mask == 1).all()
        ), f"Custom attention masks are not supported, {attention_mask=}"
        assert head_mask is None, f"Custom head masks are not supported, {head_mask=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)

        hidden_states = self.h(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
            iteration_info=iteration_info,
        )

        timestamp = time.time()

        # Remove prefix
        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        b = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
        return (b, timestamp)


class DistributedBloomForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, BloomForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedBloomModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_missing += [r"^lm_head\."]  # Missing since they are shared with input embeddings
    _keys_to_ignore_on_load_unexpected = DistributedBloomModel._keys_to_ignore_on_load_unexpected
    _supports_cache_class = True

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        BloomPreTrainedModel.__init__(self, config)
        self.transformer = DistributedBloomModel(config)
        self.lm_head = LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ) -> dict:
        # we should seperately handle input_ids, attention_mask
        # and position_ids for each request in the batch
        iteration_info = kwargs.get("iteration_info")
        # following are lists
        batch_size = iteration_info['batch_size']
        past_length = iteration_info['input_length_table']
        length_this_iter = iteration_info['iter_token_num_table']

        # handle input_ids
        new_input_ids = []
        length_idx = 0
        for i in range(batch_size):
            req_input_ids = input_ids[:, length_idx + past_length[i] - length_this_iter[i]:length_idx + past_length[i]]
            new_input_ids.append(req_input_ids)
            length_idx += past_length[i]
        input_ids = torch.concat(new_input_ids, dim=1)
        assert input_ids.shape[1] == np.sum(length_this_iter)

        # handle attention_mask and position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            new_position_ids = []
            length_idx = 0
            for i in range(batch_size):
                req_attention_mask = attention_mask[:, length_idx:length_idx + past_length[i]]
                req_position_ids = req_attention_mask.long().cumsum(-1) - 1
                req_position_ids.masked_fill_(req_attention_mask == 0, 1)
                req_position_ids = req_position_ids[:, -length_this_iter[i]]
                new_position_ids.append(req_position_ids)
            position_ids = torch.concat(new_input_ids, dim=1)
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "iteration_info": kwargs.get("iteration_info")
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_bk(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ) -> dict:
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values._seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "iteration_info": kwargs.get("iteration_info")
            }
        )
        return model_inputs

    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        return past_key_values

    def get_output_embeddings(self):
        return self.lm_head


class DistributedBloomForSequenceClassification(FromPretrainedMixin, BloomForSequenceClassification):
    _keys_to_ignore_on_load_missing = DistributedBloomModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedBloomModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        BloomPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.transformer = DistributedBloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
