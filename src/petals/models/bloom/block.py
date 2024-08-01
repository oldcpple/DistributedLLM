"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomModel, build_alibi_tensor

from petals.utils.misc import is_dummy
import numpy as np


class WrappedBloomBlock(BloomBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        iteration_info = None,
        **kwargs
    ):
        assert attention_mask is None, "Non-causal attention masks are not supported yet"
        batch_size, seq_length = hidden_states.shape[:2]
        '''
        if layer_past is not None and is_dummy(layer_past[0]):
            # Bloom cannot use cache if it was misconsctructed(e.g. Dummy tensors)
            # In this case, fallback to the old code:
            layer_past = None
        '''

        # following are actually computing the total length (seq_length + prefix_length)
        # in our case it equals to the sum of elements in input_length_table
        # (for all prompts in the batch since we consider them all together)
        #past_length = 0 if layer_past is None else layer_past[0].shape[-1]
        #seq_length_with_past = seq_length + past_length

        input_length_table = iteration_info['input_length_table']
        attention_mask_total_seq_length = np.sum(input_length_table)
        attention_mask = torch.ones((1, attention_mask_total_seq_length), device=hidden_states.device)
        if alibi is None:
            alibi = build_alibi_tensor(attention_mask, num_heads=self.num_heads, dtype=hidden_states.dtype)
        print(alibi.shape)
        # it returns a 4D mask of shape `(batch_size, 1, query_length, key_value_length)
        # at prefill, query_length of each prompts are different, while all past lengths are 0
        # so it equals to concating all attention masks at dim = 2
        stage = iteration_info['stage']
        if stage == 'prefill':
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=(1, attention_mask_total_seq_length),
                inputs_embeds=hidden_states,
                past_key_values_length=0,
            )
        # at decode, query_length of each prompts are the same 1, while past lengths varies
        # so it equals to concating all attention masks at dim = 3
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=(1, 1),
                inputs_embeds=hidden_states,
                past_key_values_length=attention_mask_total_seq_length,
            )
        attention_mask = attention_mask.bool()
        return super().forward(
            hidden_states, *args, attention_mask=attention_mask, alibi=alibi, layer_past=layer_past, iteration_info = iteration_info, **kwargs
        )
