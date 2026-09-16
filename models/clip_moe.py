import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPMLP

from models.svd_moe import SVDMoeLinear

class ViTMoEAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.k_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.v_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.out_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_outputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states, gating_outputs) * self.scale
        key_states = self.k_proj(hidden_states, gating_outputs)
        value_states = self.v_proj(hidden_states, gating_outputs)

        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights_reshaped = None
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output, gating_outputs)

        return attn_output, attn_weights_reshaped

    def forward_main_only(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj.forward_main_only(hidden_states) * self.scale
        key_states = self.k_proj.forward_main_only(hidden_states)
        value_states = self.v_proj.forward_main_only(hidden_states)

        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj.forward_main_only(attn_output)

        return attn_output


class ViTMoELayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int):
        super().__init__()
        self.self_attn = ViTMoEAttention(config, num_experts, r_main, rank_per_expert, artifact_expert_idx)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, gating_outputs: Dict[str, torch.Tensor], attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)


        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            gating_outputs=gating_outputs,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )


        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states


        return (hidden_states, attn_weights)

    def forward_main_only(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.self_attn.forward_main_only(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
