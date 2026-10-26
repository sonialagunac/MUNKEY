import torch
from typing import Optional
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTSelfAttention

class MemoryViTSelfAttention(ViTSelfAttention):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.has_memory_token = True

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # Standard ViT query, key, and value projections
        key_layer   = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Use the eager attention path for consistent token insertion.

        # scaled dot product attention
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores * self.scaling

        # no attention mask used in plain ViT
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # dropout on probs
        if self.training and self.dropout_prob > 0.0:
            attn_probs = torch.nn.functional.dropout(
                attn_probs, p=self.dropout_prob, training=True
            )

        # optional head mask
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        # Preserve the normalization used by the original memory-token path.
        if self.has_memory_token:
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # context
        context_layer = torch.matmul(attn_probs, value_layer)  # [B, H, L, d]
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_shape)

        outputs = (context_layer, attn_probs) if output_attentions else (context_layer,)
        return outputs
