import torch
import torch.nn as nn
from transformers import ViTModel
from .custom_attention import MemoryViTSelfAttention


class VIT(nn.Module):
    def __init__(self, config_model, models_path=None, pretrained=False, cross_attn_layers=None):
        super().__init__()
        if pretrained:
            self.model = ViTModel.from_pretrained(models_path, add_pooling_layer=False)
        else:
            self.model = ViTModel(config_model, add_pooling_layer=False)
        self.embed = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        self.hidden_size = self.model.config.hidden_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.cross_attn = cross_attn_layers is not None
        if self.cross_attn:
            if cross_attn_layers == "all":
                target_indices = list(range(len(self.encoder.layer)))
            elif cross_attn_layers == "set":
                target_indices = [1, 6, 11]
            else:
                raise ValueError("cross_attn_layers must be 'set' or 'all'")
            invalid_indices = [i for i in target_indices if i >= len(self.encoder.layer)]
            if invalid_indices:
                raise ValueError(f"Cross-attention layer indices {invalid_indices} exceed model depth")
            self.cross_attention_blocks = nn.ModuleDict({
                f"layer_{i}": ViTCrossAttention(
                    hidden_size=self.hidden_size,
                    num_heads=self.model.config.num_attention_heads,
                )
                for i in target_indices
            })
        for layer in self.encoder.layer:
            old_attention = layer.attention.attention
            attention = MemoryViTSelfAttention(self.model.config)
            attention.load_state_dict(old_attention.state_dict())
            layer.attention.attention = attention

    def forward(self, x, token_embed=None, drop_image=False, deep_prompt=False):
        if token_embed is None:
            for layer in self.encoder.layer:
                layer.attention.attention.has_memory_token = False
            return self.model(pixel_values=x).last_hidden_state[:, 0]
        for layer in self.encoder.layer:
            layer.attention.attention.has_memory_token = True
        x = self.embed(x)
        if token_embed.size(0) > x.size(0):
            # Evaluate each neighbour token against the same image.
            k = token_embed.size(0) // x.size(0)
            x = x.unsqueeze(1).expand(-1, k, -1, -1).reshape(token_embed.size(0), -1, x.size(-1))
        if drop_image:
            patches = self.mask_token.expand(x.size(0), x.size(1) - 1, -1)
            x = torch.cat([x[:, :1], patches], dim=1)
        h = x if self.cross_attn else torch.cat([x[:, :1], token_embed, x[:, 1:]], dim=1)
        for i, layer in enumerate(self.encoder.layer):
            x = layer(h, output_attentions=False)[0]
            if self.cross_attn:
                key = f"layer_{i}"
                h = self.cross_attention_blocks[key](x, token_embed) if key in self.cross_attention_blocks else x
            elif deep_prompt:
                h = torch.cat([x[:, :1], token_embed, x[:, 1 + token_embed.size(1):]], dim=1)
            else:
                h = x
        return self.layernorm(h)[:, 0]


class TokenMemory(nn.Module):
    def __init__(self, num_instances: int, token_dim: int, vit_dim: int):
        super().__init__()
        self.table = nn.Embedding(num_instances, token_dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.adapter = nn.Linear(token_dim, vit_dim)  # map to ViT embed dim
        self.null_token = nn.Parameter(torch.randn(1, token_dim) * 0.02) # NULL token for pathway dropout

    def get_null_tokens(self, batch_size: int) -> torch.Tensor:
        """Return NULL token for pathway dropout"""
        t = self.null_token.expand(batch_size, -1)  # [B, d_t]
        t = self.adapter(t).unsqueeze(1)  # [B, 1, D]
        return t

    def get_tokens(self, ids: torch.LongTensor) -> torch.Tensor:
        t = self.table(ids)  # [B, d_t]
        t = self.adapter(t).unsqueeze(1)  # [B, 1, D] → prepend to sequence
        return t

    @torch.no_grad()
    def get_raw(self, ids: torch.LongTensor) -> torch.Tensor:
        return self.table(ids)


class ViTCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_k = nn.LayerNorm(hidden_size)

    def forward(self, x, context):
        q = self.norm_q(x)
        k = self.norm_k(context)
        attn_out, _ = self.mha(query=q, key=k, value=k, need_weights=False)
        return x + attn_out
