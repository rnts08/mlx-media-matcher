# Vendored (trimmed) from mlx-examples/clip/model.py — Copyright © 2023-2024
# Apple Inc., MIT License. Adapted to load mlx-community/*.npz checkpoints
# (key remap + npz loader instead of safetensors).

import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass
class CLIPTextConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    max_position_embeddings: int
    vocab_size: int
    layer_norm_eps: float


@dataclass
class CLIPVisionConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    layer_norm_eps: float


@dataclass
class CLIPConfig:
    text_config: CLIPTextConfig
    vision_config: CLIPVisionConfig
    projection_dim: int


def quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int, bias: bool = False):
        super().__init__()
        if dims % num_heads != 0:
            raise ValueError(f"({dims} % {num_heads}) != 0")
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat)


class MLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.activation_fn = quick_gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.activation_fn(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask=None) -> mx.array:
        y = self.layer_norm1(x)
        y = self.self_attn(y, y, y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class TextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        embeddings = self.token_embedding(x)
        embeddings += self.position_embedding.weight[: x.shape[1]]
        return embeddings


class CLIPTextModel(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embeddings = TextEmbeddings(config)
        self.encoder = Encoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, N = x.shape
        eot_tokens = mx.argmax(x, axis=-1)
        x = self.embeddings(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N, x.dtype)
        for layer in self.encoder.layers:
            x = layer(x, mask)
        x = self.final_layer_norm(x)
        return x[mx.arange(B), eot_tokens]


class VisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        self.class_embedding = mx.zeros((config.hidden_size,))
        self.position_embedding = nn.Embedding(num_patches + 1, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        cls_embeddings = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, embed_dim)
        )
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        embeddings += self.position_embedding.weight
        return embeddings


class CLIPVisionModel(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pre_layernorm(self.embeddings(x))
        for layer in self.encoder.layers:
            x = layer(x, mask=None)
        return self.post_layernorm(x[:, 0, :])


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.text_model = CLIPTextModel(config.text_config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, config.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, config.projection_dim, bias=False
        )
        self.logit_scale = mx.array(0.0)

    def get_text_features(self, x: mx.array) -> mx.array:
        return self.text_projection(self.text_model(x))

    def get_image_features(self, x: mx.array) -> mx.array:
        return self.visual_projection(self.vision_model(x))

    @staticmethod
    def from_pretrained(path: str) -> "CLIPModel":
        path = Path(path)
        with open(path / "config.json") as fid:
            cfg = json.load(fid)

        tc, vc = cfg["text_config"], cfg["vision_config"]
        config = CLIPConfig(
            text_config=CLIPTextConfig(
                num_hidden_layers=tc["num_hidden_layers"],
                hidden_size=tc["hidden_size"],
                intermediate_size=tc["intermediate_size"],
                num_attention_heads=tc["num_attention_heads"],
                max_position_embeddings=tc["max_position_embeddings"],
                vocab_size=tc["vocab_size"],
                layer_norm_eps=tc["layer_norm_eps"],
            ),
            vision_config=CLIPVisionConfig(
                num_hidden_layers=vc["num_hidden_layers"],
                hidden_size=vc["hidden_size"],
                intermediate_size=vc["intermediate_size"],
                num_attention_heads=vc["num_attention_heads"],
                num_channels=3,
                image_size=vc["image_size"],
                patch_size=vc["patch_size"],
                layer_norm_eps=vc["layer_norm_eps"],
            ),
            projection_dim=cfg["projection_dim"],
        )
        model = CLIPModel(config)

        weight_files = sorted(glob.glob(str(path / "*.safetensors")))
        if weight_files:
            weights = {}
            for wf in weight_files:
                weights.update(mx.load(wf))
        elif (path / "weights.npz").exists():
            npz = mx.load(str(path / "weights.npz"))
            weights = dict(npz)
        else:
            raise FileNotFoundError(f"No weights found in {path}")

        weights = _remap_weights(weights)
        model.load_weights(list(weights.items()))
        model.eval()
        return model


_REMAP = [
    (re.compile(r"^(\w+_model)\.layers\.(\d+)\."), r"\1.encoder.layers.\2."),
    (re.compile(r"\.attention\."), ".self_attn."),
    (re.compile(r"\.query_proj\."), ".q_proj."),
    (re.compile(r"\.key_proj\."), ".k_proj."),
    (re.compile(r"\.value_proj\."), ".v_proj."),
    (re.compile(r"\.linear1\."), ".mlp.fc1."),
    (re.compile(r"\.linear2\."), ".mlp.fc2."),
    (re.compile(r"\.ln1\."), ".layer_norm1."),
    (re.compile(r"\.ln2\."), ".layer_norm2."),
    (re.compile(r"^(text|vision)_model\.token_embedding"), r"\1_model.embeddings.token_embedding"),
    (re.compile(r"^(text|vision)_model\.position_embedding$"), r"\1_model.embeddings.position_embedding.weight"),
    (re.compile(r"^vision_model\.class_embedding"), "vision_model.embeddings.class_embedding"),
    (re.compile(r"^vision_model\.patch_embedding"), "vision_model.embeddings.patch_embedding"),
]


def _remap_weights(weights: dict) -> dict:
    """Map checkpoint key names onto this module's parameter names."""
    out = {}
    for key, value in weights.items():
        if "position_ids" in key:
            continue
        for pattern, replacement in _REMAP:
            key = pattern.sub(replacement, key)
        out[key] = value
    return out
