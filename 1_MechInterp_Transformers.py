#%% Chapter 1
# Imports
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Callable

import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
import wandb
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers import GPT2TokenizerFast

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part1_transformer_from_scratch"
root_dir = Path("/Users/sebastin/Documents/perso/ARENA_training/ARENA_3.0") if "QIMR" in platform.node() else Path("/home/sebastin/Documents/ARENA/ARENA_3.0") 
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_transformer_from_scratch.solutions as solutions
import part1_transformer_from_scratch.tests as tests

MAIN = __name__ == "__main__"

# %% print vocab
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
print(sorted_vocab[-20:])
# %%
lengths = dict.fromkeys(range(3, 8), "")
for tok, idx in sorted_vocab:
    if not lengths.get(len(tok), True):
        lengths[len(tok)] = tok

for length, tok in lengths.items():
    print(f"{length}: {tok}")
# %% Text generation
# 1) Convert text to tokens
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# 2) map tokens to logits
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)

# 3) map logits to probabilities
probs = logits.softmax(dim=-1)
print(probs.shape)

# 4) get the most likely next token
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))

# 5) 
print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")
for i in range(10):
    print(f"{tokens.shape[-1] + 1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)

# %%
for activation_name, activation in cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")

# %%
for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

# %%
# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures
print(reference_gpt2.cfg)




# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)

# %% Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        mu = residual.mean(dim=-1, keepdim=True)
        sigma = (residual.var(dim=-1, unbiased=False, keepdim=True) +  self.cfg.layer_norm_eps).sqrt()
        resid_normed = (residual - mu) / (sigma)
        if self.cfg.debug:
            print(f"resid_normed shape {resid_normed.shape}")
        
        return resid_normed * self.w + self.b


solutions.rand_float_test(LayerNorm, [2, 4, 768])
solutions.load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
tests.test_layer_norm_epsilon(LayerNorm, cache["resid_post", 11])

# %% Embed
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


#solutions.test_embed(Embed)
solutions.rand_int_test(Embed, [2, 4])
solutions.load_gpt2_test(Embed, reference_gpt2.embed, tokens)


# %% PosEmbed
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, pos = tokens.shape
        pos_embed = t.arange(pos).repeat(batch).reshape((batch, pos))
        return self.W_pos[pos_embed]


#solutions.test_pos_embed(PosEmbed)
solutions.rand_int_test(PosEmbed, [2, 4])
solutions.load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

# %%
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        mask = t.triu(t.ones(attn_scores.shape[-2:], device=attn_scores.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, self.IGNORE)
        return attn_scores


tests.test_causal_mask(Attention.apply_causal_mask)

# %% Attention
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        q = einops.einsum(self.W_Q, normalized_resid_pre, "n_h d_m d_h, b pos d_m -> b pos n_h d_h" ) + self.b_Q 
        k = einops.einsum(self.W_K, normalized_resid_pre, "n_h d_m d_h, b pos d_m -> b pos n_h d_h" ) + self.b_K 
        v = einops.einsum(self.W_V, normalized_resid_pre, "n_h d_m d_h, b pos d_m -> b pos n_h d_h" ) + self.b_V
        attn_scores = einops.einsum(q, k, "b posq n_h d_h, b posk n_h d_h -> b n_h posq posk")
        attn_scores /= np.sqrt(q.shape[-1])
        attn_scores = self.apply_causal_mask(attn_scores)
        attn_probs = attn_scores.softmax(dim=-1)

        z = einops.einsum(attn_probs, v, "b n_h posq posk, b posk n_h d_h -> b posq n_h d_h")
        attn_out = einops.einsum(self.W_O, z, "n_h d_h d_m, b pos n_h d_h -> b pos d_m") + self.b_O
        return attn_out


    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        mask = t.triu(t.ones(attn_scores.shape[-2:], device=attn_scores.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, self.IGNORE)
        return attn_scores


tests.test_causal_mask(Attention.apply_causal_mask)
solutions.rand_float_test(Attention, [2, 4, 768])
solutions.load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

# %% MLP
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        latent = normalized_resid_mid @ self.W_in + self.b_in
        acts = gelu_new(latent)
        out = acts @ self.W_out + self.b_out
        return out
    


solutions.rand_float_test(MLP, [2, 4, 768])
solutions.load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])

# %% Tranformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        resid = self.ln1(resid_pre)
        resid_mid = self.attn(resid) + resid_pre
        resid_mid_norm = self.ln2(resid_mid)
        resid_post = self.mlp(resid_mid_norm) + resid_mid
        return resid_post


solutions.rand_float_test(TransformerBlock, [2, 4, 768])
solutions.load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

# %% Unembed
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return normalized_resid_final @ self.W_U + self.b_U


solutions.rand_float_test(Unembed, [2, 4, 768])
solutions.load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])


# %% Demo Transformer
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        emb = self.embed(tokens)
        pos_emb = self.pos_embed(tokens)
        resid = emb + pos_emb
        for block in self.blocks:
            resid = block(resid)
        resid = self.ln_final(resid)
        return self.unembed(resid)


solutions.rand_int_test(DemoTransformer, [2, 4])
solutions.load_gpt2_test(DemoTransformer, reference_gpt2, tokens)


# %%
demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)

# %%
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens


pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# %%
test_string = """Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as"""
for i in tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)

# %%
