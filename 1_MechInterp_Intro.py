# %%
import functools
import sys
from pathlib import Path
import platform
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = Path("/Users/sebastin/Documents/perso/ARENA_training/ARENA_3.0") if "QIMR" in platform.node() else Path("/home/sebastin/Documents/ARENA/ARENA_3.0") 
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# %% load the model
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

# %%
gpt2_small.cfg
# %%
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)

# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_str_tokens(["gpt2", "gpt2"]))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))

# %% Tokens guessed correctly
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
print(f"logits_shape: {logits.shape}, prediction_shape: {prediction.shape}")
input_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
print(f"input_tokens_shape: {input_tokens.shape}")
print("Predictions:\n", gpt2_small.to_string(prediction))

good_inds = t.where(prediction==input_tokens)
print("Correct words:\n", gpt2_small.to_str_tokens(prediction[good_inds]))
print("Percentage correct:", f"{len(good_inds[0])} / {len(prediction)} = {len(good_inds[0]) / len(prediction):.2%}" )


# %% Cached activations
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))


# %% display name and shapes of hooked activations 
for act_name in gpt2_cache.keys():
    print(f"{act_name:30}", tuple(gpt2_cache[act_name].shape))

# %% reconstruct attention patterns from q k activations
layer0_pattern_from_cache = gpt2_cache["pattern", 0]
# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the
# steps of the attention calculation (dot product, masking, scaling, softmax)
q = gpt2_cache["q", 0]
k = gpt2_cache["k", 0]
v = gpt2_cache["v", 0]
qk = einops.einsum(q,k, "posq n_h d_h, posk n_h d_h -> n_h posq posk")
mask = t.triu(t.ones(qk.shape[-2:], device=qk.device), diagonal=1).bool()
qk_masked = qk.masked_fill(mask, -1e9)
qk_scaled = qk_masked / np.sqrt(q.shape[-1])
layer0_pattern_from_q_and_k = t.softmax(qk_scaled, dim=-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")

# %% 
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_heads(
        tokens=gpt2_str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)

# %% Text Neuron Activations
neuron_activations_for_all_layers = t.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)
# shape = (seq_pos, layers, neurons)

cv.activations.text_neuron_activations(
    tokens=gpt2_str_tokens,
    activations=neuron_activations_for_all_layers
)

# %% Top K tokens
neuron_activations_for_all_layers_rearranged = utils.to_numpy(einops.rearrange(neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons"))

cv.topk_tokens.topk_tokens(
    # Some weird indexing required here ¯\_(ツ)_/¯
    tokens=[gpt2_str_tokens],
    activations=neuron_activations_for_all_layers_rearranged,
    max_k=7,
    first_dimension_name="Layer",
    third_dimension_name="Neuron",
    first_dimension_labels=list(range(12))
)
# %%
