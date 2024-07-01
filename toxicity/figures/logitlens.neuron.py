"""
Neuron Figure.
"""

import os
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from fancy_einsum import einsum
from transformer_lens import HookedTransformer
from toxicity.figures.fig_utils import convert, load_hooked
from constants import ROOT_DIR, MODEL_DIR, DEVICE

from transformer_lens.utils import Slice

BATCHSIZE = 4
DPO_DIR = os.path.join(MODEL_DIR, "dpo.pt")
PROMPT_PATH = os.path.join(ROOT_DIR, "toxicity/figures/shit_prompts.npy")
DEVICE = "cuda"
TOKEN_ID = 7510 # "shit" token


def get_token_prob_for_layer(model, batch, token_id, layer=0, device=None, verbose=0):
    """
    Get all neuron probabilities in a layer for a list of instances and a certain token.

    :param model HookedTransformer: The model outputing the probabilities
    :param batch torch.tensor: The tokens
    :param token_id int: The index of the token for which the probability is calculated
    :param layer int: Layer index
    :param verbose int: Verbosity level. 0 is no printing, 1 is printing.
    :returns token_probs torch.tensor: A tensor of all probabilities of the token within the layer
    """
    with torch.inference_mode():
        if verbose > 0:
            print('CREATING CACHE')
        _, cache = model.run_with_cache(batch, device=device)
    
        if verbose > 0:
            print('DECOMPOSE')
        decom = cache.get_neuron_results(layer=layer, neuron_slice=Slice(), pos_slice=Slice())

        if verbose > 0:
            print('PROJECTION')
        # Project each layer and each position onto vocab space
        vocab_proj = einsum(
            "batch pos neuron d_model, d_model d_vocab --> neuron batch pos d_vocab",
            decom,
            model.W_U,
        )
    
    if verbose > 0:
        print('PROBS')
    token_probs = vocab_proj.softmax(dim=-1)[:, :, -1, token_id]

    return token_probs

def prompts_to_tokens(model, prompts_path):
    prompts = list(
        np.load(os.path.join(prompts_path))
    )
    tokens = model.to_tokens(prompts, prepend_bos=True)

    return tokens

def load_model_with_device(model_name, weights_path=None, device="cpu"):
    # Load and configure model.
    model = load_hooked(model_name, weights_path, device=device)
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    model.cfg.device = device

    return model


if __name__ == "__main__":
    model = load_model_with_device("gpt2-medium", DPO_DIR, DEVICE)
    tokens = prompts_to_tokens(model, PROMPT_PATH)

    all_dpo_prob = None
    for idx in tqdm(range(0, tokens.shape[0], BATCHSIZE)):
        batch = tokens[idx : idx + BATCHSIZE].to(DEVICE)
        # TODO: not just one layer
        shit_probs = get_token_prob_for_layer(model, batch, TOKEN_ID, layer=0, verbose=True)

        if all_dpo_prob is None:
            all_dpo_prob = shit_probs
        else:
            all_dpo_prob = torch.concat([all_dpo_prob, shit_probs], dim=1)