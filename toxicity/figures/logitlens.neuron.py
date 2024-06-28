"""
Module Doc String
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

# TODO: remove if needed
# torch.set_default_device('cpu')


# Loading and setting up hooked model

# with open(CONFIG_DIR, "r") as f:
#     cfg = yaml.safe_load(f)



def get_token_prob_for_layer(model, batch, token_id, layer=0):
    """
    Get all neuron probabilities in a layer for a list of instances and a certain token.

    :param model HookedTransformer: The model outputing the probabilities
    :param batch torch.tensor: The tokens
    :param token_id int: The index of the token for which the probability is calculated
    :param layer int: Layer index
    :returns token_probs torch.tensor: A tensor of all probabilities of the token within the layer
    """
    with torch.inference_mode():
        # print('CREATING CACHE')
        _, cache = model.run_with_cache(batch)
    
        # print('DECOMPOSE')
        decom = cache.get_neuron_results(layer=layer, neuron_slice=Slice(), pos_slice=Slice())

        # print('PROJECTION')
        # Project each layer and each position onto vocab space
        vocab_proj = einsum(
            "batch pos neuron d_model, d_model d_vocab --> neuron batch pos d_vocab",
            decom,
            model.W_U,
        )
    
    # print('PROBS')
    token_probs = vocab_proj.softmax(dim=-1)[:, :, -1, token_id]

    return token_probs

if __name__ == "__main__":
    model = load_hooked("gpt2-medium", os.path.join(MODEL_DIR, "dpo.pt"))
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    prompts = list(
        np.load(os.path.join(ROOT_DIR, "toxicity/figures/shit_prompts.npy"))
    )
    tokens = model.to_tokens(prompts, prepend_bos=True)
    batchsize = 4
    all_dpo_prob = None
    for idx in tqdm(range(0, tokens.shape[0], batchsize)):
        batch = tokens[idx : idx + batchsize].cuda()

        # TODO: not just one layer
        shit_probs = get_token_prob_for_layer(model, batch, 7510, layer=0)
        if all_dpo_prob is None:
            all_dpo_prob = shit_probs
        else:
            all_dpo_prob = torch.concat([all_dpo_prob, shit_probs], dim=1)
    print(f"all_dpo_prob: {all_dpo_prob.shape}")