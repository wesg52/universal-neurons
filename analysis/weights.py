import os
import torch
import numpy as np
import pandas as pd
from utils import vector_moments


@torch.no_grad()
def neuron_vocab_cosine_moments(model):
    # runs computation on whatever device model is loaded on (recommended to use mps if available)
    n_layers, d_mlp, d_vocab = model.W_out.shape[0], model.W_U.shape[1], model.W_out.shape[1]
    means, variances, skews, kurtoses = [], [], [], []

    W_U = model.W_U / model.W_U.norm(dim=0, keepdim=True)

    for i in range(n_layers):
        w_out = model.W_out[i]

        direct = w_out @ W_U
        direct = direct / w_out.norm(dim=1)[:, None]

        mean, var, skew, kurt = vector_moments(direct)

        means.append(mean)
        variances.append(var)
        skews.append(skew)
        kurtoses.append(kurt)

    # Flatten the list of numpy arrays.
    means = torch.stack(means).flatten().cpu().numpy()
    variances = torch.stack(variances, dim=0).flatten().cpu().numpy()
    skews = torch.stack(skews, dim=0).flatten().cpu().numpy()
    kurtoses = torch.stack(kurtoses, dim=0).flatten().cpu().numpy()

    n_layers, d_mlp, _ = model.W_out.shape

    neuron_moments_df = pd.DataFrame({
        'vocab_mean': means,
        'vocab_var': variances,
        'vocab_skew': skews,
        'vocab_kurt': kurtoses
    }, index=pd.MultiIndex.from_product([range(n_layers), range(d_mlp)]))
    neuron_moments_df.index.names = ['layer', 'neuron']

    return neuron_moments_df
