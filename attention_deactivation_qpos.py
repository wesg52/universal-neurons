
import torch as t
from torch import Tensor
from tqdm import tqdm
from jaxtyping import Float, Int, Bool
from itertools import combinations
import einops
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from transformer_lens import utils, HookedTransformer, ActivationCache
from datasets import load_dataset
import plotly.express as px
import pandas as pd
from fancy_einsum import einsum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial
import re
import scipy
import numpy as np
from utils import * 
import argparse
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)
import datasets
import os
from torchmetrics.regression import SpearmanCorrCoef
spearman = SpearmanCorrCoef()
from torch.utils.data import DataLoader
t.set_grad_enabled(False)

pair = [8,3,7] # The head and MLP layer of interest [Attention Layer, Attention Head, MLP Layer]
act_name_post = utils.get_act_name("post", pair[2])
act_name_z = utils.get_act_name("z", pair[0])
act_name_pattern = utils.get_act_name("pattern", pair[0])
act_name_resid_pre = utils.get_act_name("resid_pre", pair[0])


def run_ablation(model, batched_dataset, neuron):
    
    def path_ablate_neuron_hook(
        resid_pre: Float[t.Tensor, "batch pos d_m"],
        hook: HookPoint
    ) -> Float[t.Tensor, "batch pos d_m"]:
        resid_pre[:, Qpos] -= einsum('b, d_m -> b d_m', 
                                    n_activations[:, Qpos - after_pos],
                                    model.W_out[pair[2], 
                                                sorted_BOS_n_inds[pair[0], pair[1], pair[2], -(neuron + 1)]])
        return resid_pre
    
    def correct_k_vecs(
        k: Float[t.Tensor, "batch head Kpos d_h"],
        hook: HookPoint
    ) -> Float[t.Tensor, "batch head Kpos d_h"]:

        return original_cache[utils.get_act_name("k", pair[0])].cuda()

    def correct_v_vecs(
        v: Float[t.Tensor, "batch head Kpos d_h"],
        hook: HookPoint
    ) -> Float[t.Tensor, "batch head Kpos d_h"]:

        return original_cache[utils.get_act_name("v", pair[0])].cuda()


    def get_attn_score_hook(
        pattern: Float[t.Tensor, "batch head Qpos Kpos"],
        hook: HookPoint
    ) -> Float[t.Tensor, "batch head Qpos Kpos"]:
        single_n_cache_score[:, Qpos - after_pos] = pattern[:, pair[1], Qpos, 0]
        return pattern

    def get_attn_norm(
        z: Float[t.Tensor, "batch head Qpos Kpos"],
        hook: HookPoint
    ) -> Float[t.Tensor, "batch head Qpos Kpos"]:
        n_att_norm[:, Qpos - after_pos] = z[:, Qpos, pair[1]].norm(dim=-1)
        return z
    
    single_n_cache_score = t.zeros(len(batched_dataset), ctx_len - after_pos).to(device=args.device)
    n_att_norm = t.zeros(len(batched_dataset), ctx_len - after_pos).to(device=args.device)
    _, original_cache = model.run_with_cache(
            batched_dataset, 
            stop_at_layer = pair[0] + 1, 
            names_filter = [act_name_post,
                            act_name_z,
                            act_name_pattern, 
                            utils.get_act_name("k", pair[0]), 
                            utils.get_act_name("v", pair[0])])
    n_activations = original_cache[act_name_post][:, after_pos:, sorted_BOS_n_inds[pair[0], pair[1], pair[2], -(neuron + 1)]]

    for Qpos in tqdm(range(after_pos,ctx_len)):
        hook_run = model.run_with_hooks(
            batched_dataset,
            stop_at_layer = pair[0] + 1,
            fwd_hooks=[
                (act_name_resid_pre, path_ablate_neuron_hook),
                (utils.get_act_name("k", pair[0]), correct_k_vecs),
                (utils.get_act_name("v", pair[0]), correct_v_vecs),
                (act_name_pattern, get_attn_score_hook),
                (act_name_z, get_attn_norm)]
                )

    norm_diffs = n_att_norm - original_cache['z', pair[0]][:,after_pos:,pair[1]].norm(dim=-1)
    pattern_diffs = single_n_cache_score - original_cache['pattern', pair[0]][:, pair[1], after_pos:, 0]
    return n_activations, norm_diffs, pattern_diffs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_name', default='stanford-gpt2-small-a',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--token_dataset', type=str)
    parser.add_argument(
        '--context_length', type=int, default=256) 
    parser.add_argument(
        '--batch_size', default=32, type=int)
    parser.add_argument(
        '--device', type=str, default='cuda')
    parser.add_argument(
        '--after_pos', type=int, default=64)

    args = parser.parse_args()

    t.autograd.set_grad_enabled(False)
    print(f"Visible CUDA devices: {t.cuda.device_count()}")
    model = HookedTransformer.from_pretrained(
        args.model_name,
        device='cpu', 
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True
    )
    model.set_use_split_qkv_input(True)
    model.to(args.device)
    model.eval()

    model_family = get_model_family(args.model_name)

    tokenized_dataset = datasets.load_from_disk(
        os.path.join(
            os.getenv('DATASET_DIR', 'token_datasets'),
            model_family,
            args.token_dataset
        )
    )
    
    _, BOS_cache = model.run_with_cache(model.to_tokens(""), 
                                        names_filter=[utils.get_act_name('k', i) 
                                                      for i in range(model.cfg.n_layers)])
    BOS_cache.to(device='cpu')
    BOS_k_dir = t.stack([BOS_cache['k', i][0,0] for i in range(model.cfg.n_layers)])
    BOS_eff = (einsum('Al h d_m d_h, Ql n d_m, Al h d_h -> h n Al Ql',
                       model.W_Q.cpu(), model.W_out.cpu(), BOS_k_dir))/np.sqrt(model.cfg.d_head)
    causal_BOS_eff = einops.rearrange(BOS_eff.tril(diagonal=-1), 'h n Al Ql -> Al h Ql n')
    sorted_BOS_n, sorted_BOS_n_inds = causal_BOS_eff.abs().sort(dim=-1)

        
    dataloader = DataLoader(
        tokenized_dataset['tokens'], batch_size=args.batch_size, shuffle=False)

    ctx_len = args.context_length
    after_pos = args.after_pos
    corr_dict = {}
    
    neuron_idx = 2
# for neuron_idx in range(1, model.d_mlp): 
    total_n_activations, total_norm_diffs, total_pattern_diffs = None, None, None
    for step, batch in enumerate(tqdm(dataloader)):
        batch = batch[:, :ctx_len].to(device=args.device)
        n_activations, norm_diffs, pattern_diffs = run_ablation(model, batch, neuron_idx)
        n_activations = n_activations.cpu()
        norm_diffs = norm_diffs.cpu()
        pattern_diffs = pattern_diffs.cpu()
        if total_n_activations is None:
            total_n_activations = n_activations
            total_norm_diffs = norm_diffs
            total_pattern_diffs = pattern_diffs
        else: 
            total_n_activations = t.cat([total_n_activations, n_activations], dim=0)
            total_norm_diffs = t.cat([total_norm_diffs, norm_diffs], dim=0)
            total_pattern_diffs = t.cat([total_pattern_diffs, pattern_diffs], dim=0)
    pearson_corr = t.corrcoef(t.stack([t.flatten(n_activations), t.flatten(norm_diffs)]))[0,1]
    spearman_corr = spearman(t.flatten(n_activations), t.flatten(norm_diffs))
    print(f"Neuron {neuron_idx} Pearson Correlation: {pearson_corr} Spearman Correlation: {spearman_corr}")
    corr_dict[neuron_idx] = {
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }

    
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'attention_deactivation_results'),
        args.model_name,
        args.token_dataset,
    )
    os.makedirs(save_path, exist_ok=True)
    t.save(corr_dict, os.path.join(save_path, 'corr_dict.pt'))
