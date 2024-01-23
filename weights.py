import os
import copy
import pickle
import tqdm
import argparse
import einops
import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from utils import timestamp, adjust_precision, vector_histogram, vector_moments


def load_composition_scores():
    raise NotImplementedError


def compute_neuron_composition(model, layer, zero_diag=False):
    """
    Takes in a model and a layer, and dot projection of """
    W_in = einops.rearrange(model.W_in, 'l d n -> l n d')
    W_out = model.W_out

    W_in /= torch.norm(W_in, dim=-1, keepdim=True)
    W_out /= torch.norm(W_out, dim=-1, keepdim=True)

    in_in_cos = einops.einsum(
        W_in, W_in[layer, :, :], f'l n d, m d -> m l n')
    in_out_cos = einops.einsum(
        W_out, W_in[layer, :, :], f'l n d, m d -> m l n')
    out_in_cos = einops.einsum(
        W_in, W_out[layer, :, :], f'l n d, m d -> m l n')
    out_out_cos = einops.einsum(
        W_out, W_out[layer, :, :], f'l n d, m d -> m l n')

    if zero_diag:
        diag_ix = torch.arange(in_in_cos.shape[-1])
        in_in_cos[diag_ix, layer, diag_ix] = 0
        in_out_cos[diag_ix, layer, diag_ix] = 0
        out_in_cos[diag_ix, layer, diag_ix] = 0
        out_out_cos[diag_ix, layer, diag_ix] = 0

    return in_in_cos, in_out_cos, out_in_cos, out_out_cos


def compute_attention_composition(model, layer):
    W_in = einops.rearrange(model.W_in[layer], 'd n -> n d')
    W_in /= torch.norm(W_in, dim=-1, keepdim=True)
    W_out = model.W_out[layer]
    W_out /= torch.norm(W_out, dim=-1, keepdim=True)

    k_comps, q_comps, v_comps, o_comps = [], [], [], []
    for attn_layer in range(model.cfg.n_layers):
        W_QK = model.QK[attn_layer].T.AB
        W_QK /= torch.norm(W_QK, dim=(1, 2), keepdim=True)
        k_comp = einops.einsum(W_QK, W_out, 'h q d, n d -> n h q').norm(dim=-1)
        q_comp = einops.einsum(W_QK, W_out, 'h d k, n d -> n h k').norm(dim=-1)

        W_OV = model.OV[attn_layer].T.AB
        W_OV /= torch.norm(W_OV, dim=(1, 2), keepdim=True)
        v_comp = einops.einsum(W_OV, W_out, 'h o d, n d -> n h o').norm(dim=-1)
        o_comp = einops.einsum(W_OV, W_in, 'h d v, n d -> n h v').norm(dim=-1)

        k_comps.append(k_comp)
        q_comps.append(q_comp)
        v_comps.append(v_comp)
        o_comps.append(o_comp)

    # return is d_mlp x n_layers x n_heads
    k_comps = torch.stack(k_comps, dim=1)
    q_comps = torch.stack(q_comps, dim=1)
    v_comps = torch.stack(v_comps, dim=1)
    o_comps = torch.stack(o_comps, dim=1)

    return k_comps, q_comps, v_comps, o_comps


def compute_vocab_composition(model, layer):
    W_in = einops.rearrange(model.W_in[layer, :, :], 'd n -> n d')
    W_out = model.W_out[layer, :, :]

    W_in /= torch.norm(W_in, dim=-1, keepdim=True)
    W_out /= torch.norm(W_out, dim=-1, keepdim=True)

    # W_E is (d_vocab, d_model), W_U is (d_model, d_vocab)
    W_E = model.W_E / torch.norm(model.W_E, dim=-1, keepdim=True)
    W_U = model.W_U / torch.norm(model.W_U, dim=0, keepdim=True)

    in_E_cos = einops.einsum(W_E, W_in, 'v d, n d -> n v')
    in_U_cos = einops.einsum(W_U, W_in, 'd v, n d -> n v')
    out_E_cos = einops.einsum(W_E, W_out, 'v d, n d -> n v')
    out_U_cos = einops.einsum(W_U, W_out, 'd v, n d -> n v')

    return in_E_cos, in_U_cos, out_E_cos, out_U_cos


def compute_neuron_statistics(model):

    W_in = einops.rearrange(model.W_in, 'l d n -> l n d')
    W_out = model.W_out

    layers, d_mlp, d_model = W_in.shape

    W_in_norms = torch.norm(W_in, dim=-1)
    W_out_norms = torch.norm(W_out, dim=-1)

    # Calculate cosine similarity: dot(A, B) / (||A|| * ||B||)
    dot_product = (W_in * W_out).sum(dim=-1)
    cos_sim = dot_product / (W_in_norms * W_out_norms)

    index = pd.MultiIndex.from_product(
        [range(layers), range(4*d_model)],
        names=["layer", "neuron_ix"]
    )
    stat_df = pd.DataFrame({
        "input_weight_norm": W_in_norms.detach().numpy().flatten(),
        "input_bias": model.b_in.detach().numpy().flatten(),
        "output_weight_norm": W_out_norms.detach().numpy().flatten(),
        # note output bias is not d_mlp, but d_model
        "in_out_sim": cos_sim.detach().numpy().flatten()
    }, index=index)

    return stat_df


def run_weight_summary(args, model):
    save_path = os.path.join(
        args.save_path,
        args.model,
        'weights'
    )
    os.makedirs(save_path, exist_ok=True)

    stat_df = compute_neuron_statistics(model)
    stat_df.to_csv(os.path.join(save_path, 'neuron_stats.csv'))

    # attention composition summary
    k_comps, q_comps, v_comps, o_comps = [], [], [], []
    for layer in tqdm.tqdm(range(model.cfg.n_layers)):
        k_comp, q_comp, v_comp, o_comp = compute_attention_composition(
            model, layer)
        k_comps.append(k_comp)
        q_comps.append(q_comp)
        v_comps.append(v_comp)
        o_comps.append(o_comp)
    k_comps = torch.stack(k_comps, dim=0)
    q_comps = torch.stack(q_comps, dim=0)
    v_comps = torch.stack(v_comps, dim=0)
    o_comps = torch.stack(o_comps, dim=0)
    torch.save(k_comps.to(torch.float16),
               os.path.join(save_path, 'k_comps.pt'))
    torch.save(q_comps.to(torch.float16),
               os.path.join(save_path, 'q_comps.pt'))
    torch.save(v_comps.to(torch.float16),
               os.path.join(save_path, 'v_comps.pt'))
    torch.save(o_comps.to(torch.float16),
               os.path.join(save_path, 'o_comps.pt'))
    print('finished attention composition summary')

    # vocab composition summary
    bin_edges = torch.linspace(-1, 1, 100)
    vocab_comp_data = {
        'top_vocab_value': [],
        'top_vocab_ix': [],
        'bottom_vocab_value': [],
        'bottom_vocab_ix': [],
        'comp_hist': [],
        'comp_mean': [],
        'comp_var': [],
        'comp_skew': [],
        'comp_kurt': []
    }
    vocab_composition_types = ['E_in', 'U_in', 'E_out', 'U_out']
    vocab_comp_dict = {
        k: copy.deepcopy(vocab_comp_data) for k in vocab_composition_types
    }
    for layer in tqdm.tqdm(range(model.cfg.n_layers)):
        layer_comps = compute_vocab_composition(model, layer)
        for comp_type, comp_score in zip(vocab_composition_types, layer_comps):
            comp_hist = vector_histogram(comp_score, bin_edges)
            vocab_comp_dict[comp_type]['comp_hist'].append(comp_hist)

            top, top_ix = torch.topk(comp_score, 100, dim=1, largest=True)
            vocab_comp_dict[comp_type]['top_vocab_value'].append(top)
            vocab_comp_dict[comp_type]['top_vocab_ix'].append(top_ix)

            bottom, bottom_ix = torch.topk(
                comp_score, 100, dim=1, largest=False)
            vocab_comp_dict[comp_type]['bottom_vocab_value'].append(bottom)
            vocab_comp_dict[comp_type]['bottom_vocab_ix'].append(bottom_ix)

            mean, var, skew, kurt = vector_moments(comp_score)
            vocab_comp_dict[comp_type]['comp_mean'].append(mean)
            vocab_comp_dict[comp_type]['comp_var'].append(var)
            vocab_comp_dict[comp_type]['comp_skew'].append(skew)
            vocab_comp_dict[comp_type]['comp_kurt'].append(kurt)

    vocab_comp_dict = {
        comp_type: {data_type: torch.stack(data_dict, dim=0)
                    for data_type, data_dict in comp_type_dict.items()}
        for comp_type, comp_type_dict in vocab_comp_dict.items()
    }
    torch.save(vocab_comp_dict, os.path.join(save_path, 'vocab_comps.pt'))
    print('finished vocab composition summary')

    # neuron composition summary
    neuron_composition_types = ['in_in', 'in_out', 'out_in', 'out_out']
    neuron_comp_data = {
        'top_neuron_value': [],
        'top_neuron_ix': [],
        'bottom_neuron_value': [],
        'bottom_neuron_ix': [],
        'comp_hist': [],
        'comp_mean': [],
        'comp_var': [],
        'comp_skew': [],
        'comp_kurt': []
    }
    neuron_comp_dict = {
        k: copy.deepcopy(neuron_comp_data) for k in neuron_composition_types
    }
    for layer in tqdm.tqdm(range(model.cfg.n_layers)):
        layer_comps = compute_neuron_composition(model, layer, zero_diag=True)
        for comp_type, comp_score in zip(neuron_composition_types, layer_comps):
            comp_score = einops.rearrange(comp_score, 'm l n -> m (l n)')
            comp_hist = vector_histogram(comp_score, bin_edges)
            neuron_comp_dict[comp_type]['comp_hist'].append(comp_hist)

            top, top_ix = torch.topk(comp_score, 20, dim=1, largest=True)
            neuron_comp_dict[comp_type]['top_neuron_value'].append(top)
            neuron_comp_dict[comp_type]['top_neuron_ix'].append(top_ix)

            bottom, bottom_ix = torch.topk(
                comp_score, 20, dim=1, largest=False)
            neuron_comp_dict[comp_type]['bottom_neuron_value'].append(bottom)
            neuron_comp_dict[comp_type]['bottom_neuron_ix'].append(bottom_ix)

            mean, var, skew, kurt = vector_moments(comp_score)
            neuron_comp_dict[comp_type]['comp_mean'].append(mean)
            neuron_comp_dict[comp_type]['comp_var'].append(var)
            neuron_comp_dict[comp_type]['comp_skew'].append(skew)
            neuron_comp_dict[comp_type]['comp_kurt'].append(kurt)

    neuron_comp_dict = {
        comp_type: {data_type: torch.stack(data_dict, dim=0)
                    for data_type, data_dict in comp_type_dict.items()}
        for comp_type, comp_type_dict in neuron_comp_dict.items()
    }
    torch.save(neuron_comp_dict, os.path.join(save_path, 'neuron_comps.pt'))
    print('finished neuron composition summary')


def run_full_weight_analysis(model, causal_only=False, save_precision=8, save_path='results/weights'):

    save_path = os.path.join(save_path, model.cfg.model_name)
    os.makedirs(save_path, exist_ok=True)

    print(f'{timestamp()} starting analysis')
    stat_df = compute_neuron_statistics(model)
    stat_df.to_csv(os.path.join(save_path, 'neuron_stats.csv'))
    print(f'{timestamp()} saved neuron df')

    neuron_cos_path = os.path.join(save_path, 'neuron_cosine')
    attn_cos_path = os.path.join(save_path, 'attn_cosine')
    vocab_cos_path = os.path.join(save_path, 'vocab_cosine')
    os.makedirs(neuron_cos_path, exist_ok=True)
    os.makedirs(attn_cos_path, exist_ok=True)
    os.makedirs(vocab_cos_path, exist_ok=True)

    for layer in range(model.cfg.n_layers):
        # neuron composition
        in_in_cos, in_out_cos, out_out_cos = compute_neuron_composition(
            model, layer)

        torch.save(
            adjust_precision(in_in_cos, save_precision,
                             per_channel=False, cos_sim=True),
            os.path.join(neuron_cos_path, f'in_in_cos_{layer}.pt')
        )
        torch.save(
            adjust_precision(in_out_cos, save_precision,
                             per_channel=False, cos_sim=True),
            os.path.join(neuron_cos_path, f'in_out_cos_{layer}.pt')
        )
        torch.save(
            adjust_precision(out_out_cos, save_precision,
                             per_channel=False, cos_sim=True),
            os.path.join(neuron_cos_path, f'out_out_cos_{layer}.pt')
        )
        del in_in_cos, in_out_cos, out_out_cos
        print(f'{timestamp()} saved neuron cosines for layer {layer}')

        # vocab composition
        in_E_cos, in_U_cos, out_E_cos, out_U_cos = compute_vocab_composition(
            model, layer)
        torch.save(
            adjust_precision(in_E_cos, save_precision,
                             per_channel=False, cos_sim=True),
            os.path.join(vocab_cos_path, f'in_E_cos_{layer}.pt')
        )
        torch.save(
            adjust_precision(in_U_cos, save_precision,
                             per_channel=False, cos_sim=True),
            os.path.join(vocab_cos_path, f'in_U_cos_{layer}.pt')
        )
        torch.save(
            adjust_precision(out_E_cos, save_precision,
                             per_channel=False, cos_sim=True),
            os.path.join(vocab_cos_path, f'out_E_cos_{layer}.pt')
        )
        torch.save(
            adjust_precision(out_U_cos, save_precision,
                             per_channel=False, cos_sim=True),
            os.path.join(vocab_cos_path, f'out_U_cos_{layer}.pt')
        )
        del in_E_cos, in_U_cos, out_E_cos, out_U_cos
        print(f'{timestamp()} saved vocab cosines for layer {layer}')

        # attention composition
        k_comps, q_comps, v_comps, o_comps = compute_attention_composition(
            model, layer)
        torch.save(
            adjust_precision(o_comps, save_precision),
            os.path.join(attn_cos_path, f'o_comp_{layer}.pt')
        )
        torch.save(
            adjust_precision(v_comps, save_precision),
            os.path.join(attn_cos_path, f'v_comp_{layer}.pt')
        )
        torch.save(
            adjust_precision(q_comps, save_precision),
            os.path.join(attn_cos_path, f'q_comp_{layer}.pt')
        )
        torch.save(
            adjust_precision(k_comps, save_precision),
            os.path.join(attn_cos_path, f'k_comp_{layer}.pt')
        )
        del o_comps, v_comps, q_comps, k_comps
        print(f'{timestamp()} saved attention cosines for layer {layer}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--causal_only', action='store_true', default=False)
    parser.add_argument(
        '--save_precision', type=int, default=8, choices=[8, 16, 32],
        help='Number of bits to use for saving cosine similarities')
    parser.add_argument(
        '--save_path', default='summary_data')
    parser.add_argument(
        '--compute_full_stats', action='store_true', default=False)

    args = parser.parse_args()

    print(f'{timestamp()} loading model')
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model, device='cpu')

    if args.compute_full_stats:
        # not recently tested
        run_full_weight_analysis(
            model,
            causal_only=args.causal_only,
            save_precision=args.save_precision,
            save_path=args.save_path
        )
    else:
        run_weight_summary(args, model)
