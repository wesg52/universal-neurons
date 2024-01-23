import os
import time
import tqdm
import torch
import einops
import datasets
import argparse
import numpy as np
import pandas as pd
from functools import partial
from utils import get_model_family
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import torch.nn.functional as F
from transformer_lens.utils import lm_cross_entropy_loss
from activations import get_correct_token_rank


def quantize_neurons(activation_tensor, output_precision=8):
    activation_tensor = activation_tensor.to(torch.float32)
    min_vals = activation_tensor.min(dim=0)[0]
    max_vals = activation_tensor.max(dim=0)[0]
    num_quant_levels = 2**output_precision
    scale = (max_vals - min_vals) / (num_quant_levels - 1)
    zero_point = torch.round(-min_vals / scale)
    return torch.quantize_per_channel(
        activation_tensor, scale, zero_point, 1, torch.quint8)


def zero_ablation_hook(activations, hook, neuron):
    activations[:, :, neuron] = 0
    return activations


def threshold_ablation_hook(activations, hook, neuron, threshold=0):
    activations[:, :, neuron] = torch.min(
        activations[:, :, neuron],
        threshold * torch.ones_like(activations[:, :, neuron])
    )
    return activations


def relu_ablation_hook(activations, hook, neuron):
    activations[:, :, neuron] = torch.relu(activations[:, :, neuron])
    return activations


def fixed_activation_hook(activations, hook, neuron, fixed_act=0):
    activations[:, :, neuron] = fixed_act
    return activations


def make_hooks(args, layer, neuron):
    if args.intervention_type == 'zero_ablation':
        hook_fn = partial(zero_ablation_hook, neuron=neuron)
    elif args.intervention_type == 'threshold_ablation':
        hook_fn = partial(
            threshold_ablation_hook,
            neuron=neuron,
            threshold=args.intervention_param)
    elif args.intervention_type == 'fixed_activation':
        hook_fn = partial(
            fixed_activation_hook,
            neuron=neuron,
            fixed_act=args.intervention_param)
    elif args.intervention_type == 'relu_ablation':
        hook_fn = partial(relu_ablation_hook, neuron=neuron)
    else:
        raise ValueError(
            f'Unknown intervention type: {args.intervention_type}')

    hook_loc = f'blocks.{layer}.{args.activation_location}'

    return [(hook_loc, hook_fn)]


def run_intervention_experiment(args, model, dataset, device):

    n, d = dataset['tokens'].shape

    layer, neuron = args.neuron.split('.')
    layer, neuron = int(layer), int(neuron)

    hooks = make_hooks(args, layer, neuron)

    loss_tensor = torch.zeros(n, d-1, dtype=torch.float16)
    entropy_tensor = torch.zeros(n, d, dtype=torch.float16)
    rank_tensor = torch.zeros(n, d-1, dtype=torch.int32)

    dataloader = DataLoader(
        dataset['tokens'], batch_size=args.batch_size, shuffle=False)

    offset = 0
    for step, batch in enumerate(tqdm.tqdm(dataloader)):
        batch = batch.to(device)
        logits = model.run_with_hooks(
            batch,
            fwd_hooks=hooks
        )
        bs = batch.shape[0]
        token_loss = lm_cross_entropy_loss(logits, batch, per_token=True).cpu()
        probs = F.softmax(logits, dim=-1)
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).cpu()
        token_ranks = get_correct_token_rank(logits, batch).cpu()

        loss_tensor[offset:offset+bs] = token_loss
        entropy_tensor[offset:offset+bs] = entropies
        rank_tensor[offset:offset+bs] = token_ranks

        offset += batch.shape[0]

        model.reset_hooks()

    save_path = os.path.join(
        args.output_dir,
        args.model,
        args.token_dataset,
        args.intervention_type+'_'+str(args.intervention_param),
        args.neuron,
    )
    os.makedirs(save_path, exist_ok=True)

    torch.save(loss_tensor, os.path.join(save_path, 'loss.pt'))
    torch.save(entropy_tensor, os.path.join(save_path, 'entropy.pt'))
    torch.save(rank_tensor, os.path.join(save_path, 'rank.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general arguments
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--token_dataset',
        help='Name of cached feature dataset')
    parser.add_argument(
        '--activation_location', default='mlp.hook_pre',
        help='Model component to save')

    # activation processing/subsetting arguments
    parser.add_argument(
        '--batch_size', default=32, type=int)

    parser.add_argument(
        '--neuron', type=str, default=None,
        help='Path to file containing neuron subset')
    parser.add_argument(
        '--intervention_type', type=str, default='zero_ablation',
        help='Type of intervention to perform')
    parser.add_argument(
        '--intervention_param', type=float, default=0,
        help='Parameter for intervention type (eg, threshold or fixed activation)')

    # saving arguments
    parser.add_argument(
        '--save_precision', default=16, type=int)
    parser.add_argument(
        '--output_dir', default='intervention_results')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'))

    model = HookedTransformer.from_pretrained(args.model, device='cpu')
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    model_family = get_model_family(args.model)

    tokenized_dataset = datasets.load_from_disk(
        os.path.join(
            os.getenv('DATASET_DIR', 'token_datasets'),
            model_family,
            args.token_dataset
        )
    )

    run_intervention_experiment(
        args, model, tokenized_dataset, device)
