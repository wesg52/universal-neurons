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
from intervention import (
    zero_ablation_hook,
    threshold_ablation_hook,
    relu_ablation_hook,
    fixed_activation_hook,
    quantize_neurons
)


def multiply_activation_hook(activations, hook, neuron, multiplier=1):
    activations[:, :, neuron] = activations[:, :, neuron] * multiplier
    return activations

def save_layer_norm_scale_hook(activations, hook):
    hook.ctx['activation'] = activations.detach().cpu()


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

    elif args.intervention_type == 'multiply_activation':
        hook_fn = partial(
            multiply_activation_hook,
            neuron=neuron,
            multiplier=args.intervention_param)
    else:
        raise ValueError(
            f'Unknown intervention type: {args.intervention_type}')

    hook_loc = f'blocks.{layer}.{args.activation_location}'

    return [(hook_loc, hook_fn)]


def run_intervention_experiment(args, model, dataset, device):

    neuron_subset = args.neuron_subset

    hooks = []
    for lix, nix in neuron_subset:
        hooks += make_hooks(args, lix, nix)

    hooks.append(('ln_final.hook_scale', save_layer_norm_scale_hook))

    n, d = dataset['tokens'].shape
    loss_tensor = torch.zeros(n, d, dtype=torch.float16)
    entropy_tensor = torch.zeros(n, d, dtype=torch.float16)
    rank_tensor = torch.zeros(n, d, dtype=torch.int32)
    scale_tensor = torch.zeros(n, d, dtype=torch.float16)

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

        loss_tensor[offset:offset+bs, :-1] = token_loss
        entropy_tensor[offset:offset+bs] = entropies
        rank_tensor[offset:offset+bs, :-1] = token_ranks

        scale = model.hook_dict['ln_final.hook_scale'].ctx['activation'].squeeze()
        scale_tensor[offset:offset+bs] = scale

        offset += batch.shape[0]

        model.reset_hooks()

    save_path = os.path.join(
        args.output_dir,
        args.model,
        args.token_dataset,
        '_'.join([f'{l}.{n}' for l, n in neuron_subset]),
        args.intervention_type+'_'+str(args.intervention_param),
    )
    os.makedirs(save_path, exist_ok=True)
    torch.save(loss_tensor, os.path.join(save_path, f'loss.pt'))
    torch.save(entropy_tensor, os.path.join(
        save_path, f'entropy.pt'))
    torch.save(rank_tensor, os.path.join(save_path, f'rank.pt'))
    torch.save(scale_tensor, os.path.join(save_path, f'scale.pt'))


def parse_neuron_str(neuron_str: str):
    neurons = []
    for group in neuron_str.split(','):
        lix, nix = group.split('.')
        neurons.append((int(lix), int(nix)))
    return neurons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general arguments
    parser.add_argument(
        '--model', default='stanford-gpt2-small-a',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--token_dataset',
        help='Name of cached feature dataset')
    parser.add_argument(
        '--activation_location', default='mlp.hook_post',
        help='Model component to save')

    # activation processing/subsetting arguments
    parser.add_argument(
        '--batch_size', default=32, type=int)
    parser.add_argument(
        '--device', default=torch.device('cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')), type=str,
    )

    parser.add_argument(
        '--neuron_subset', type=parse_neuron_str, default=None,
        help='list of neurons')

    parser.add_argument(
        '--intervention_type', type=str, default='fixed_activation',
        help='Type of intervention to perform')
    parser.add_argument(
        '--intervention_param', type=float, default=None,
        help='Parameter for intervention type (eg, threshold or fixed activation)')

    # saving arguments
    parser.add_argument(
        '--save_precision', default=16, type=int)
    parser.add_argument(
        '--output_dir', default='intervention_results')

    args = parser.parse_args()

    device = args.device

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
