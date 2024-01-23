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
from transformer_lens.utils import lm_cross_entropy_loss


def quantize_neurons(activation_tensor, output_precision=8):
    activation_tensor = activation_tensor.to(torch.float32)
    min_vals = activation_tensor.min(dim=0)[0]
    max_vals = activation_tensor.max(dim=0)[0]
    num_quant_levels = 2**output_precision
    scale = (max_vals - min_vals) / (num_quant_levels - 1)
    zero_point = torch.round(-min_vals / scale)
    return torch.quantize_per_channel(
        activation_tensor, scale, zero_point, 1, torch.quint8)


def process_layer_activation_batch(batch_activations, activation_aggregation):
    if activation_aggregation is None:
        batch_activations = einops.rearrange(
            batch_activations, 'b c d -> (b c) d')
    elif activation_aggregation == 'mean':
        batch_activations = batch_activations.mean(dim=1)
    elif activation_aggregation == 'max':
        batch_activations = batch_activations.max(dim=1).values
    elif batch_activations == 'last':
        batch_activations = batch_activations[:, -1, :]
    else:
        raise ValueError(
            f'Invalid activation aggregation: {activation_aggregation}')
    return batch_activations


def process_masked_layer_activation_batch(batch_activations, batch_mask, activation_aggregation):
    if activation_aggregation is None:
        # only save the activations for the required indices
        batch_activations = einops.rearrange(
            batch_activations, 'b c d -> (b c) d')  # batch, context, dim
        processed_activations = batch_activations[batch_mask.flatten()]

    elif activation_aggregation == 'mean':
        # average over the context dimension for non-masked tokens only
        masked_activations = batch_activations * batch_mask[:, :, None]
        seq_indices = batch_mask.sum(dim=1)
        processed_activations = masked_activations.sum(
            dim=1) / seq_indices[:, None]

    elif activation_aggregation == 'max':
        # max over the context dimension for non-masked only (set masked tokens to -1)
        batch_mask = batch_mask[:, :, None].to(int)
        # set masked tokens to -1
        masked_activations = batch_activations * batch_mask + (batch_mask - 1)
        processed_activations = masked_activations.max(dim=1)[0]

    elif activation_aggregation == 'last':
        # TODO: save the last non-masked token in the mask
        raise NotImplementedError

    else:
        raise ValueError(
            f'Invalid activation aggregation: {activation_aggregation}')

    return processed_activations


def get_layer_activations(args, model, dataset, device):

    index_mask = None if 'index_mask' not in dataset.column_names \
        else dataset['index_mask']

    # preallocate memory for activations
    n, d = dataset['tokens'].shape
    n_layers = model.cfg.n_layers
    activation_rows = n \
        if args.activation_aggregation is not None \
        else n * d
    layer_activations = {
        l: torch.zeros(activation_rows, model.cfg.d_mlp, dtype=torch.float16)
        for l in range(n_layers)
    }

    def save_layer_activation_hook(tensor, hook):
        hook.ctx['activation'] = tensor.detach().cpu().to(torch.float16)

    # define hooks to save activations from each layer
    hooks = [
        (f'blocks.{layer_ix}.{args.activation_location}',
         save_layer_activation_hook)
        for layer_ix in range(n_layers)
    ]

    dataloader = DataLoader(
        dataset['tokens'], batch_size=args.batch_size, shuffle=False)

    mask_offset = 0
    layer_offset = 0
    for step, batch in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            model.run_with_hooks(
                batch.to(device),
                fwd_hooks=hooks,
            )

        for lix, (hook_pt, _) in enumerate(hooks):
            batch_activations = model.hook_dict[hook_pt].ctx['activation']
            bs = batch_activations.shape[0]
            if index_mask is not None:
                batch_mask = index_mask[mask_offset:mask_offset + bs]
                batch_activations = process_masked_layer_activation_batch(
                    batch_activations, batch_mask, args.activation_aggregation)
            else:
                batch_activations = process_layer_activation_batch(
                    batch_activations, args.activation_aggregation)

            pbs = batch_activations.shape[0]  # after processing
            layer_activations[lix][layer_offset:layer_offset +
                                   pbs] = batch_activations

        # assumed to be same for all layers
        mask_offset += bs
        layer_offset += pbs

        model.reset_hooks()

    # save activations
    save_path = os.path.join(
        args.output_dir,
        args.model,
        args.token_dataset
    )
    os.makedirs(save_path, exist_ok=True)
    agg = 'none' if args.activation_aggregation is None else args.activation_aggregation
    for layer_ix, activations in layer_activations.items():
        torch.save(
            quantize_neurons(activations, args.save_precision),
            os.path.join(save_path, f'{layer_ix}.all.{agg}.all.pt')
        )


def get_correct_token_rank(logits, indices):
    """
    :param logits: Tensor of shape [b, pos, token] with token logits
    :param indices: Tensor of shape [b, pos] with token indices
    :return: Tensor of shape [b, pos] with ranks of the correct next token
    """
    # Offset to account for next token prediction
    indices = indices[:, 1:].to(torch.int32)
    logits = logits[:, :-1, :]
    # Sort logits and get the sorted indices
    _, sorted_indices = logits.sort(descending=True, dim=-1)

    sorted_indices = sorted_indices.to(torch.int32)

    # Expand dims for indices to match the shape of sorted_indices
    expanded_indices = indices.unsqueeze(-1).expand_as(sorted_indices)
    # Find the rank of the true token
    ranks = (sorted_indices == expanded_indices).nonzero(as_tuple=True)[-1]
    # Reshape ranks to [b, pos]
    ranks = ranks.reshape(logits.size(0), logits.size(1))

    return ranks


def save_neurons_in_layer_hook(tensor, hook, neurons):
    n_acts = tensor[:, :, neurons].detach().cpu().to(torch.float16)
    hook.ctx['activation'] = n_acts


def get_neuron_activations(args, model, dataset, device, neuron_subset):
    unique_layers = list(sorted(set([l for l, _ in neuron_subset])))
    layer_neurons = {
        l: sorted([n for l_, n in neuron_subset if l_ == l])
        for l in unique_layers
    }
    # preallocate memory for activations
    n, d = dataset['tokens'].shape
    activation_rows = n * d
    layer_activations = {
        l: torch.zeros(activation_rows, len(neurons), dtype=torch.float16)
        for l, neurons in layer_neurons.items()
    }
    hooks = [
        (f'blocks.{layer_ix}.{args.activation_location}',
            partial(save_neurons_in_layer_hook, neurons=torch.tensor(neurons)))
        for layer_ix, neurons in layer_neurons.items()
    ]

    loss_tensor = torch.zeros(n, d, dtype=torch.float16)
    entropy_tensor = torch.zeros(n, d, dtype=torch.float16)
    rank_tensor = torch.zeros(n, d, dtype=torch.int32)

    dataloader = DataLoader(
        dataset['tokens'], batch_size=args.batch_size, shuffle=False)

    act_offset = 0
    batch_offset = 0
    for step, batch in enumerate(tqdm.tqdm(dataloader)):
        batch = batch.to(device)
        logits = model.run_with_hooks(
            batch,
            fwd_hooks=hooks
        )

        bs = batch.shape[0]
        token_loss = lm_cross_entropy_loss(logits, batch, per_token=True).cpu()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).cpu()
        token_ranks = get_correct_token_rank(logits, batch).cpu()

        loss_tensor[batch_offset:batch_offset+bs, :-1] = token_loss
        entropy_tensor[batch_offset:batch_offset+bs] = entropies
        rank_tensor[batch_offset:batch_offset+bs, :-1] = token_ranks

        for hook_ix, (hook_pt, _) in enumerate(hooks):
            layer_ix = unique_layers[hook_ix]
            batch_activations = model.hook_dict[hook_pt].ctx['activation']
            batch_activations = einops.rearrange(
                batch_activations, 'b d n -> (b d) n')
            n_acts = batch_activations.shape[0]
            layer_activations[layer_ix][act_offset:act_offset +
                                        n_acts] = batch_activations

        act_offset += n_acts
        batch_offset += batch.shape[0]

        model.reset_hooks()

    # save activations
    save_path = os.path.join(
        args.output_dir,
        args.model,
        args.token_dataset
    )
    os.makedirs(save_path, exist_ok=True)

    torch.save(loss_tensor, os.path.join(save_path, 'loss.pt'))
    torch.save(entropy_tensor, os.path.join(save_path, 'entropy.pt'))
    torch.save(rank_tensor, os.path.join(save_path, 'rank.pt'))

    for layer_ix, activations in layer_activations.items():
        for neuron_ix, neuron in enumerate(layer_neurons[layer_ix]):
            torch.save(
                activations[:, neuron_ix].clone(),
                os.path.join(save_path, f'{layer_ix}.{neuron}.pt')
            )


def parse_neuron_str(neuron_str):
    lix, nix = neuron_str.split('.')
    return (int(lix), int(nix))


def load_neuron_subset_csv(model, neuron_subset_file, return_df=False):
    '''
    Load a neuron subset of the form [(lix, nix), ...] from csv file
    '''
    # path = os.path.join(
    #     os.environ.get('INTERPRETABLE_NEURONS_DIR', 'interpretable_neurons'),
    #     model,
    #     neuron_subset_file
    # )
    ndf = pd.read_csv(neuron_subset_file)

    neuron_subset = [
        (int(lix), int(nix)) for lix, nix
        in zip(ndf['layer'].values, ndf['neuron'].values)
    ]
    if return_df:
        return neuron_subset, ndf
    else:
        return neuron_subset


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
        '--activation_aggregation', default=None,
        help='Average activations across all tokens in a sequence')
    # parser.add_argument(
    #     '--layer_subset', default='all', type=str,
    #     help='Comma separated list of layer indices to save')
    parser.add_argument(
        '--neuron_subset', nargs='+', type=parse_neuron_str, default=None,
        help='Neurons to save')
    parser.add_argument(
        '--neuron_subset_file', default=None,
        help='name of csv file containing a layer,neuron pairs with additional metadata)')

    # saving arguments
    parser.add_argument(
        '--save_precision', default=8, type=int)
    parser.add_argument(
        '--output_dir', default='cached_activations')

    # TODO:
    # - add parameter for index mask
    # - add parameter for save by batch
    # - save by layer

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

    if args.neuron_subset is not None or args.neuron_subset_file is not None:
        if args.neuron_subset is not None:
            neuron_subset = args.neuron_subset
        else:
            neuron_subset = load_neuron_subset_csv(
                args.model, args.neuron_subset_file)
        print(neuron_subset)
        get_neuron_activations(
            args, model, tokenized_dataset, device, neuron_subset)

    else:
        get_layer_activations(args, model, tokenized_dataset, device)
