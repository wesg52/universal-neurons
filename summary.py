import os
import time
import numpy as np
import tqdm
import torch
import einops
import datasets
import argparse
from utils import *
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer


def bin_activations(activations, neuron_bin_edges, neuron_bin_counts):
    # TODO filter out padding tokens
    bin_index = torch.searchsorted(neuron_bin_edges, activations)

    neuron_bin_counts[:] = neuron_bin_counts.scatter_add_(
        2, bin_index, torch.ones_like(bin_index, dtype=torch.int32)
    )


def update_vocabulary_statistics(
        batch, activations, neuron_vocab_max, neuron_vocab_sum, vocab_counts):
    # TODO: reduce memory needs (perhaps compute by layer)
    layers, neurons, tokens = activations.shape

    vocab_index = batch.flatten()
    extended_index = einops.repeat(  # flattened tokens per neuron
        vocab_index, 't -> l n t', l=layers, n=neurons)

    neuron_vocab_max[:] = neuron_vocab_max.scatter_reduce(
        -1, extended_index, activations, reduce='max')

    neuron_vocab_sum[:] = neuron_vocab_sum.scatter_reduce(
        -1, extended_index, activations.to(torch.float32), reduce='sum')

    token_ix, batch_count = torch.unique(vocab_index, return_counts=True)
    vocab_counts[token_ix] += batch_count


def update_top_dataset_examples(
        activations, neuron_max_activating_index, neuron_max_activating_value, index_offset):
    n_layer, n_neuron, k = neuron_max_activating_value.shape

    values = torch.cat([neuron_max_activating_value, activations], dim=2)

    batch_indices = torch.arange(activations.shape[2]) + index_offset
    extended_batch_indices = einops.repeat(
        batch_indices, 't -> l n t', l=n_layer, n=n_neuron)
    indices = torch.cat([
        neuron_max_activating_index,
        extended_batch_indices
    ], dim=2)

    # get top k
    neuron_max_activating_value[:], top_k_indices = torch.topk(
        values, k, dim=2)
    neuron_max_activating_index[:] = torch.gather(indices, 2, top_k_indices)


def save_activation(tensor, hook):
    hook.ctx['activation'] = tensor.detach().to(torch.float16).cpu()


def summarize_activations(args, model, dataset, device):

    d_mlp = model.cfg.d_mlp
    n_layers = model.cfg.n_layers
    d_vocab = model.cfg.d_vocab

    # TODO: make bin edges adaptive
    neuron_bin_edges = torch.linspace(-10, 15, args.n_bins)
    neuron_bin_counts = torch.zeros(
        n_layers, d_mlp, args.n_bins+1, dtype=torch.int32)

    neuron_vocab_max = torch.zeros(
        n_layers, d_mlp, d_vocab, dtype=torch.float16)
    neuron_vocab_sum = torch.zeros(  # for computing average
        n_layers, d_mlp, d_vocab, dtype=torch.float32)
    vocab_counts = torch.zeros(d_vocab)

    neuron_max_activating_index = torch.zeros(
        n_layers, d_mlp, args.top_k_dataset_examples, dtype=torch.int64)
    neuron_max_activating_value = torch.zeros(
        n_layers, d_mlp, args.top_k_dataset_examples, dtype=torch.float32)

    # define hooks to save activations from each layer
    pre_hooks = [f'blocks.{layer}.mlp.hook_pre' for layer in range(n_layers)]
    post_hooks = [f'blocks.{layer}.mlp.hook_post' for layer in range(n_layers)]
    all_hook_pts = pre_hooks + post_hooks
    all_hooks = [(hook_pt, save_activation) for hook_pt in all_hook_pts]

    dataloader = DataLoader(
        dataset['tokens'], batch_size=args.batch_size, shuffle=False)

    index_offset = 0
    for step, batch in enumerate(tqdm.tqdm(dataloader)):
        model.run_with_hooks(
            batch.to(device),
            fwd_hooks=all_hooks,
        )
        # stack and reformat activations
        pre_acts = torch.stack([
            model.hook_dict[hook_pt].ctx['activation'] for hook_pt in pre_hooks
        ], dim=2)
        post_acts = torch.stack([
            model.hook_dict[hook_pt].ctx['activation'] for hook_pt in post_hooks
        ], dim=2)
        model.reset_hooks()

        pre_acts = einops.rearrange(
            pre_acts, 'batch context l n -> l n (batch context)')
        post_acts = einops.rearrange(
            post_acts, 'batch context l n -> l n (batch context)')

        # update neuron statistics (all performed in place)
        bin_activations(pre_acts, neuron_bin_edges, neuron_bin_counts)

        update_vocabulary_statistics(
            batch, post_acts, neuron_vocab_max, neuron_vocab_sum, vocab_counts)

        update_top_dataset_examples(
            post_acts, neuron_max_activating_index, neuron_max_activating_value, index_offset)

        batch_size, ctx_len = batch.shape
        index_offset += batch_size * ctx_len

    # save statistics (TODO: consider saving by layer; 8bit quantization)
    save_path = os.path.join(
        args.output_dir,
        args.model,
        'activations',
        args.token_dataset,
    )
    os.makedirs(save_path, exist_ok=True)

    torch.save(neuron_bin_counts, os.path.join(
        save_path, 'neuron_bin_counts.pt'))
    torch.save(neuron_bin_edges, os.path.join(
        save_path, 'neuron_bin_edges.pt'))
    torch.save(neuron_max_activating_index, os.path.join(
        save_path, 'neuron_max_activating_index.pt'))
    torch.save(neuron_max_activating_value.to(torch.float16), os.path.join(
        save_path, 'neuron_max_activating_value.pt'))

    if args.top_vocab_k_truncate > 0:
        # TODO: filter out averages with low counts
        k = args.top_vocab_k_truncate
        top_vocab_avg, top_vocab_avg_ixs = torch.topk(
            neuron_vocab_sum / (vocab_counts + 1e-3), k, dim=-1)

        top_vocab_max, top_vocab_max_ixs = torch.topk(
            neuron_vocab_max.to(torch.float32), k, dim=-1)

        # assumes <65536 vocab words (uint16 not implemented in pytorch)
        top_vocab_avg_ixs = top_vocab_avg_ixs.numpy().astype(np.uint16)
        top_vocab_max_ixs = top_vocab_max_ixs.numpy().astype(np.uint16)

        torch.save(top_vocab_avg.to(torch.float16), os.path.join(
            save_path, 'neuron_vocab_mean.pt'))
        np.save(os.path.join(save_path, 'neuron_vocab_mean_ixs.npz'),
                top_vocab_avg_ixs)
        torch.save(top_vocab_max.to(torch.float16), os.path.join(
            save_path, 'neuron_vocab_max.pt'))
        np.save(os.path.join(save_path, 'neuron_vocab_max_ixs.npz'),
                top_vocab_max_ixs)
        torch.save(vocab_counts, os.path.join(
            save_path, 'vocab_counts.pt'))
    elif args.top_vocab_k_truncate == 0:  # don't save
        pass
    else:  # save all (-1)
        torch.save(neuron_vocab_max, os.path.join(
            save_path, 'neuron_vocab_max.pt'))
        torch.save((neuron_vocab_sum / vocab_counts).to(torch.float16), os.path.join(
            save_path, 'neuron_vocab_mean.pt'))
        torch.save(vocab_counts, os.path.join(
            save_path, 'vocab_counts.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--token_dataset',
        help='Name of cached feature dataset')
    parser.add_argument(
        '--output_dir', default='summary_data')
    parser.add_argument(
        '--batch_size', default=32, type=int)
    parser.add_argument(
        '--n_bins', default=256, type=int)
    parser.add_argument(
        '--top_k_dataset_examples', default=50, type=int,
        help='Number of top dataset examples to save')
    parser.add_argument(
        '--top_vocab_k_truncate', default=100, type=int,
        help='Number of top vocab words (by avg and max) to save (-1 for all)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    summarize_activations(args, model, tokenized_dataset, device)
