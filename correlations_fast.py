import os
import time
import tqdm
import torch as t
import einops
import datasets
import argparse
from utils import *
from functools import partial
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from analysis.correlations import summarize_correlation_matrix, flatten_layers


class StreamingPearsonComputer:
    def __init__(self, model_1, model_2, device='cpu'):
        m1_layers = model_1.cfg.n_layers
        m2_layers = model_2.cfg.n_layers
        m1_dmlp = model_1.cfg.d_mlp
        m2_dmlp = model_2.cfg.d_mlp
        self.device = device

        self.m1_sum = t.zeros(
            (m1_layers, m1_dmlp), dtype=t.float64, device=device)
        self.m1_sum_sq = t.zeros(
            (m1_layers, m1_dmlp), dtype=t.float64, device=device)

        self.m2_sum = t.zeros(
            (m2_layers, m2_dmlp), dtype=t.float64, device=device)
        self.m2_sum_sq = t.zeros(
            (m2_layers, m2_dmlp), dtype=t.float64, device=device)

        self.m1_m2_sum = t.zeros(
            (m1_layers, m1_dmlp, m2_layers, m2_dmlp),
            dtype=t.float64, device=device
        )
        self.n = 0

    def update_correlation_data(self, batch_1_acts, batch_2_acts):

        for l1 in range(batch_1_acts.shape[0]):
            # iterating over layers in batch_2_acts
            batch_1_acts_l1 = batch_1_acts[l1].to(torch.float32)

            for l2 in range(batch_2_acts.shape[0]):
                layerwise_result = einops.einsum(
                    batch_1_acts_l1, batch_2_acts[l2].to(
                        torch.float32), 'l1 t, l2 t -> l1 l2'
                )
                self.m1_m2_sum[l1, :, l2, :] += layerwise_result.cpu()

        # batch_1_acts = batch_1_acts.to(self.device)
        # batch_2_acts = batch_2_acts.to(self.device)

        self.m1_sum += batch_1_acts.sum(dim=-1).cpu()
        self.m1_sum_sq += (batch_1_acts**2).sum(dim=-1).cpu()
        self.m2_sum += batch_2_acts.sum(dim=-1).cpu()
        self.m2_sum_sq += (batch_2_acts**2).sum(dim=-1).cpu()

        # TODO: reduce memory consumption (consider doing layerwise)
        # for large models may need to do disk caching

        self.n += batch_1_acts.shape[-1]

    def compute_correlation(self):
        layer_correlations = []
        # compute layerwise for memory efficiency
        for l1 in range(self.m1_sum.shape[0]):
            numerator = self.m1_m2_sum[l1, :, :, :] - (1 / self.n) * einops.einsum(
                self.m1_sum[l1, :], self.m2_sum, 'n1, l2 n2 -> n1 l2 n2')

            m1_norm = (self.m1_sum_sq[l1, :] -
                       (1 / self.n) * self.m1_sum[l1, :]**2)**0.5
            m2_norm = (self.m2_sum_sq - (1 / self.n) * self.m2_sum**2)**0.5

            l_correlation = numerator / einops.einsum(
                m1_norm, m2_norm, 'n1, l2 n2 -> n1 l2 n2'
            )
            layer_correlations.append(l_correlation.to(t.float16))

        correlation = t.stack(layer_correlations, dim=0)
        return correlation


def save_activation_hook(tensor, hook, device='cpu'):
    hook.ctx['activation'] = tensor.detach().to(torch.float16).to(device)


def get_activations(model, inputs, filter_padding=True):
    """Get the activations for a given model and dataset. 
    Inputs should already be appropriately batched
    inputs: (n_tokens, n_sequences) 512 x 32 by default
    out: (n_tokens, n_sequences, (n_layers * d_mlp))
    """
    hooks = [
        (f'blocks.{layer_ix}.mlp.hook_post',
         partial(save_activation_hook, device=args.model_1_device))
        for layer_ix in range(model.cfg.n_layers)
    ]

    with t.no_grad():
        model.run_with_hooks(
            inputs,
            fwd_hooks=hooks,
            stop_at_layer=model.cfg.n_layers+1  # don't compute logits to save memory
        )
    activations = torch.stack(
        [model.hook_dict[hook_pt[0]].ctx['activation'] for hook_pt in hooks], dim=2)
    model.reset_hooks()

    activations = einops.rearrange(
        activations, 'batch context l n -> l n (batch context)')

    if filter_padding:
        # In Pythia and GPT2, pad and bos tokens are the same id
        forbidden_tokens = torch.tensor([
            model.tokenizer.pad_token_id,
            model.to_single_token('\n')
        ]).to(inputs.device)
        valid_tokens = torch.isin(inputs, forbidden_tokens, invert=True)
        activations = activations[:, :, valid_tokens.flatten()]

    return activations


def run_correlation_experiment(args, model_1, model_2, token_dataset):

    # set up the streaming correlation data structures
    if args.similarity_type == 'pearson':
        corr_computer = StreamingPearsonComputer(
            model_1, model_2, device=args.correlation_device)
    # elif args.similarity_type == 'jaccard':
    #     corr_computer = StreamingJaccardComputer(
    #         model_1, model_2, device=args.correlation_device)
    # elif args.similarity_type == 'cosine':
    #     corr_computer = StreamingCosineSimComputer(
    #         model_1, model_2, device=args.correlation_device)
    else:
        raise ValueError(f'Invalid similarity type: {args.similarity_type}')

    dataloader = DataLoader(
        token_dataset['tokens'], batch_size=args.batch_size, shuffle=False)
    start_time = time.time()
    # run models

    if args.baseline == 'rotation':
        # TODO: consider making this actually orthogonal
        # eg, scipy.stats.special_ortho_group
        # see https://math.stackexchange.com/questions/3839152/sample-a-random-rotation-in-n-dimensions
        rotation_matrix = t.randn(
            (model_2.cfg.n_layers, model_2.cfg.d_mlp, model_2.cfg.d_mlp))
        rotation_matrix /= rotation_matrix.norm(dim=-1, keepdim=True)
        rotation_matrix = rotation_matrix.to(
            torch.float16).to(args.model_1_device)

    for step, batch in enumerate(tqdm.tqdm(dataloader)):
        m1_activations = get_activations(
            model_1, batch.to(args.model_1_device))

        # do special processing for the baselines
        m2_activations = get_activations(
            model_2, batch.to(args.model_2_device))

        if args.baseline == 'rotation':
            # rotate the neuron basis
            rotated_acts = []
            for l in range(m2_activations.shape[0]):
                rotated_acts.append(
                    t.einsum(
                        'n t, m n -> m t',
                        m2_activations[l, :, :].to(torch.float32),
                        rotation_matrix[l, :, :].to(torch.float32)
                    )
                )
            m2_activations = t.stack(rotated_acts, dim=0)

        corr_computer.update_correlation_data(m1_activations, m2_activations)
    end_time = time.time()
    print(f"Total gpu time is {end_time - start_time}", flush=True)
    correlation = corr_computer.compute_correlation()
    end_time = time.time()
    print(f"Total time is {end_time - start_time}", flush=True)
    return correlation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_1_name', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--model_2_name', default='pythia-70m-v0')
    parser.add_argument(
        '--token_dataset', type=str)

    parser.add_argument(
        '--baseline', type=str, default='none',
        choices=['none', 'gaussian', 'permutation', 'rotation'])
    parser.add_argument(
        '--similarity_type', type=str, default='pearson',
        choices=['pearson', 'jaccard', 'cosine'])
    parser.add_argument(
        '--jaccard_threshold', type=float, default=0)

    parser.add_argument(
        '--batch_size', default=32, type=int)
    parser.add_argument(
        '--model_1_device', type=str, default='cpu')
    parser.add_argument(
        '--model_2_device', type=str, default='cpu')
    parser.add_argument(
        '--correlation_device', type=str, default='cpu')

    parser.add_argument(
        '--save_full_correlation_matrix', action='store_true',
        help='Whether to save the full correlation matrix (always save the summary)')
    parser.add_argument(
        '--save_precision', type=int, default=16, choices=[8, 16, 32],
        help='Number of bits to use for saving full correlation matrix')

    args = parser.parse_args()
    print(f"Visible CUDA devices: {t.cuda.device_count()}")
    t.autograd.set_grad_enabled(False)

    model_1 = HookedTransformer.from_pretrained(
        args.model_1_name, device='cpu')
    model_1.to(args.model_1_device)
    model_1.eval()

    model_2 = HookedTransformer.from_pretrained(
        args.model_2_name, device='cpu')
    model_2.to(args.model_2_device)
    model_2.eval()

    model_family = get_model_family(args.model_1_name)

    tokenized_dataset = datasets.load_from_disk(
        os.path.join(
            os.getenv('DATASET_DIR', 'token_datasets'),
            model_family,
            args.token_dataset
        )
    )

    correlation = run_correlation_experiment(
        args, model_1, model_2, tokenized_dataset)

    similarity_type = f'jaccard-{args.jaccard_threshold:.2f}'\
        if args.similarity_type == 'jaccard' \
        else args.similarity_type

    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'correlation_results'),
        args.model_1_name + '+' + args.model_2_name,
        args.token_dataset,
        f'{similarity_type}.{args.baseline}'
    )
    os.makedirs(save_path, exist_ok=True)

    if args.save_full_correlation_matrix:
        torch.save(
            adjust_precision(correlation.cpu(), args.save_precision),
            os.path.join(save_path, 'correlation.pt')
        )

    # save both the summary and the summary on the transpose
    correlation = flatten_layers(correlation.cpu()).to(torch.float32)
    corr_summary = summarize_correlation_matrix(correlation)
    corr_summary_T = summarize_correlation_matrix(correlation.T)

    torch.save(
        corr_summary,
        os.path.join(save_path, 'correlation_summary.pt')
    )
    torch.save(
        corr_summary_T,
        os.path.join(save_path, 'correlation_summary_T.pt')
    )
