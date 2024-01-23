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
from transformer_lens.utils import gelu_fast
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import set_device
import time
import multiprocessing
from multiprocessing import Process, Pipe
from tqdm.contrib.concurrent import process_map


def update_correlation_data(batch_1_acts, batch_2_acts, m1_layers, m1_dmlp, m2_layers, m2_dmlp):
    device = 'cpu'
    m1_sum = t.zeros(
        (m1_layers, m1_dmlp), dtype=t.float64, device=device)
    m1_sum_sq = t.zeros(
        (m1_layers, m1_dmlp), dtype=t.float64, device=device)

    m2_sum = t.zeros(
        (m2_layers, m2_dmlp), dtype=t.float64, device=device)
    m2_sum_sq = t.zeros(
        (m2_layers, m2_dmlp), dtype=t.float64, device=device)

    m1_m2_sum = t.zeros(
        (m1_layers, m1_dmlp, m2_layers, m2_dmlp),
        dtype=t.float64, device=device
    )

    for l1 in range(batch_1_acts.shape[0]):
        # iterating over layers in batch_2_acts
        for l2 in range(batch_2_acts.shape[0]):
            layerwise_result = einops.einsum(
                batch_1_acts[l1], batch_2_acts[l2],
                'n1 t, n2 t -> n1 n2'
            )
            m1_m2_sum[l1, :, l2, :] += layerwise_result

    m1_sum += batch_1_acts.sum(dim=-1)
    m1_sum_sq += (batch_1_acts**2).sum(dim=-1)

    m2_sum += batch_2_acts.sum(dim=-1)
    m2_sum_sq += (batch_2_acts**2).sum(dim=-1)

    n = batch_1_acts.shape[-1]

    return t.tensor([m1_sum, m1_sum_sq, m2_sum, m2_sum_sq, m1_m2_sum, n])


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

        # iterating over layers in batch_1_acts
        for l1 in range(batch_1_acts.shape[0]):
            # iterating over layers in batch_2_acts
            batch_1_acts_l1 = batch_1_acts[l1].to(torch.float32)
            for l2 in range(batch_2_acts.shape[0]):
                # layerwise_result = einops.einsum(
                #     batch_1_acts_l1, batch_2_acts[l2].to(
                #         torch.float32), 'l1 t, l2 t -> l1 l2'
                # )
                # layerwise_result = t.einsum(
                #     'at,bt->ab',
                #     batch_1_acts[l1], batch_2_acts[l2],
                # )
                layerwise_result = torch.matmul(
                    batch_1_acts_l1, batch_2_acts[l2].to(torch.float32).T)
                self.m1_m2_sum[l1, :, l2, :] += layerwise_result.cpu()

        # self.m1_m2_sum += einops.einsum(
        #     batch_1_acts, batch_2_acts, 'l1 n1 t, l2 n2 t -> l1 n1 l2 n2'
        # )

        self.m1_sum += batch_1_acts.sum(dim=-1).cpu()
        self.m1_sum_sq += (batch_1_acts**2).sum(dim=-1).cpu()

        self.m2_sum += batch_2_acts.sum(dim=-1).cpu()
        self.m2_sum_sq += (batch_2_acts**2).sum(dim=-1).cpu()

        self.n += batch_1_acts.shape[-1].cpu()

    def update_correlation_data_bulk(self, data_list):
        data_list = t.stack(data_list)
        self.m1_sum += t.sum(data_list[:, 0])
        self.m1_sum_sq += t.sum(data_list[:, 1])
        self.m2_sum += t.sum(data_list[:, 2])
        self.m2_sum_sq += t.sum(data_list[:, 3])
        self.m1_m2_sum += t.sum(data_list[:, 4])
        self.n += t.sum(data_list[:, 5])

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


def get_activations_parallel(rank, model, dataset, conn, filter_padding=True):
    """Get the activations for a given model and dataset. 
    Inputs should already be appropriately batched
    inputs: (n_tokens, n_sequences) 512 x 32 by default
    out: (n_tokens, n_sequences, (n_layers * d_mlp))
    """
    # device = torch.device(f"cuda:{rank}")
    # model.to(device)

    activations_list = []
    print(f'cuda parallel start {rank}')
    for step, inputs in enumerate(tqdm.tqdm(dataset, position=rank, desc=f"rank {rank}")):
        inputs.to(rank)
        hooks = [
            (f'blocks.{layer_ix}.mlp.hook_post',
                partial(save_activation_hook, device=f"cuda:{rank}"))
            for layer_ix in range(model.cfg.n_layers)
        ]
        with t.no_grad():
            model.run_with_hooks(
                inputs,
                fwd_hooks=hooks,
                stop_at_layer=model.cfg.n_layers+1  # don't compute logits to save memory
            )
        activations = t.stack(
            [model.hook_dict[hook_pt[0]].ctx['activation'] for hook_pt in hooks], dim=2)
        model.reset_hooks()

        activations = einops.rearrange(
            activations, 'batch context l n -> l n (batch context)')

        if filter_padding:
            activations = activations[:, :, inputs.flatten() > 0]

        activations_list.append(activations.to('cuda:0'))

    conn.send(activations_list)
    print(f'cuda parallel end {rank}')


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

    start_time = time.time()
    # if args.baseline != 'gaussian':
    mp.set_start_method('spawn', force=True)

    chunk_size = args.batch_size
    for i in range(0, len(token_dataset['tokens']), chunk_size):
        cycle_start_time = time.time()
        # for i, chunk in enumerate(tqdm.tqdm(dataloader)):
        print(f"========= This is Chunk {i} ==========")

        dataloader1 = DataLoader(
            token_dataset['tokens'][i:i+chunk_size], batch_size=args.batch_size, shuffle=False)
        dataloader2 = DataLoader(
            token_dataset['tokens'][i:i+chunk_size], batch_size=args.batch_size, shuffle=False)
        print(f"chunk.shape {len(dataloader1)}")

        recv1, conn1 = mp.Pipe()
        recv2, conn2 = mp.Pipe()
        p1 = mp.Process(target=get_activations_parallel,
                        args=(0, model_1, dataloader1, conn1))
        p1.start()
        time.sleep(1)
        p2 = mp.Process(target=get_activations_parallel,
                        args=(1, model_2, dataloader2, conn2))
        p2.start()

        m1_flag = False
        m2_flag = False

        m1_activations_list = recv1.recv()
        m2_activations_list = recv2.recv()
        # while True:
        #     if m1_flag and m2_flag:
        #         break
        #     elif recv1.poll():
        #         m1_activations_list = recv1.recv()
        #         m1_flag = True
        #         print(f"m1_activations_list {len(m1_activations_list)}")
        #     elif recv2.poll():
        #         m2_activations_list = recv2.recv()
        #         m2_flag = True
        #         print(f"m2_activations_list {len(m2_activations_list)}")

        p1.join()
        p2.join()
        print(f"finished", flush=True)
        end_time = time.time()
        print(f"total gpu time is {end_time - cycle_start_time}", flush=True)

        if args.baseline == 'rotation':
            # TODO: consider making this actually orthogonal
            # eg, scipy.stats.special_ortho_group
            # see https://math.stackexchange.com/questions/3839152/sample-a-random-rotation-in-n-dimensions
            rotation_matrix = t.randn(
                (model_2.cfg.n_layers, model_2.cfg.d_mlp, model_2.cfg.d_mlp))
            rotation_matrix /= rotation_matrix.norm(dim=-1, keepdim=True)
            rotation_matrix = rotation_matrix.to(args.model_2_device)
        print(f"m1_activations {len(m1_activations_list)}")

        for m1_activations, m2_activations in tqdm.tqdm(
                zip(m1_activations_list, m2_activations_list), total=len(m1_activations_list)):

            if args.baseline == 'rotation':
                # rotate the neuron basis
                rotated_acts = []
                for l in range(m2_activations.shape[0]):
                    rotated_acts.append(
                        t.einsum(
                            'n t, m n -> m t',
                            m2_activations[l, :, :].to(rotation_matrix.device),
                            rotation_matrix[l, :, :]
                        ).to(args.correlation_device)
                    )
                m2_activations = t.stack(rotated_acts, dim=0)

            corr_computer.update_correlation_data(
                m1_activations, m2_activations)
        del m1_activations_list
        del m2_activations_list
        # m1_layers = model_1.cfg.n_layers
        # m2_layers = model_2.cfg.n_layers
        # m1_dmlp = model_1.cfg.d_mlp
        # m2_dmlp = model_2.cfg.d_mlp
        # with multiprocessing.Pool(processes=8) as pool:
        #     data = [(m1_activations_list[i], m2_activations_list[i], m1_layers, m1_dmlp, m2_layers, m2_dmlp)
        #             for i in range(len(m1_activations_list))]
        #     results = pool.starmap(update_correlation_data,
        #                            data)

        #     pool.close()
        #     pool.join()

        # print("done with cpu multiprocessing")
        # corr_computer.update_correlation_data_bulk(results)

        end_time = time.time()
        print(f"total gpu time is {end_time - cycle_start_time}", flush=True)
        time.sleep(1)

    correlation = corr_computer.compute_correlation()
    end_time = time.time()
    print(f"total time is {end_time - start_time}", flush=True)
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
        '--model_1_device', type=str, default='cuda:0')
    parser.add_argument(
        '--model_2_device', type=str, default='cuda:1')
    parser.add_argument(
        '--correlation_device', type=str, default='cpu')
    # TODO: properly implement these
    parser.add_argument(
        '--save_precision', type=int, default=16, choices=[8, 16, 32],
        help='Number of bits to use for saving correlation matrix')
    parser.add_argument(
        '--checkpoint_dir', default='results/correlations')

    args = parser.parse_args()
    print(f"Visible CUDA devices: {t.cuda.device_count()}", flush=True)
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

    # TODO: consider saving by layer and delegate to stream computers
    torch.save(
        adjust_precision(correlation, args.save_precision),
        os.path.join(save_path, 'correlation.pt')
    )
