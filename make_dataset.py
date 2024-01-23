import requests
import jsonlines
import zstandard
from transformer_lens.utils import tokenize_and_concatenate, get_dataset
from utils import get_model_family
import argparse
import os
import io
import math
import datasets
import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate


DATASET_ALIASES = {
    "openwebtext": "stas/openwebtext-10k",
    "owt": "stas/openwebtext-10k",
    "pile": "NeelNanda/pile-10k",
    "c4": "NeelNanda/c4-10k",
    "code": "NeelNanda/code-10k",
    "python": "NeelNanda/code-10k",
    "c4_code": "NeelNanda/c4-code-20k",
    "c4-code": "NeelNanda/c4-code-20k",
    "wiki": "NeelNanda/wiki-10k",
}

PILE_SUBSET_ALIASES = {
    'ArXiv': 'arxiv',
    'BookCorpus2': 'bookcorpus2',
    'Books3': 'books3',
    'DM Mathematics': 'dm_mathematics',
    'Enron Emails': 'enron_emails',
    'EuroParl': 'europarl',
    'FreeLaw': 'freelaw',
    'Github': 'github',
    'Gutenberg (PG-19)': 'gutenberg',
    'HackerNews': 'hackernews',
    'NIH ExPorter': 'nih_exporter',
    'OpenSubtitles': 'opensubtitles',
    'OpenWebText2': 'openwebtext2',
    'PhilPapers': 'philpapers',
    'Pile-CC': 'pile_cc',
    'PubMed Abstracts': 'pubmed_abstracts',
    'PubMed Central': 'pubmed_central',
    'StackExchange': 'stackexchange',
    'USPTO Backgrounds': 'uspto_backgrounds',
    'Ubuntu IRC': 'ubuntu_irc',
    'Wikipedia (en)': 'wikipedia',
    'YoutubeSubtitles': 'youtubesubtitles'
}


def get_pile_split(split='test'):
    PILE_URL = f'https://the-eye.eu/public/AI/pile/{split}.jsonl.zst'
    # Download the file
    response = requests.get(PILE_URL, stream=True)
    response.raise_for_status()  # Ensure we got a valid response

    # Prepare a streaming decompression context
    dctx = zstandard.ZstdDecompressor()
    stream_reader = dctx.stream_reader(io.BytesIO(response.content))

    # Wrap the binary stream reader with a TextIOWrapper so jsonlines can read it
    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

    lines = []
    # Process the JSON lines file
    with jsonlines.Reader(text_stream) as reader:
        for obj in reader:
            lines.append(
                {'text': obj['text'], 'subset': obj['meta']['pile_set_name']})
    ds = datasets.Dataset.from_list(lines)

    return ds


def tokenize_pile_subsets(pile_ds, model, ctx_len=512):
    seq_char_len = np.array([len(t) for t in pile_ds['text']])
    valid_ixs = np.arange(len(pile_ds))[seq_char_len > 50]
    ds = pile_ds.select(valid_ixs)

    seq_subset = np.array(ds['subset'])
    subsets = np.unique(seq_subset)

    sub_ds_dict = {}
    print(subsets)
    for subset in subsets:
        print('Tokenizing subset:', subset)
        mask = seq_subset == subset
        sub_ds = ds.select(np.arange(len(ds))[mask])
        sub_ds_tokens = tokenize_and_concatenate(
            sub_ds, model.tokenizer, max_length=ctx_len)

        # format = {'type': 'torch', 'format_kwargs': {'dtype': torch.int}}
        # sub_ds_tokens.set_format(**format)

        sub_ds_dict[subset] = sub_ds_tokens

    return sub_ds_dict


def create_pile_subset(model_family, n_tokens, n_tokens_name):
    base_path = os.path.join('token_datasets', model_family)
    dsets = []
    for ds_file in os.listdir(base_path):
        if 'all' in ds_file:
            continue
        ds = datasets.load_from_disk(os.path.join(base_path, ds_file))
        parent_ds, split, subset, ctx_len = ds_file.split('.')
        ds = ds.add_column('subset', [subset for _ in range(len(ds))])
        dsets.append(ds)

    all_ds = datasets.concatenate_datasets(dsets)

    ctx_len = len(all_ds[0]['tokens'])
    n_sequences = math.ceil(n_tokens / ctx_len)

    subsample_ds = all_ds.shuffle().select(range(n_sequences))

    save_name = f'pile.test.all-{n_tokens_name}.{ctx_len}'
    save_path = os.path.join(base_path, save_name)

    subsample_ds.save_to_disk(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--hf_dataset', default='EleutherAI/pile', help='Name of HuggingFace dataset')
    parser.add_argument(
        '--hf_dataset_split', default='test')
    parser.add_argument(
        '--ctx_len', default=512, type=int, help='Context length')
    parser.add_argument(
        '--n_seq', default=-1, type=int, help='Number of sequences')
    parser.add_argument(
        '--output_dir', default='token_datasets', help='Path to save dataset')

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model, device='cpu')
    model_family = get_model_family(args.model)

    # get data
    if args.hf_dataset in DATASET_ALIASES:
        dataset = get_dataset(args.hf_dataset)
    elif args.hf_dataset == 'EleutherAI/pile':
        print('Downloading pile')
        dataset = get_pile_split(args.hf_dataset_split)
    else:
        dataset = datasets.load_dataset(
            args.hf_dataset, split=args.hf_dataset_split, streaming=True)

    # tokenize and save
    if args.hf_dataset == 'EleutherAI/pile':
        ds_dict = tokenize_pile_subsets(dataset, model, ctx_len=args.ctx_len)
        for subset, sub_ds in ds_dict.items():
            subset_name = PILE_SUBSET_ALIASES[subset]
            save_path = os.path.join(
                args.output_dir, model_family,
                f'pile.{args.hf_dataset_split}.{subset_name}.{args.ctx_len}'
            )
            os.makedirs(save_path, exist_ok=True)
            sub_ds.save_to_disk(save_path)
    else:
        if args.n_seq > 0:
            dataset = dataset.select(range(args.n_seq))

        token_dataset = tokenize_and_concatenate(
            dataset, model.tokenizer, max_length=args.ctx_len)

        save_path = os.path.join(
            args.output_dir, model_family,
            f'{args.hf_dataset}.{args.n_seq}.{args.ctx_len}'
        )
        os.makedirs(save_path, exist_ok=True)
        token_dataset.save_to_disk(save_path)
