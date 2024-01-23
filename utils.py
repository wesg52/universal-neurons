import torch
import datetime

PILE_DATASETS = [
    'pile.test.arxiv.512',
    'pile.test.bookcorpus2.512',
    'pile.test.books3.512',
    'pile.test.dm_mathematics.512',
    'pile.test.enron_emails.512',
    'pile.test.europarl.512',
    'pile.test.freelaw.512',
    'pile.test.github.512',
    'pile.test.gutenberg.512',
    'pile.test.hackernews.512',
    'pile.test.nih_exporter.512',
    'pile.test.opensubtitles.512',
    'pile.test.openwebtext2.512',
    'pile.test.philpapers.512',
    'pile.test.pile_cc.512',
    'pile.test.pubmed_abstracts.512',
    'pile.test.pubmed_central.512',
    'pile.test.stackexchange.512',
    'pile.test.ubuntu_irc.512',
    'pile.test.uspto_backgrounds.512',
    'pile.test.wikipedia.512',
    'pile.test.youtubesubtitles.512'
]

MODEL_FAMILIES = ['pythia', 'gpt2']


def get_model_family(model_name):
    for family in MODEL_FAMILIES:
        if family in model_name:
            return family
    raise ValueError(f'Invalid model name: {model_name}')


def timestamp():
    return datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")


def vector_histogram(values, bin_edges):
    bin_index = torch.searchsorted(bin_edges, values)
    bin_counts = torch.zeros(values.shape[0], len(
        bin_edges)+1, dtype=torch.int32)

    return bin_counts.scatter_add_(
        -1, bin_index, torch.ones_like(bin_index, dtype=torch.int32))


def vector_moments(values, dim=1):
    mean = torch.mean(values, dim=dim)
    diffs = values - mean[:, None]
    var = torch.mean(torch.pow(diffs, 2.0), dim=dim)
    std = torch.pow(var, 0.5)
    zscore = diffs / std[:, None]
    skew = torch.mean(torch.pow(zscore, 3.0), dim=dim)
    kurt = torch.mean(torch.pow(zscore, 4.0), dim=dim)
    return mean, var, skew, kurt


def adjust_precision(activation_tensor, output_precision=8, per_channel=True, cos_sim=False):
    '''
    Adjust the precision of the activation subset
    '''
    if output_precision == 64:
        return activation_tensor.to(torch.float64)

    elif output_precision == 32:
        return activation_tensor.to(torch.float32)

    elif output_precision == 16:
        return activation_tensor.to(torch.float16)

    elif output_precision == 8 and not per_channel:
        min_val = activation_tensor.min().item() if not cos_sim else -1
        max_val = activation_tensor.max().item() if not cos_sim else 1
        num_quant_levels = 2**output_precision
        scale = (max_val - min_val) / (num_quant_levels - 1)
        zero_point = round(-min_val / scale)
        return torch.quantize_per_tensor(
            activation_tensor, scale, zero_point, torch.quint8)

    elif output_precision == 8 and per_channel:
        min_vals = activation_tensor.min(dim=0)[0] if not cos_sim else -1
        max_vals = activation_tensor.max(dim=0)[0] if not cos_sim else 1
        num_quant_levels = 2**output_precision
        scale = (max_vals - min_vals) / (num_quant_levels - 1)
        zero_point = torch.round(-min_vals / scale)
        return torch.quantize_per_channel(
            activation_tensor, scale, zero_point, 1, torch.quint8)

    elif output_precision == 1:
        return activation_tensor > 0

    else:
        raise ValueError(f'Invalid output precision: {output_precision}')
