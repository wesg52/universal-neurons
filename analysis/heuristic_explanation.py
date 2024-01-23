import tqdm
import numpy as np
import pandas as pd


def compute_binary_variance_reduction(activation_df, neuron_cols):
    neuron_variance = activation_df[neuron_cols].var(axis=0)
    feature_variance = activation_df.groupby('feature')[neuron_cols].var().T
    feature_count = activation_df.groupby('feature').size()

    false_ratio = feature_count[False] / (feature_count[True] + feature_count[False])
    true_ratio = 1 - false_ratio

    split_variance = false_ratio * feature_variance[False] + true_ratio * feature_variance[True]

    variance_reduction = (neuron_variance - split_variance) / neuron_variance
    return variance_reduction


def compute_feature_variance_reduction_df(activation_df, feature_df, neuron_cols, feature_type='token', prev_token=False):
    binary_feature_cols = feature_df.columns[feature_df.dtypes == bool]

    variance_reduction_dict = {}
    for feature_col in tqdm.tqdm(binary_feature_cols):

        if feature_type == 'token':
            feature_tokens = feature_df.index[feature_df[feature_col]].values
            token_col = 'token' if not prev_token else 'prev_token'
            activation_df['feature'] = activation_df[token_col].isin(feature_tokens)
        elif feature_type == 'sequence':
            feature_vals = feature_df[feature_col].values
            if prev_token:
                feature_vals = np.roll(feature_vals, 1)
            activation_df['feature'] = feature_vals

        else:
            raise ValueError(f'feature_type {feature_type} not recognized')

        variance_reduction = compute_binary_variance_reduction(activation_df, neuron_cols)
        variance_reduction_dict[feature_col] = variance_reduction

    var_reduction_df = pd.DataFrame(variance_reduction_dict)
    var_reduction_df.index.name = 'neuron'

    return var_reduction_df


def compute_mean_dif_df(activation_df, feature_df, neuron_cols, prev_token=False):
    binary_feature_cols = feature_df.columns[feature_df.dtypes == bool]

    mean_dif_dict = {}
    for feature_col in tqdm.tqdm(binary_feature_cols):
        feature_tokens = feature_df.index[feature_df[feature_col]].values

        token_col = 'token' if not prev_token else 'prev_token'
        activation_df['feature'] = activation_df[token_col].isin(feature_tokens)

        neuron_mean = activation_df[neuron_cols].mean(axis=0)
        feature_mean = activation_df.groupby('feature')[neuron_cols].mean().T

        mean_dif = (feature_mean[True] - feature_mean[False]) / neuron_mean

        mean_dif_dict[feature_col] = mean_dif

    mean_dif_df = pd.DataFrame(mean_dif_dict)
    mean_dif_df.index.name = 'neuron'

    return mean_dif_df

