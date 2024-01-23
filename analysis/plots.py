import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


def token_histogram_by_class(
    values, classes, class_labels,
    logy=False, plot_dist=True, n_bins=50, ax=None, legend_title='token', legend_loc='best'
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    _, bins = np.histogram(values, bins=n_bins)

    for class_ix in range(len(class_labels)):
        class_mask = classes == class_ix
        class_values = values[class_mask]
        if plot_dist:
            count, _ = np.histogram(class_values, bins=bins)
            dist = count / count.sum()
            ax.hist(bins[:-1], bins, weights=dist,
                    alpha=0.5, label=class_labels[class_ix])
        else:
            ax.hist(class_values, bins=bins, alpha=0.5,
                    label=class_labels[class_ix])

    if logy:
        ax.set_yscale('log')
    if legend_loc == 'skew':
        legend_loc = 'upper left' if scipy.stats.skew(
            values) < 0 else 'upper right'
    ax.legend(title=legend_title, loc=legend_loc)
    ax.set_xlabel('activation')
    return ax
