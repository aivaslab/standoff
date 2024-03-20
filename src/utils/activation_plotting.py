import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns


def plot_info_matrices(confusion_matrix, confusion_matrix_n, models, percentile, path, ig_matrix):
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="coolwarm_r",
                xticklabels=models, yticklabels=models, vmin=0, vmax=1)
    plt.title('Probability B is Correct, Given A is Correct')
    plt.xlabel('Feature B')
    plt.ylabel(f'Feature A (given condition correct, i.e. <{percentile} percentile MSE)')
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix_n, annot=True, fmt=".2f", cmap="coolwarm_r",
                xticklabels=models, yticklabels=models, vmin=0, vmax=1)
    plt.title('Probability B is Correct, Given A is Incorrect')
    plt.xlabel('Feature B')
    plt.ylabel(f'Feature A (given condition incorrect, i.e. >= {percentile} percentile MSE)')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"mlp-confusion.png"))

    plt.figure(figsize=(10, 8))
    sns.heatmap((confusion_matrix - confusion_matrix_n) / (confusion_matrix + confusion_matrix_n), annot=True,
                fmt=".2f", cmap="coolwarm_r",
                xticklabels=models, yticklabels=models, vmin=-1, vmax=1)
    plt.title('Influence of A on B')
    plt.xlabel('Feature B')
    plt.ylabel('Feature A')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"mlp-influence.png"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(ig_matrix, annot=True, fmt=".2f", cmap="coolwarm_r", xticklabels=models, yticklabels=models, vmin=-1,
                vmax=1)
    plt.title('Information gain of B given A')
    plt.xlabel('Feature B')
    plt.ylabel('Feature A')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"mlp-infogain.png"))


def plot_histogram(path, name, target, histogram, models_used):
    histogram_used = [np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan) for arr in histogram]

    labels_array = np.concatenate([[mod] * len(data) for mod, data in zip(models_used, histogram_used)])
    flattened = np.concatenate(histogram_used)
    data_for_plotting = pd.DataFrame({
        'Values': flattened,
        'Labels': labels_array
    })
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(data=data_for_plotting, x='Values', bins=20, binrange=(0, 0.99), shrink=0.8, hue='Labels', hue_order=models_used, element="bars", stat="count", multiple='dodge')
    plt.title('Comparison of Validation MSEs')
    plt.xlabel('NMSE Value')
    plt.ylabel('Count')
    plt.legend(title='Matrix')
    ax.legend(labels=models_used, title='Models', bbox_to_anchor=(0.6, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{name}hist-{target}.png"))


def plot_bars(histogram, histogram2, path, name, target, models_used):
    plt.figure(figsize=(12, 8))
    num_models = len(histogram)

    means_minregime = [np.mean(minregime) for minregime in histogram]
    stds_minregime = [np.std(minregime) for minregime in histogram]
    means_allregimes = [np.mean(allregimes) for allregimes in histogram2]
    stds_allregimes = [np.std(allregimes) for allregimes in histogram2]

    print(means_minregime)
    print(means_allregimes)

    bar_width = 0.3
    index = np.arange(num_models)
    bar1_pos = index - bar_width / 2
    bar2_pos = index + bar_width / 2

    plt.bar(bar1_pos, means_minregime, bar_width, yerr=stds_minregime, capsize=5, label='MinRegime', alpha=0.6)
    plt.bar(bar2_pos, means_allregimes, bar_width, yerr=stds_allregimes, capsize=5, label='AllRegimes', alpha=0.6)

    plt.xlabel('Models')
    plt.ylabel('NMSE')
    plt.title('Comparison between MinRegime and AllRegimes for Each Model')
    plt.xticks(index, models_used, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{name}bar-{target}.png"))


def plot_scatter(histogram, histogram2, keys, keys1, min_matrix_f2f, models_used, path, name, target):
    for tiny in [True, False]:
        plt.figure(figsize=(12, 8))
        for reg, la in zip(histogram, models_used):
            ordered = sorted(reg)
            plt.plot(ordered, label=la)
        plt.legend()
        plt.xlabel('nth lowest NMSE pair')
        plt.ylabel('NMSE')
        plt.ylim((-0.05, 1.1))

        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{name}step-{target}.png"))

        plt.figure(figsize=(10, 10))
        plt.plot([(0, 0), (1, 1)], c='black', linestyle='dashed', label='_nolegend_')
        differences = []
        num_rows, num_columns = min_matrix_f2f.shape
        for i, (minregime, allregimes) in enumerate(zip(histogram, histogram2)):
            for j, (min_val, all_val) in enumerate(zip(minregime, allregimes)):
                diff = min_val - all_val

                original_row = j // num_columns
                original_column = j % num_columns
                if original_row != original_column and min_val < 0.75:
                    differences.append((diff, i, original_row, original_column, j))  # Store difference along with indices
        top_differences = sorted(differences, reverse=False)[:3]
        print(top_differences)

        for i, (minregime, allregimes) in enumerate(zip(histogram, histogram2)):
            plt.scatter(allregimes, minregime, s=25, alpha=0.6)
            for diff, hist_idx, or_row, or_col, j in top_differences:
                if i == hist_idx:  # Check if this point belongs to the current histogram
                    plt.annotate(f'{keys[or_row]},{keys1[or_col]}', (allregimes[j], minregime[j]))
                    # todo: save models used for the best one

        plt.xlabel('NMSE (trained on all)')
        plt.ylabel('NMSE (minimum of contrastive regimes)')
        plt.tight_layout()
        plt.ylim((-0.05, 1.1))
        plt.xlim((-0.05, 1.1))
        plt.legend(title='Model', labels=models_used)
        plt.savefig(os.path.join(path, f"{name}scatter-{target}.png"))
