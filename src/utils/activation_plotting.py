import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


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
    #plt.style.use('ggplot')
    plt.savefig(os.path.join(path, f"{name}hist-{target}.png"))


def plot_bars(histogram, histogram2, path, name, target, models_used):
    plt.figure(figsize=(12, 6))
    num_models = len(histogram)

    means_minregime = [np.mean(minregime) for minregime in histogram]
    stds_minregime = [np.std(minregime) for minregime in histogram]
    means_allregimes = [np.mean(allregimes) for allregimes in histogram2]
    stds_allregimes = [np.std(allregimes) for allregimes in histogram2]

    lowers_minregime = [np.percentile(minregime, 25) for minregime in histogram]
    uppers_minregime = [np.percentile(minregime, 75) for minregime in histogram]
    lowers_allregimes = [np.percentile(allregimes, 25) for allregimes in histogram2]
    uppers_allregimes = [np.percentile(allregimes, 75) for allregimes in histogram2]

    print(means_minregime)
    print(means_allregimes)

    bar_width = 0.3
    index = np.arange(num_models)
    bar1_pos = index - bar_width / 2
    bar2_pos = index + bar_width / 2

    plt.bar(bar1_pos, means_minregime, bar_width, yerr=[np.subtract(means_minregime, lowers_minregime), np.subtract(uppers_minregime, means_minregime)], capsize=5, label='min', alpha=1)
    plt.bar(bar2_pos, means_allregimes, bar_width, yerr=[np.subtract(means_allregimes, lowers_allregimes), np.subtract(uppers_allregimes, means_allregimes)], capsize=5, label='all', alpha=1)

    plt.xlabel('Models')
    plt.ylabel('NMSE')
    plt.title('Comparison between All and Min for each model')
    plt.xticks(index, models_used, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{name}bar-{target}.png"))

    # difference fig

    plt.figure(figsize=(12, 6))
    num_models = len(histogram)

    diff = []
    for h1, h2 in zip(histogram, histogram2):
        diff.append(h1 - h2)

    means_diff = [np.mean(minregime) for minregime in diff]
    stds_minregime = [np.std(minregime) for minregime in diff]
    lowers_diff = [np.percentile(minregime, 25) for minregime in diff]
    uppers_diff = [np.percentile(minregime, 75) for minregime in diff]

    print(means_minregime)

    bar_width = 0.5
    index = np.arange(num_models)
    bar1_pos = index - bar_width / 2

    plt.bar(bar1_pos, means_diff, bar_width, yerr=[np.subtract(means_diff, lowers_diff), np.subtract(uppers_diff, means_diff)], capsize=5, label='difference', alpha=1)

    plt.xlabel('Models')
    plt.ylabel('NMSE')
    plt.title('Difference between All and Min for each model ')
    plt.xticks(index, models_used, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{name}bar-diff-{target}.png"))


def plot_scatter(histogram, histogram2, keys, keys1, min_matrix_f2f, models_used, path, name, target):
    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
        '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
        '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]

    for tiny in [True, False]:
        plt.figure(figsize=(12, 8))
        i=0
        for reg, la in zip(histogram, models_used):
            ordered = sorted(reg)
            plt.plot(ordered, label=la, c=colors[i])
            i+=1
        plt.legend()
        plt.xlabel('nth lowest NMSE pair')
        plt.ylabel('NMSE')
        plt.ylim((-0.05, 1.1))
        if tiny:
            plt.ylim((-0.01, 0.2))
            plt.xlim((-0.01, 45))
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"{name}step-{target}-s.png"))
        else:
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
                if min_val < 0.75:
                    differences.append((diff, i, original_row, original_column, j))  # Store difference along with indices
        top_differences = sorted(differences, reverse=False)[:10]
        print(top_differences)

        for i, (minregime, allregimes) in enumerate(zip(histogram, histogram2)):
            plt.scatter(minregime, allregimes, s=25, alpha=0.6, c=colors[i])
            for diff, hist_idx, or_row, or_col, j in top_differences:
                if i == hist_idx:  # Check if this point belongs to the current histogram
                    plt.annotate(f'{keys[or_row]},{keys1[or_col]}', (minregime[j], allregimes[j]))
                    # todo: save models used for the best one

        plt.ylabel('NMSE (trained on all)')
        plt.xlabel('NMSE (minimum of contrastive regimes)')
        plt.tight_layout()
        plt.legend(title='Model', labels=models_used)
        #plt.style.use('ggplot')

        if tiny:
            plt.ylim((-0.001, 0.011))
            plt.xlim((-0.001, 0.011))
            plt.savefig(os.path.join(path, f"{name}scatter-{target}-s.png"))
        else:
            plt.ylim((-0.05, 1.1))
            plt.xlim((-0.05, 1.1))
            plt.savefig(os.path.join(path, f"{name}scatter-{target}.png"))

    # do mlp2 vs anything ones:
    for tiny in [True, False]:
        for min in [True, False]:
            plt.figure(figsize=(10, 10))
            plt.plot([(0, 0), (1, 1)], c='black', linestyle='dashed', label='_nolegend_')

            for i, x in enumerate(histogram[1:] if min else histogram2[1:]):
                plt.scatter(histogram[0] if min else histogram2[0], x, s=25, alpha=0.6, c=colors[i])
            plt.ylabel('NMSE (other)')
            plt.xlabel('NMSE (mlp2)')
            plt.tight_layout()
            plt.legend(title='Model', labels=models_used[1:])
            #plt.style.use('ggplot')

            if tiny:
                plt.ylim((-0.001, 0.011))
                plt.xlim((-0.001, 0.011))
                plt.savefig(os.path.join(path, f"{name}scatter-d-{target}-s-{min}.png"))
            else:
                plt.ylim((-0.05, 1.1))
                plt.xlim((-0.05, 1.1))
                plt.savefig(os.path.join(path, f"{name}scatter-d-{target}-{min}.png"))
