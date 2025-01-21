import math
import pickle
import re

import numpy as np
import torch
import umap
from matplotlib import pyplot as plt, gridspec
from scipy.stats import pearsonr, stats
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import pandas as pd
import json
import seaborn as sns
from stable_baselines3.common.monitor import get_monitor_files
import matplotlib.colors as mcolors


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    plt.close()


def plot_regime_lengths(regime_lengths, grouped, savepath):
    regime_names = list(grouped.index)
    accuracies = list(grouped.values)

    # Matching regimes and extracting lengths
    lengths = [regime_lengths[regime] for regime in regime_names]

    # Plotting
    plt.scatter(lengths, accuracies)
    for i, regime in enumerate(regime_names):
        plt.annotate(regime, (lengths[i], accuracies[i]))

    plt.xlabel('Regime Length')
    plt.ylabel('Accuracy')
    plt.title('Scatterplot of Regime Length vs. Accuracy')
    plt.savefig(savepath)
    plt.close()


def plot_split(indexer, df, mypath, title, window, values=None, use_std=True):
    if values is None:
        values = ["accuracy"]
    new_df = df.pivot(index=indexer, columns="configName", values=[x + "_mean" for x in values] if use_std else values)
    fig = plt.figure(title)
    new_df.plot()
    plt.xlabel('Timestep')
    plt.ylabel(values)
    plt.xlim(0, plt.xlim()[1])
    plt.ylim(0, 1)
    plt.title(title + " " + values[0])
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    name = title + values[0]
    plt.savefig(os.path.join(mypath, name))
    plt.close()


def plot_progression(path, save_path):
    with open(path, 'rb') as f:
        loaded_accuracies = pickle.load(f)

    blue_gradient = mcolors.LinearSegmentedColormap.from_list("blue_gradient", ["darkblue", "lightblue"])
    orange_gradient = mcolors.LinearSegmentedColormap.from_list("orange_gradient", ["darkorange", "lightsalmon"])

    oracle_gradients = {
        '0': blue_gradient,
        '1': orange_gradient
    }
    legend_handles = {'0': None, '1': None}

    print(loaded_accuracies)
    n = len(loaded_accuracies) // 2
    print('n', n)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.1 * (n + 1), 2.5))
    fig.subplots_adjust(wspace=0.06)

    mean_accuracies = {'0': [], '1': []}
    timestep_counts = {'0': 0, '1': 0}

    # Calculate mean accuracies for each label
    for key, values in loaded_accuracies.items():
        oracle = key.split('_')[0]
        if values:  # Ensure non-empty values
            if len(mean_accuracies[oracle]) == 0:
                mean_accuracies[oracle] = np.zeros(len(values))
            mean_accuracies[oracle] += values
            timestep_counts[oracle] += 1

    for oracle in mean_accuracies:
        if timestep_counts[oracle] > 0:
            mean_accuracies[oracle] /= timestep_counts[oracle]

    for k, (key, values) in enumerate(loaded_accuracies.items()):
        oracle = key.split('_')[0]
        num_points = len(values)
        colors = oracle_gradients[oracle](np.linspace(0, 1, num_points))
        x_values = range(num_points)

        label = 'No Oracle' if oracle == '0' else 'Oracle' if legend_handles[oracle] is None else None

        for cur, ax in enumerate([axes[k // 2]]):  # , axes[-1]
            ax.set_xticks(range(10))
            ax.set_ylim(0.4, 1.0)
            if k // 2 == 0 and cur == 0:
                ax.set_yticks(np.arange(0.4, 1.05, 0.2))
                ax.set_ylabel('Mean Accuracy (op)')
            else:
                ax.set_yticks([])
            if num_points > 0:
                line, = ax.plot(x_values, values, color=colors[k], label=label)

                if label:
                    legend_handles[oracle] = line

        axes[-1].set_xticks(range(10))
        axes[-1].set_ylim(0.4, 1.0)
        axes[-1].set_yticks([])
        axes[-1].plot(x_values, mean_accuracies[oracle], '--', color=colors[k], label=label)

    valid_handles = [handle for handle in [legend_handles['0'], legend_handles['1']] if handle is not None]
    valid_labels = ['No Oracle' if handle == legend_handles['0'] else 'Oracle' for handle in valid_handles]

    plt.legend(handles=valid_handles, labels=valid_labels, loc='lower right')

    # plt.title('Progression Trial Accuracies')
    plt.tight_layout(rect=[0, 0.1, 1, 1.0])
    fig.text(0.5, 0.05, 'Number of Opponent Regimes', ha='center', va='center', fontsize=12)
    plt.savefig(save_path)
    plt.close()


def plot_merged(indexer, df, mypath, title, window, values=None,
                labels=None, _range=None, use_std=True, scatter_dots=True, stacked_bar=False):
    if _range is None:
        _range = [0, 1]
    if labels is None:
        labels = ["selected any box", "selected best box"]
    if values is None:
        values = ["valid", "accuracy"]
    fig = plt.figure(title)
    plt.ylim(_range[0], _range[1])

    colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if stacked_bar:
        bar_width = (max(df[indexer]) - min(df[indexer])) / (
                2 * len(df['selectedBig_mean']) + 1)  # (len(df['selectedBig_mean']) * 2 + 1
        values_mean = [df[value + "_mean"] for value in values]
        plt.bar(df[indexer], values_mean[0], label=labels[0], width=bar_width, color=colors[0])
        for i in range(1, len(values)):
            plt.bar(df[indexer], values_mean[i], bottom=sum(values_mean[:i]), label=labels[i],
                    width=bar_width, color=colors[i % len(colors)])
    else:
        for i, (value, label) in enumerate(zip(values, labels)):
            color = colors[i % len(colors)]
            if use_std:
                plt.plot(df[indexer], df[value + "_mean"], label=label, color=color)
                if scatter_dots:
                    plt.scatter(x=df[indexer], y=df[value + "_mean"], label=label, color=color)
                plt.fill_between(df[indexer], df[value + "_mean"] - df[value + "_std"],
                                 df[value + "_mean"] + df[value + "_std"], alpha=.1, color=color)
            else:
                plt.plot(df[indexer], df[value], label=label, color=color)
                if scatter_dots:
                    plt.scatter(x=df[indexer], y=df[value], label=label, color=color)
    plt.xlim(0, plt.xlim()[1])
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 1))
    plt.xlabel('Timestep')
    plt.ylabel('Percent')
    plt.title(title + " " + values[0])
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.tight_layout()
    plt.savefig(os.path.join(mypath, title + "-merged-" + values[0] + '.png'))
    plt.close()


def plot_selection(indexer, df, mypath, title, window, bars=False):
    fig = plt.figure(title)
    plt.xlabel('Timestep')
    plt.ylabel('Reward Type')
    plt.plot([], [], label='SelectedBig', color='green')
    plt.plot([], [], label='SelectedSmall', color='blue')
    plt.plot([], [], label='SelectedNeither', color='orange')
    if bars:
        # make stacked bar plot, where bar width is a proportion of the total width
        bar_width = max(df[indexer]) / (len(df['selectedBig_mean']) * 2 + 1)
        plt.bar(df[indexer], df["selectedBig_mean"], color='green', width=bar_width)
        plt.bar(df[indexer], df["selectedSmall_mean"], bottom=df["selectedBig_mean"], color='blue', width=bar_width)
        plt.bar(df[indexer], df["selectedNeither_mean"], bottom=df["selectedBig_mean"] + df["selectedSmall_mean"],
                color='orange', width=bar_width)
    else:
        plt.stackplot(df[indexer], df["selectedBig_mean"], df["selectedSmall_mean"], df["selectedNeither_mean"],
                      colors=['green', 'blue', 'orange', ])
    # plt.scatter(x=df[indexer], y=df["selectedBig_mean"])
    plt.xlim(0, plt.xlim()[1])
    plt.legend(['Big', 'Small', 'Neither'])
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.title(title + "-" + str('Reward Type'))
    plt.tight_layout()
    plt.savefig(os.path.join(mypath, title + "-" + 'reward-type' + '.png'))
    plt.close()


def plot_train(log_folder, window=1000):
    monitor_files = get_monitor_files(log_folder)
    print('plotting train', monitor_files)

    for file_name in monitor_files:
        plt.figure()
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#", print(first_line)
            metadata = json.loads(first_line[1:].replace('""', ''))
            rank = metadata['rank'] if 'rank' in metadata.keys() else -1
            header = json.loads(first_line[1:])
            df = pd.read_csv(file_handler, index_col=None, on_bad_lines='skip')  # , usecols=cols)
            if len(df):
                print('plotting train curve', file_name)
                df['index_col'] = df.index
                title = os.path.basename(file_name).replace('.', '-')[:-12]

                x = df.index
                y = df.r
                plt.scatter(x, y, marker='.', alpha=0.3, label='episodes')
                df['yrolling'] = df['r'].rolling(window=window).mean()
                plt.plot(x, df.yrolling, color='red', label='moving average, window=' + str(window))

                plt.rcParams["figure.figsize"] = (10, 5)
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title(title + "-" + str('Learning curve'))
                plt.tight_layout()
                plt.legend()
                if not os.path.exists(os.path.join(log_folder, 'figures')):
                    os.mkdir(os.path.join(log_folder, 'figures'))
                plt.savefig(os.path.join(log_folder, 'figures', title + "-" + 'lcurve' + '.png'))
            plt.close()


def plot_train_many(train_paths, window=1000, path=None):
    for col, name in [("index", "Episode"), ("t", "realtime")]:
        plt.figure()
        for log_folder in train_paths:
            monitor_files = get_monitor_files(log_folder)

            for file_name in monitor_files:
                with open(file_name) as file_handler:
                    first_line = file_handler.readline()
                    assert first_line[0] == "#", print(first_line)
                    df = pd.read_csv(file_handler, index_col=None, on_bad_lines='skip')  # , usecols=cols)
                    if len(df):
                        if col == 'index':
                            df['index_col'] = df.index
                            realcol = 'index_col'
                        else:
                            realcol = col
                        df['yrolling'] = df['r'].rolling(window=window).mean()
                        plt.plot(df[realcol], df.yrolling, label=os.path.basename(log_folder))

        plt.rcParams["figure.figsize"] = (15, 5)
        plt.gcf().set_size_inches(15, 5)
        plt.xlabel(name)
        plt.ylabel('Reward')
        plt.title('Learning curve (smoothed window = ' + str(window) + ')')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'all_trains_' + name + '.png'))
        plt.close()


def plot_train_curriculum(start_paths, train_paths, window=1000, path=None):
    plt.figure()
    max_episode = 0  # to track maximum episode seen so far
    col = "index"

    for log_folder in start_paths:
        monitor_files = get_monitor_files(log_folder)

        for file_name in monitor_files:
            with open(file_name) as file_handler:
                first_line = file_handler.readline()
                assert first_line[0] == "#", print(first_line)
                df = pd.read_csv(file_handler, index_col=None, on_bad_lines='skip')  # , usecols=cols)
                if len(df):
                    if col == 'index':
                        df['index_col'] = df.index
                        realcol = 'index_col'
                    else:
                        realcol = col
                    # df['index_col'] = df.index + max_episode  # shift the episode numbers
                    df['yrolling'] = df['r'].rolling(window=window).mean()
                    plt.scatter(df.index, df.r, marker='.', alpha=0.05, s=0.1, label=os.path.basename(log_folder))
                    plt.plot(df[realcol], df.yrolling, label=os.path.basename(log_folder))
        max_episode = df['index_col'].max()

        # we want the last 1000 datapoints in df to be in df_start:
        df_start = df.iloc[-1000:]

    for log_folder in train_paths:
        monitor_files = get_monitor_files(log_folder)
        for file_name in monitor_files:
            with open(file_name) as file_handler:
                first_line = file_handler.readline()
                assert first_line[0] == "#", print(first_line)
                df = pd.read_csv(file_handler, index_col=None, on_bad_lines='skip')  # , usecols=cols)
                # here we combine df_start and df
                df_combined = pd.concat([df_start, df])
                if len(df):
                    if col == 'index':
                        df['index_col'] = df.index
                        df_combined['index_col'] = df_combined.index
                        realcol = 'index_col'
                    else:
                        realcol = col
                    plt.scatter(df.index + max_episode, df.r, marker='.', alpha=0.05, s=0.1,
                                label=os.path.basename(log_folder))
                    df_combined['yrolling'] = df_combined['r'].rolling(window=window).mean()  # .iloc[1000:]
                    plt.plot(df_combined[realcol] + max_episode, df_combined.yrolling,
                             label=os.path.basename(log_folder))

    plt.axvline(x=max_episode, color='k', linestyle='--')

    plt.rcParams["figure.figsize"] = (15, 5)
    plt.gcf().set_size_inches(15, 5)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning curve (smoothed window = ' + str(window) + ')')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'all_trains_curriculum' + 'Episode' + '.png'))
    plt.close()


def plot_results2(log_folder, policynames, modelnames, repetitions, env_name, title='Learning Curve', window=50):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    fig = plt.figure(title)
    for modelname in modelnames:
        for policyname in policynames:
            x, y = [], []
            for repetition in range(repetitions):
                dir2 = os.path.join(log_folder, str(modelname)[26:29], str(policyname)[0:5], str(repetition), env_name)
                results = load_results(dir2)
                x0, y0 = ts2xy(results, 'timesteps')
                x.append(list(x0))
                y.append(list(y0))

            flatx = [item for sub in x for item in sub]
            maxx = max(flatx)  # maxx is a numpy float64, convert to int:

            mean_x_axis = [i for i in range(int(maxx))]
            ys_interp = [np.interp(mean_x_axis, x[i], y[i]) for i in range(len(x))]
            mean_y_axis = np.mean(ys_interp, axis=0)

            # y = np.asarray(y).astype(float)
            mean_y_axis = moving_average(mean_y_axis, window=window * repetitions)
            # print('results', x, y)
            mean_x_axis = mean_x_axis[len(mean_x_axis) - len(mean_y_axis):]
            plt.plot(mean_x_axis, mean_y_axis, label=str(modelname)[26:29] + str(policyname)[0:5])

    plt.xlabel('Timesteps (window=' + str(window) + ')')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed, mean of " + str(repetitions))
    plt.legend()
    plt.show()
    plt.close()


def plot_transfer_matrix(matrix_data, row_names, col_names, output_file):
    fig, ax = plt.subplots()
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)

    plt.tick_params(axis='x', rotation=90)
    ax.imshow(matrix_data, cmap=plt.cm.Blues, aspect='auto', vmin=0, vmax=1)

    for i in range(len(row_names)):
        for j in range(len(col_names)):
            ax.text(j, i, str(round(matrix_data[i][j] * 100) / 100), va='center', ha='center')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_tsne(data, labels, index, color):
    print(index)
    # sorted_indices = sorted(indices, key=lambda i: int(labels[i].replace(train_name, '')))
    # sorted_data = data[sorted_indices]
    all_sizes = [int(x[:-1]) for x in labels[index[0]:index[1]]]
    max_size = max(all_sizes)
    all_sizes = [50 * x / max_size for x in all_sizes]
    plt.plot(data[index[0]:index[1], 0], data[index[0]:index[1], 1], marker=None, label=index[2], c=color)
    plt.scatter(data[index[0]:index[1], 0], data[index[0]:index[1], 1], marker='o', label=index[2], s=all_sizes,
                c=color)
    plt.close()

    # for i, name in enumerate(labels[index[0]:index[1]]):
    #    plt.annotate(name, (data[i + index[0], 0], data[i + index[0], 1]), textcoords="offset points", xytext=(-10, 5), ha='center')


def save_delta_figures(dir, df_summary, df_x):
    '''for var in ['dpred', 'dpred_correct', 'dpred_accurate']:
        df_list = []
        for key_val, sub_df in df_summary.items():
            for _, row in sub_df.iterrows():
                informedness = row['Informedness']
                mean, std = row[var].strip().split(' ')
                mean = float(mean)
                std = float(std.strip('()'))
                df_list.append([key_val, informedness, mean, std, row[var]])

        df = pd.DataFrame(df_list, columns=["key_val", "Informedness", "mean", "std"])
        pivot_df = df.pivot(index="key_val", columns="Informedness", values="mean")

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(pivot_df, vmin=0, vmax=1, annot=pivot_df, fmt='.2f', cmap='crest', linewidths=0.5, linecolor='white')
        plt.title(f"Heatmap of " + var)

        plt.tight_layout()

        os.makedirs(dir, exist_ok=True)
        plot_save_path = os.path.join(dir, var + '_heatmap.png')
        plt.savefig(plot_save_path)
        plt.close()'''

    # subfigs of accuracy across operators
    all_cols = ['t', 'f', 'tt', 'tf', 'ft', 'ff']
    p_cols = ['p' + col for col in all_cols]
    m_cols = ['m' + col for col in all_cols]
    new_cols = p_cols + m_cols

    # generate ideal row
    # df_x2 = df_x.copy()
    row_df = list(df_x.items())[0][1]
    print('row df', row_df)
    temp = {}
    for col_idx, (_, row) in enumerate(row_df.iterrows()):
        op = row['operator']
        op_dict = {}

        op_dict['pt'] = row['pt'] + row['pf']
        op_dict['pf'] = 0
        op_dict['mt'] = 0
        op_dict['mf'] = row['mt'] + row['mf']
        op_dict['ptt'] = row['ptt'] + row['pft'] + row['ptf']
        op_dict['ptf'] = 0
        op_dict['pft'] = 0
        op_dict['pff'] = 0
        op_dict['mtt'] = row['mtt'] + row['mft'] + row['mtf']
        op_dict['mtf'] = 0
        op_dict['mft'] = 0
        op_dict['mff'] = 0
        op_dict['operator'] = op

        temp[op] = op_dict
    temp = pd.DataFrame(temp).transpose()

    all_regimes = df_x.keys()
    if "direct" in all_regimes:
        all_regimes = ['noOpponent', 'direct', 'everything']
    df_x2 = [('ideal', temp)] + [(reg, df_x[reg]) for reg in all_regimes]

    colmap = {'t': 'Δb', 'f': '=b', 'tt': '1-1', 'tf': '1-0', 'ft': '0-1', 'ff': '0-0'}

    for columns, colors, pathname in zip([['t', 'f'], ['tt', 'tf', 'ft', 'ff']], [None, ['blue', 'lightgreen', 'lightblue', 'pink']], ['dpred', 'acc']):
        nrows = len(df_x2)
        ncols = len(df_x2[0][1]['operator'].unique())
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 1.2 * nrows))

        fig.text(0.02, 0.5, 'training set', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.text(0.5, 0.03, 'informedness operator', ha='center', va='center', fontsize=12)

        for idx, (key_val, df) in enumerate(df_x2):
            for col_idx, (_, row) in enumerate(df.iterrows()):  # row is each operator
                if nrows > 1:
                    ax = axes[idx, col_idx]
                else:
                    ax = axes[col_idx]
                data = {colmap[col]: [row[f'p{col}'], row[f'm{col}']] for col in columns}
                tmp_df = pd.DataFrame(data, index=['Δb*', '=b*'])
                if idx == 0:
                    ax.set_facecolor('0.9')

                tmp_df.plot(kind='bar', stacked=True, ax=ax, width=0.6, legend=False, color=colors)
                ax.set_ylim([0, 1])

                op = row['operator']
                if op == '1-0':
                    op = 'Op-NoOp'

                if idx == nrows - 1:
                    ax.set_xlabel(op)
                else:
                    ax.set_xticks([])
                if col_idx == 0:
                    if 'direct' in all_regimes:
                        this_name = {'ideal': 'ideal', 'noOpponent': 'no opponent', 'direct': 'opponent', 'everything': 'everything'}[key_val]
                    else:
                        this_name = key_val
                    ax.set_ylabel(this_name)
                else:
                    ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=4)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.12, wspace=0.06, top=0.92, left=0.09, bottom=0.18)
        plt.savefig(os.path.join(dir, f'small_multiples-{pathname}.png'))
        plt.close()

    # heatmap versions
    for var in ['dpred', 'dpred_correct', 'dpred_accurate']:
        df_list = []
        for key_val, sub_df in df_x.items():
            for _, row in sub_df.iterrows():
                informedness = row['operator']
                mean, std = row[var].strip().split(' ')
                mean = float(mean)
                std = float(std.strip('()'))
                df_list.append([key_val, informedness, mean, std, row[var]])

        df = pd.DataFrame(df_list, columns=["key_val", "operator", "mean", "std", "original_val"])
        pivot_df = df.pivot(index="key_val", columns="operator", values="mean")
        annot_df = df.pivot(index="key_val", columns="operator", values="original_val")

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(pivot_df, vmin=0, vmax=1, annot=annot_df, fmt='', cmap='RdBu', linewidths=0.5, linecolor='white')
        plt.title(f"Heatmap of " + var)

        plt.tight_layout()

        os.makedirs(dir, exist_ok=True)
        plot_save_path = os.path.join(dir, var + '_heatmap-ops.png')
        plt.savefig(plot_save_path)
        plt.close()


def save_double_param_figures(save_dir, top_pairs, avg_loss, last_epoch_df):
    this_save_dir = os.path.join(save_dir, 'doubleparams')
    os.makedirs(this_save_dir, exist_ok=True)
    for (param1, param2), _ in top_pairs:
        plt.figure(figsize=(10, 6))
        unique_values1 = last_epoch_df[param1].unique()
        unique_values2 = last_epoch_df[param2].unique()
        if len(unique_values2) * len(unique_values1) > 12:
            continue
        for value1 in unique_values1:
            for value2 in unique_values2:
                sub_df = avg_loss[(param1, param2)][
                    (avg_loss[(param1, param2)][param1] == value1) & (
                            avg_loss[(param1, param2)][param2] == value2)]

                str1 = f'{param1} = {value1}' if not isinstance(value1, str) or value1[0:3] != "N/A" or value1[
                                                                                                        0:3] != "na" else value1
                str2 = f'{param2} = {value2}' if not isinstance(value2, str) or value2[0:3] != "N/A" or value1[
                                                                                                        0:2] != "na" else value2
                plt.plot(sub_df['epoch'], sub_df['mean'],
                         label=f'{str1}, {str2}')
                plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)

        plt.title(f'Average accuracy vs Epoch for {param1} and {param2}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(os.path.join(os.getcwd(), os.path.join(this_save_dir, f'{param1}{param2}.png')))
        plt.close()

        # Creating the histogram
        plt.figure(figsize=(10, 6))
        hist_data = []
        labels = []
        for value1 in unique_values1:
            for value2 in unique_values2:
                value_df = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == value1)]
                mean_acc = value_df.groupby('param')['accuracy'].mean()
                mean_acc.index = mean_acc.index.astype('category')
                hist_data.append(mean_acc)
                labels.append(f'{param2} = {value2}, {param1} = {value1}')
        hist_data = [np.array(data) for data in hist_data]
        plt.hist(hist_data, bins=np.arange(0, 1.01, 0.05), stacked=True, label=labels, alpha=0.5)

        plt.title(f'Histogram of accuracy for {param2} and {param1}')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(os.getcwd(), os.path.join(this_save_dir, f'hist_{param1}{param2}.png')))
        plt.close()

from matplotlib.patches import Rectangle


def plot_dependency_bar_graphs_new(data, save_dir, strategies, novelty, layer='all-activations', retrain=False, strat=""):
    datasets = ['s1', 's2', 's3'] if not novelty else ['s1', 's2']
    architectures = ['mlp', 'cnn', 'clstm']
    architecture_colors = {'mlp': '#B22222', 'cnn': '#663399', 'clstm': '#4682B4'}

    fig, axs = plt.subplots(len(architectures), len(datasets),
                            figsize=(8 * len(datasets), 2.5 * len(architectures) + 0.8),
                            squeeze=False)
    fig.suptitle(f'Dependency Comparison Across Models and Datasets - {"Novel" if novelty else "Familiar"} Tasks',
                 fontsize=16)

    bar_width = 0.35
    feature_gap = 0.2
    strategy_gap = 0.25

    background_colors = ['#FFBBBB', '#FFFFBB', '#BBFFBB']

    for row, architecture in enumerate(architectures):
        for col, dataset in enumerate(datasets):
            ax = axs[row, col]

            # Add background colors for strategy sections
            strategy_widths = [len(features) * (bar_width + feature_gap) + 0.5*strategy_gap for features in strategies.values()]
            start = 0
            for color, width in zip(background_colors, strategy_widths):
                ax.add_patch(Rectangle((start - 0.25, 0), width, 1, fill=True, color=color, alpha=0.3, zorder=0))
                start += width

            x_offset = 0
            for strategy_idx, (strategy, features) in enumerate(strategies.items()):
                for feature in features:
                    subset = data[(data['model'].str.endswith(dataset)) &
                                  (data['model'].str.contains(architecture)) &
                                  (data['is_novel'] == novelty) &
                                  (data['activation'] == layer) &
                                  (data['feature'] == feature)]

                    if not subset.empty:
                        # Calculate portions for stacked bars
                        feature_acc = subset['feature_acc'].values[0]  # Total feature accuracy
                        pred_given_feat = subset['f_implies_p_mean'].values[0]  # Prediction accuracy when feature is correct
                        pred_given_not_feat = subset['notf_implies_p_mean'].values[0]  # Prediction accuracy when feature is incorrect

                        # Feature is correct bar (bottom)
                        ax.bar(x_offset, feature_acc, bar_width,
                               color=architecture_colors[architecture],
                               label='Feature Correct' if row == 0 and col == 0 else "")

                        # Prediction accuracy within feature correct
                        ax.bar(x_offset, feature_acc * pred_given_feat, bar_width*0.5,
                               color='white', alpha=1.0,
                               label='Prediction Correct' if row == 0 and col == 0 else "")

                        # Feature is incorrect bar (top)
                        ax.bar(x_offset, 1 - feature_acc, bar_width,
                               bottom=feature_acc,
                               color=architecture_colors[architecture], alpha=0.5,
                               label='Feature Incorrect' if row == 0 and col == 0 else "")

                        # Prediction accuracy within feature incorrect
                        ax.bar(x_offset, (1 - feature_acc) * pred_given_not_feat, bar_width*0.5,
                               bottom=feature_acc,
                               color='white', alpha=1.0,
                               label='Prediction Correct' if row == 0 and col == 0 else "")

                        ax.text(x_offset - 0.25, feature_acc, f'{feature_acc:.0%}',
                                ha='center', va='center', fontsize=8)
                        ax.text(x_offset + 0.25, feature_acc * pred_given_feat,
                                f'{pred_given_feat:.0%}', ha='center', va='center',
                                fontsize=8, color='darkblue')
                        ax.text(x_offset + 0.25, feature_acc + (1 - feature_acc) * pred_given_not_feat,
                                f'{pred_given_not_feat:.0%}', ha='center', va='center',
                                fontsize=8, color='darkblue')

                        # Feature labels
                        ax.text(x_offset, -0.05, feature, ha='center', va='top', fontsize=8)

                    x_offset += bar_width + feature_gap
                x_offset += strategy_gap

            ax.set_ylabel('Probability' if col == 0 else '')
            ax.set_ylim(-0.1, 1.1)  # Adjusted to make room for labels
            ax.set_xlim(-bar_width, x_offset - strategy_gap)
            ax.set_xticks([])
            #ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

            if col != 0:
                ax.yaxis.set_ticks([])

            if row == 0:
                ax.set_title(f'{dataset.upper()}', fontsize=14)
            if col == 0:
                ax.text(-0.1, 0.5, f'{architecture.upper()}', va='center', ha='right',
                        rotation=90, transform=ax.transAxes, fontsize=14)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0),
               ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{save_dir}/dependency_comparison_{layer}_{"novel" if novelty else "familiar"}_rt_{retrain}_{strat}.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_dependency_bar_graphs_flipped(data, save_dir, strategies, novelty, layer='all-activations', retrain=False, strat=""):
    datasets = ['s1', 's2', 's3'] if not novelty else ['s1', 's2']
    architectures = ['mlp', 'cnn', 'clstm']
    architecture_colors = {'mlp': '#B22222', 'cnn': '#663399', 'clstm': '#4682B4'}

    fig, axs = plt.subplots(len(architectures), len(datasets),
                           figsize=(8 * len(datasets), 2.5 * len(architectures) + 0.8),
                           squeeze=False)
    fig.suptitle(f'Prediction Implies Feature - {"Novel" if novelty else "Familiar"} Tasks',
                 fontsize=16)

    bar_width = 0.35
    feature_gap = 0.2
    strategy_gap = 0.25

    background_colors = ['#FFBBBB', '#FFFFBB', '#BBFFBB']

    for row, architecture in enumerate(architectures):
        for col, dataset in enumerate(datasets):
            ax = axs[row, col]

            # Add background colors for strategy sections
            strategy_widths = [len(features) * (bar_width + feature_gap) + 0.5*strategy_gap for features in strategies.values()]
            start = 0
            for color, width in zip(background_colors, strategy_widths):
                ax.add_patch(Rectangle((start - 0.25, 0), width, 1, fill=True, color=color, alpha=0.3, zorder=0))
                start += width

            x_offset = 0
            for strategy_idx, (strategy, features) in enumerate(strategies.items()):
                for feature in features:
                    subset = data[(data['model'].str.endswith(dataset)) &
                                (data['model'].str.contains(architecture)) &
                                (data['is_novel'] == novelty) &
                                (data['activation'] == layer) &
                                (data['feature'] == feature)]

                    if not subset.empty:
                        # For flipped analysis: prediction accuracy is now the primary metric
                        pred_acc = subset['pred_acc'].values[0]  # Total prediction accuracy
                        feat_given_pred = subset['p_implies_f_mean'].values[0]  # Feature accuracy when prediction is correct
                        feat_given_not_pred = subset['notp_implies_f_mean'].values[0]  # Feature accuracy when prediction is incorrect

                        # Prediction is correct bar (bottom)
                        ax.bar(x_offset, pred_acc, bar_width,
                              color=architecture_colors[architecture],
                              label='Prediction Correct' if row == 0 and col == 0 else "")

                        # Feature accuracy within prediction correct
                        ax.bar(x_offset, pred_acc * feat_given_pred, bar_width*0.5,
                              color='white', alpha=1.0,
                              label='Feature Correct' if row == 0 and col == 0 else "")

                        # Prediction is incorrect bar (top)
                        ax.bar(x_offset, 1 - pred_acc, bar_width,
                              bottom=pred_acc,
                              color=architecture_colors[architecture], alpha=0.5,
                              label='Prediction Incorrect' if row == 0 and col == 0 else "")

                        # Feature accuracy within prediction incorrect
                        ax.bar(x_offset, (1 - pred_acc) * feat_given_not_pred, bar_width*0.5,
                              bottom=pred_acc,
                              color='white', alpha=1.0,
                              label='Feature Correct' if row == 0 and col == 0 else "")

                        # Add percentage labels
                        ax.text(x_offset - 0.25, pred_acc, f'{pred_acc:.0%}',
                              ha='center', va='center', fontsize=8)
                        ax.text(x_offset + 0.25, pred_acc * feat_given_pred,
                              f'{feat_given_pred:.0%}', ha='center', va='center',
                              fontsize=8, color='darkblue')
                        ax.text(x_offset + 0.25, pred_acc + (1 - pred_acc) * feat_given_not_pred,
                              f'{feat_given_not_pred:.0%}', ha='center', va='center',
                              fontsize=8, color='darkblue')

                        # Feature labels
                        ax.text(x_offset, -0.05, feature, ha='center', va='top', fontsize=8)

                    x_offset += bar_width + feature_gap
                x_offset += strategy_gap

            ax.set_ylabel('Probability' if col == 0 else '')
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(-bar_width, x_offset - strategy_gap)
            ax.set_xticks([])

            if col != 0:
                ax.yaxis.set_ticks([])

            if row == 0:
                ax.set_title(f'{dataset.upper()}', fontsize=14)
            if col == 0:
                ax.text(-0.1, 0.5, f'{architecture.upper()}', va='center', ha='right',
                        rotation=90, transform=ax.transAxes, fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0),
              ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{save_dir}/p_implies_f_{layer}_{"novel" if novelty else "familiar"}_rt_{retrain}_{strat}.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_dependency_bar_graphs(data, save_dir, strategies, novelty, layer='all-activations', retrain=False, strat=""):
    datasets = ['s1', 's2', 's3'] if not novelty else ['s1', 's2']
    architectures = ['mlp', 'cnn', 'clstm']
    architecture_colors = {'mlp': '#B22222', 'cnn': '#663399', 'clstm': '#4682B4'}  # Slightly duller colors

    fig, axs = plt.subplots(len(architectures), len(datasets), figsize=(8 * len(datasets), 2.5 * len(architectures) + 0.8), squeeze=False)
    fig.suptitle(f'Dependency Comparison Across Models and Datasets - {"Novel" if novelty else "Familiar"} Tasks', fontsize=16)

    all_features = sorted(set([feature for features in strategies.values() for feature in features]))

    bar_width = 0.35
    feature_gap = 0.05
    strategy_gap = 0.25

    background_colors = ['#FFBBBB', '#FFFFBB', '#BBFFBB']  # Light red, yellow, green

    for row, architecture in enumerate(architectures):
        for col, dataset in enumerate(datasets):
            ax = axs[row, col]

            # Add background colors
            strategy_widths = [len(features) * (2 * bar_width + feature_gap) - feature_gap + strategy_gap for features in strategies.values()]
            start = 0
            for color, width in zip(background_colors, strategy_widths):
                ax.add_patch(Rectangle((start - 0.25, 0), width, 1, fill=True, color=color, alpha=0.3, zorder=0))
                start += width

            x_offset = 0
            for strategy_idx, (strategy, features) in enumerate(strategies.items()):
                for feature_idx, feature in enumerate(features):
                    subset = data[(data['model'].str.endswith(dataset)) &
                                  (data['model'].str.contains(architecture)) &
                                  (data['is_novel'] == novelty) &
                                  (data['activation'] == layer) &
                                  (data['feature'] == feature)]

                    if not subset.empty:
                        f_implies_p = subset['f_implies_p_mean'].values[0]
                        f_implies_p_std = subset['f_implies_p_std'].values[0]
                        notf_implies_notp = subset['notf_implies_notp_mean'].values[0]
                        notf_implies_notp_std = subset['notf_implies_notp_std'].values[0]

                        ax.bar(x_offset, f_implies_p, bar_width, color=architecture_colors[architecture], label='P(A|F)' if row == 0 and col == 0 and feature_idx == 0 else "")
                        ax.bar(x_offset + bar_width, notf_implies_notp, bar_width, color=architecture_colors[architecture], alpha=0.5, label='P(¬A|¬F)' if row == 0 and col == 0 and feature_idx == 0 else "")

                        # Add feature labels below the bars
                        ax.text(x_offset + bar_width / 2, -0.05, feature, ha='center', va='top', fontsize=8)

                    x_offset += 2 * bar_width + feature_gap

                x_offset += strategy_gap

            ax.set_ylabel('Probability' if col == 0 else '')
            ax.set_ylim(0, 1)
            ax.set_xlim(-bar_width, x_offset - strategy_gap)
            ax.set_xticks([])

            if col != 0:
                ax.yaxis.set_ticks([])

            if row == 0:
                ax.set_title(f'{dataset.upper()}', fontsize=14)
            if col == 0:
                ax.text(-0.1, 0.5, f'{architecture.upper()}', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=14)

    # Add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.savefig(f'{save_dir}/dependency_comparison_{layer}_{"novel" if novelty else "familiar"}_rt_{retrain}_{strat}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def create_faceted_heatmap(data, novelty, layer, output_file, strategies):
    datasets = ['s1', 's2', 's3'] if novelty == False else ['s1', 's2']
    architectures = ['mlp', 'cnn', 'clstm']
    features = data['feature'].unique()

    fig, axes = plt.subplots(len(datasets), len(architectures), figsize=(5 * len(architectures), 5 * len(datasets)))
    fig.suptitle(f"Prediction Dependency Heatmap - {'Novel' if novelty else 'Familiar'} Tasks", fontsize=16)
    ordered_features = [f for group in strategies.values() for f in group if f in features]


    for i, dataset in enumerate(datasets):
        for j, architecture in enumerate(architectures):
            ax = axes[i, j]

            subset = data[(data['model'].str.endswith(dataset)) &
                          (data['model'].str.contains(architecture)) &
                          (data['is_novel'] == novelty) &
                          (data['activation'] == layer)]

            heatmap_data = []
            annotations = []
            for feature in ordered_features:
                feature_data = subset[subset['feature'] == feature]
                if not feature_data.empty:
                    f_implies_p = feature_data['f_implies_p_mean'].values[0]
                    f_implies_p_std = feature_data['f_implies_p_std'].values[0]
                    notf_implies_notp = feature_data['notf_implies_notp_mean'].values[0]
                    notf_implies_notp_std = feature_data['notf_implies_notp_std'].values[0]
                    heatmap_data.append([f_implies_p, notf_implies_notp])
                    annotations.append([f"{f_implies_p:.3f} ({f_implies_p_std:.3f})", f"{notf_implies_notp:.3f} ({notf_implies_notp_std:.3f})"])
                else:
                    print(f"Warning: No data for {feature} in {dataset}-{architecture}")
                    heatmap_data.append([0, 0])
                    annotations.append(["0.000 (±0.000)", "0.000 (±0.000)"])

            sns.heatmap(heatmap_data, annot=np.array(annotations), fmt='', cmap='YlOrRd', cbar=False, ax=ax, annot_kws={'size': 12})

            if i == 0:
                ax.set_title(f"{architecture.upper()}")
            ax.set_yticklabels(ordered_features, rotation=0)
            ax.set_xticklabels(['P(A|F)', 'P(¬A|¬F)'])
            ax.tick_params(axis='both', which='both', length=0)

            if j == 0:
                ax.set_ylabel(dataset.upper(), rotation=0, ha='right', va='center', fontsize=12)
                ax.set_yticklabels(ordered_features, rotation=0)
            else:
                ax.set_yticklabels([])

            if i == len(datasets) - 1:
                ax.xaxis.set_ticks_position('bottom')
            else:
                ax.set_xticklabels([])


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_graphs_special(baselines_df, results_df, save_dir, strategies):
    for strategy, features in strategies.items():
        fig, ax = plt.subplots(figsize=(24, 8))

        x = []
        tick_labels = []
        group_labels = []
        width = 1/14
        print('results columns', results_df.columns)
        model_types = results_df["Model_Type"].unique()
        print('unique models', results_df['Model'].unique())

        colors = plt.cm.get_cmap('Dark2')(np.linspace(0, 1, 10))
        for novelty in ['Familiar', 'Novel']:
            datasets = ['s1', 's2', 's3'] if novelty == 'Familiar' else ['s1', 's2']

            for feature in features:
                group_start = len(x)
                baselines = baselines_df[(baselines_df['Feature'] == feature)]

                for dataset in datasets:
                    models = [model for model in results_df['Model'].unique() if model.endswith(dataset)]
                    tick_labels.append(f"{dataset}")
                    x.append(len(tick_labels)*1.1)

                    for k, model_type in enumerate(model_types):
                        baseline_models = baselines[baselines['Model_Type'] == model_type]
                        model_color = 'gray'

                        if feature != 'pred':
                            if len(baseline_models) > 0:
                                baseline_mean = baseline_models[f'{novelty} accuracy (input-activations)'].values[0]
                                baseline_q1 = baseline_models[f'{novelty} q1 (input-activations)'].values[0]
                                baseline_q3 = baseline_models[f'{novelty} q3 (input-activations)'].values[0]
                                bar_color = mcolors.to_rgba(model_color, alpha=1.0 - 0.15 * k)
                                ax.bar(x[-1] + width*k, baseline_mean, width*0.9,
                                       yerr=[[baseline_mean - baseline_q1], [baseline_q3 - baseline_mean]],
                                       capsize=5, label=f'baseline {model_type}' if len(x) == 4 else "", color=bar_color)

                    results = results_df[(results_df['Feature'] == feature)]

                    for i, model in enumerate(models, start=1):
                        results_use = results[results['Model'] == model]
                        model_color = colors[i % len(colors)]

                        for k, model_type in enumerate(model_types):
                            results_use_2 = results_use[results_use['Model_Type'] == model_type]
                            if len(results) > 0:
                                for j, activation_type in enumerate(['all', 'final-layer'], start=1):
                                    result_median = results_use_2[f'{novelty} accuracy ({activation_type}-activations)'].values[0]
                                    result_q1 = results_use_2[f'{novelty} q1 ({activation_type}-activations)'].values[0]
                                    result_q3 = results_use_2[f'{novelty} q3 ({activation_type}-activations)'].values[0]
                                    bar_color = mcolors.to_rgba(model_color, alpha=1.0 - 0.15*j - 0.3*k)
                                    ax.bar(x[-1] + (4 * i - 4 + j + 2*k + 1) * width, result_median, width*0.9,
                                           yerr=[[result_median - result_q1], [result_q3 - result_median]],
                                           capsize=5, label=f'{model[:-7]} {model_type} ({activation_type})' if len(x) == 4 else "", color=bar_color)

                group_center = (x[group_start] + x[-1]) / 2
                group_labels.append((group_center, f"{novelty}"))

        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xlim(min(x) - 0.08, max(x) + 1.0)
        ax.set_xticks([i + (len(datasets) * 2) * width / 2 for i in x])
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        for center, label in group_labels:
            ax.text(center, -0.1, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontweight='bold')

        plt.subplots_adjust(bottom=0.2, left=0.01, right=0.99, top=0.92)
        fig.suptitle(f'{strategy} Intrinsic Feature Representation', fontsize=16)
        plt.savefig(f'{save_dir}/{strategy.lower()}_spec_accuracy_comparison.png', bbox_inches='tight')
        plt.close(fig)

def get_sort_key(model):
    if 'mlp' in model.lower():
        return 1
    elif 'cnn' in model.lower():
        return 2
    elif 'lstm' in model.lower():
        return 3
    else:
        return 4


def plot_bar_graphs_new3(baselines_df, results_df, save_dir, strategies, one_dataset=True, one_model=False, layer='all', r_type='mlp1'):
    model_classes = sorted(set([model[:-7] for model in results_df['Model'].unique()]), key=get_sort_key)
    datasets = sorted(set([model[-2:] for model in results_df['Model'].unique()]))
    all_features = sorted(set([feature for features in strategies.values() for feature in features]))

    model_name_map = {
        'smlp': 'MLP Model',
        'cnn': 'CNN Model',
        'clstm': 'CLSTM Model'
    }

    model_colors = {
        'Raw Input': '#808080',
        'smlp': '#FF4444',
        'cnn': '#9944FF',
        'clstm': '#4444FF'
    }

    strategy_colors = {
        'No-Mindreading': '#FFE5E5',
        'Low-Mindreading': '#FFFAE5',
        'High-Mindreading': '#E5F0FF'
    }

    dataset_name_map = {
        's1': 'Stage 1 Training',
        's2': 'Stage 2 Training',
        's3': 'Stage 3 Training'
    }

    fig = plt.figure(figsize=(8 * len(datasets), 5.5 * (len(model_classes) + 1) + 2))

    gs = fig.add_gridspec(len(model_classes) + 2, len(datasets),
                          height_ratios=[4] * (len(model_classes) + 1) + [1],
                          hspace=0.45,
                          wspace=0.05)
    axs = [[fig.add_subplot(gs[i, j]) for j in range(len(datasets))] for i in range(len(model_classes) + 1)]

    bar_width = 0.15
    feature_gap = 0.03
    strategy_gap = 0.25 if 'spe' not in r_type else 0.15
    error_offset = bar_width * 0.25

    # Calculate total width for each strategy group
    strategy_widths = {}
    for strategy, features in strategies.items():
        total_bars = len(features)
        strategy_widths[strategy] = total_bars * bar_width + (total_bars - 1) * feature_gap + strategy_gap

    # Add strategy backgrounds
    for col in range(len(datasets)):
        ax = axs[0][col]
        x_offset = -bar_width
        for strategy, features in strategies.items():
            width = strategy_widths[strategy]
            rect = plt.Rectangle((x_offset, 0), width, 1,
                                 facecolor=strategy_colors[strategy],
                                 transform=ax.get_xaxis_transform(),
                                 zorder=-1,
                                 clip_on=False)
            ax.add_patch(rect)

            for row in range(1, len(model_classes) + 1):
                rect = plt.Rectangle((x_offset, 0), width, 1,
                                     facecolor=strategy_colors[strategy],
                                     transform=axs[row][col].get_xaxis_transform(),
                                     zorder=-1,
                                     clip_on=False)
                axs[row][col].add_patch(rect)
            x_offset += width

    # Plot baselines row (row 0)
    for col, dataset in enumerate(datasets):
        ax = axs[0][col]
        x_offset = 0

        for strategy_idx, (strategy, features) in enumerate(strategies.items()):
            for feature_idx, feature in enumerate(features):
                baseline = baselines_df[(baselines_df['Feature'] == feature) &
                                        (baselines_df['Model'] == dataset)]

                if len(baseline) > 0:
                    baseline_familiar = baseline['Familiar accuracy (input-activations)'].values[0]
                    baseline_novel = baseline['Novel accuracy (input-activations)'].values[0]
                    baseline_familiar_err = [baseline['Familiar q1 (input-activations)'].values[0],
                                             baseline['Familiar q3 (input-activations)'].values[0]]
                    baseline_novel_err = [baseline['Novel q1 (input-activations)'].values[0],
                                          baseline['Novel q3 (input-activations)'].values[0]]

                    # Draw bars without error bars
                    ax.bar(x_offset, baseline_familiar, bar_width,
                           color='white', edgecolor=model_colors['Raw Input'])
                    ax.bar(x_offset, baseline_novel, bar_width,
                           color=model_colors['Raw Input'])

                    # Add error bars separately with offset
                    ax.errorbar(x_offset + error_offset, baseline_familiar,
                                yerr=[[baseline_familiar - baseline_familiar_err[0]],
                                      [baseline_familiar_err[1] - baseline_familiar]],
                                color='black', capsize=3, fmt='none')
                    ax.errorbar(x_offset - error_offset, baseline_novel,
                                yerr=[[baseline_novel - baseline_novel_err[0]],
                                      [baseline_novel_err[1] - baseline_novel]],
                                color='black', capsize=3, fmt='none')

                    ax.text(x_offset, -0.05, feature, ha='right', va='top',
                            rotation=45, rotation_mode='anchor', fontsize=12)

                x_offset += bar_width + feature_gap
            x_offset += strategy_gap

        ax.set_ylabel('Accuracy' if col == 0 else '')
        ax.set_ylim(0, 1)
        ax.set_xlim(-bar_width, x_offset - strategy_gap)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if col == 0:
            ax.text(-0.25, 0.5, 'Raw Input', va='center', ha='right',
                    color=model_colors['Raw Input'], fontsize=24, fontweight='bold')

        if col != 0:
            ax.yaxis.set_ticks([])

        ax.set_title(dataset_name_map[dataset], fontsize=16, fontweight='bold')

    # Plot model results (rows 1 onwards)
    for row, model_class in enumerate(model_classes, start=1):
        for col, dataset in enumerate(datasets):
            ax = axs[row][col]
            x_offset = 0

            for strategy_idx, (strategy, features) in enumerate(strategies.items()):
                for feature_idx, feature in enumerate(features):
                    model_name = f"{model_class}-loc-{dataset}"
                    results = results_df[(results_df['Feature'] == feature) &
                                         (results_df['Model'] == model_name)]

                    if len(results) > 0:
                        result_familiar = results[f'Familiar accuracy ({layer}-activations)'].values[0]
                        result_novel = results[f'Novel accuracy ({layer}-activations)'].values[0]
                        result_familiar_err = [results[f'Familiar q1 ({layer}-activations)'].values[0],
                                               results[f'Familiar q3 ({layer}-activations)'].values[0]]
                        result_novel_err = [results[f'Novel q1 ({layer}-activations)'].values[0],
                                            results[f'Novel q3 ({layer}-activations)'].values[0]]

                        # Draw bars without error bars
                        ax.bar(x_offset, result_familiar, bar_width,
                               color='white', edgecolor=model_colors[model_class])
                        ax.bar(x_offset, result_novel, bar_width,
                               color=model_colors[model_class])

                        # Add error bars separately with offset
                        ax.errorbar(x_offset + error_offset, result_familiar,
                                    yerr=[[result_familiar - result_familiar_err[0]],
                                          [result_familiar_err[1] - result_familiar]],
                                    color=model_colors[model_class], capsize=3, fmt='none')
                        ax.errorbar(x_offset - error_offset, result_novel,
                                    yerr=[[result_novel - result_novel_err[0]],
                                          [result_novel_err[1] - result_novel]],
                                    color=model_colors[model_class], capsize=3, fmt='none')

                        ax.text(x_offset, -0.05, feature, ha='right', va='top',
                                rotation=45, rotation_mode='anchor', fontsize=12)

                    x_offset += bar_width + feature_gap
                x_offset += strategy_gap

            ax.set_ylabel('Accuracy' if col == 0 else '')
            ax.set_ylim(0, 1)
            ax.set_xlim(-bar_width, x_offset - strategy_gap)
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col != 0:
                ax.yaxis.set_ticks([])

            if col == 0:
                ax.text(-0.25, 0.5, model_name_map[model_class], va='center', ha='right',
                        color=model_colors[model_class], fontsize=24, fontweight='bold')

    # Add strategy labels at bottom
    for col in range(len(datasets)):
        ax = axs[-1][col]
        x_offset = -bar_width
        for strategy, features in strategies.items():
            width = strategy_widths[strategy]
            center = x_offset + width / 2
            strategy_text = strategy.replace('-', '\n')
            ax.text(center, -0.3, strategy_text, ha='center', va='top',
                    fontsize=16, fontweight='bold')
            x_offset += width
        ax.axis('off')

    plt.savefig(f'{save_dir}/{layer}_models_datasets_accuracy_comparison_{r_type}.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_bar_graphs_new(baselines_df, results_df, save_dir, strategies, one_dataset=True, one_model=False, layer='all', r_type='mlp1'):
    model_classes = sorted(set([model[:-7] for model in results_df['Model'].unique()]), key=get_sort_key)
    datasets = sorted(set([model[-2:] for model in results_df['Model'].unique()]))
    all_features = sorted(set([feature for features in strategies.values() for feature in features]))

    fig, axs = plt.subplots(len(model_classes), len(datasets), figsize=(8 * len(datasets), 4 * len(model_classes) + 1), squeeze=False)
    fig.suptitle('Accuracy Comparison Across Models and Datasets', fontsize=16)

    strategies2 = {
        'No-Mindreading': ['pred', 'opponents', 'big-loc', 'small-loc'],
        'Low-Mindreading': ['vision', 'fb-exist'],
        'High-Mindreading': ['fb-loc', 'b-loc', 'target-loc', 'labels']
    }
    all_features2 = sorted(set([feature for features in strategies2.values() for feature in features]))
    feature_colors = {feature: color for feature, color in zip(all_features2, plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(all_features2))))}

    bar_width = 0.15
    feature_gap = 0.03
    strategy_gap = 0.25
    if 'spe' in r_type:
        strategy_gap = 0.15

    for row, model_class in enumerate(model_classes):
        for col, dataset in enumerate(datasets):
            ax = axs[row, col]

            x_offset = 0
            for strategy_idx, (strategy, features) in enumerate(strategies.items()):
                for feature_idx, feature in enumerate(features):
                    baseline = baselines_df[(baselines_df['Feature'] == feature) & (baselines_df['Model'] == dataset)]
                    results = results_df[(results_df['Feature'] == feature) & (results_df['Model'] == f"{model_class}-loc-{dataset}")]

                    if len(baseline) > 0:
                        baseline_familiar = baseline['Familiar accuracy (input-activations)'].values[0]
                        baseline_novel = baseline['Novel accuracy (input-activations)'].values[0]
                        baseline_familiar_err = [baseline['Familiar q1 (input-activations)'].values[0], baseline['Familiar q3 (input-activations)'].values[0]]
                        baseline_novel_err = [baseline['Novel q1 (input-activations)'].values[0], baseline['Novel q3 (input-activations)'].values[0]]
                        if baseline_familiar >= baseline_novel:
                            ax.bar(x_offset, baseline_familiar, bar_width, yerr=[[baseline_familiar - baseline_familiar_err[0]], [baseline_familiar_err[1] - baseline_familiar]], capsize=3, color='white', edgecolor='gray', hatch='///')
                            ax.bar(x_offset, baseline_novel, bar_width, yerr=[[baseline_novel - baseline_novel_err[0]], [baseline_novel_err[1] - baseline_novel]], capsize=3, color='gray', )
                        else:
                            ax.bar(x_offset, baseline_novel, bar_width, yerr=[[baseline_novel - baseline_novel_err[0]], [baseline_novel_err[1] - baseline_novel]], capsize=3, color='gray', )
                            ax.bar(x_offset, baseline_familiar, bar_width, yerr=[[baseline_familiar - baseline_familiar_err[0]], [baseline_familiar_err[1] - baseline_familiar]], capsize=3,  color='white', edgecolor='gray', hatch='///')

                        x_offset += 1 * bar_width

                    if len(results) > 0:
                        result_familiar = results[f'Familiar accuracy ({layer}-activations)'].values[0]
                        result_novel = results[f'Novel accuracy ({layer}-activations)'].values[0]
                        result_familiar_err = [results[f'Familiar q1 ({layer}-activations)'].values[0], results[f'Familiar q3 ({layer}-activations)'].values[0]]
                        result_novel_err = [results[f'Novel q1 ({layer}-activations)'].values[0], results[f'Novel q3 ({layer}-activations)'].values[0]]

                        if result_familiar >= result_novel or np.isnan(result_novel):
                            ax.bar(x_offset, result_familiar, bar_width, yerr=[[result_familiar - result_familiar_err[0]], [result_familiar_err[1] - result_familiar]], capsize=3,  color='white', edgecolor=feature_colors[feature], hatch='///')
                            ax.bar(x_offset, result_novel, bar_width, yerr=[[result_novel - result_novel_err[0]], [result_novel_err[1] - result_novel]], label=feature if row == 0 and col == 0 else "", capsize=3, color=feature_colors[feature])
                        else:
                            ax.bar(x_offset, result_novel, bar_width, yerr=[[result_novel - result_novel_err[0]], [result_novel_err[1] - result_novel]], capsize=3, label=feature if row == 0 and col == 0 else "", color=feature_colors[feature])
                            ax.bar(x_offset, result_familiar, bar_width, yerr=[[result_familiar - result_familiar_err[0]], [result_familiar_err[1] - result_familiar]], capsize=3, color='white', alpha=0.5, edgecolor=feature_colors[feature], hatch='///')

                        x_offset += 1 * bar_width + feature_gap

                x_offset += strategy_gap

            ax.set_ylabel('Accuracy' if col == 0 else '')
            ax.set_ylim(0, 1)
            ax.set_xlim(-bar_width, x_offset - strategy_gap)
            ax.set_xticks([])

            if col != 0:
                ax.yaxis.set_ticks([])

            if row == 0:
                ax.set_title(f'{dataset.upper()}', fontsize=14)
            if col == 0:
                ax.text(-0.1, 0.5, f'{model_class}', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=14)

            if row == len(model_classes) - 1:
                strategy_centers = []
                current_center = 0
                for strategy, features in strategies.items():
                    width = len(features) * (2 * bar_width + feature_gap) - feature_gap
                    center = current_center + width / 2
                    strategy_centers.append(center)
                    current_center += width + strategy_gap
                ax.set_xticks(strategy_centers)
                ax.set_xticklabels(strategies.keys(), rotation=0, ha='center', fontsize=14)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(all_features), fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{save_dir}/{layer}_models_datasets_accuracy_comparison_{r_type}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_bar_graphs(baselines_df, results_df, save_dir, strategies, one_dataset=True, one_model=False):
    for novelty in ['Familiar', 'Novel']:
        fig, ax = plt.subplots(figsize=(24, 8))

        x = []
        xx = []
        tick_labels = []
        group_labels = []
        strat_labels = []
        width = 0.2
        if one_model:
            width += 0.2
        datasets = ['s1', 's2', 's3'] if novelty == 'Familiar' else ['s1', 's2']
        if one_dataset:
            datasets = ['s2']


        colors = plt.cm.get_cmap('Dark2')(np.linspace(0, 1, 10))

        for strategy, features in strategies.items():
            strat_start = len(xx)
            for feature in features:
                group_start = len(x)
                baseline = baselines_df[baselines_df['Feature'] == feature]

                for dataset in datasets:
                    models = [model for model in results_df['Model'].unique() if model.endswith(dataset)]

                    models = sorted(models,key=get_sort_key)
                    baseline_models = baseline[baseline['Model'] == dataset]

                    if len(baseline_models) > 0:
                        tick_labels.append(f"{dataset}")
                        x.append(len(tick_labels))
                        xx.append(len(tick_labels))

                        if feature != 'pred':
                            baseline_mean = baseline_models[f'{novelty} accuracy (input-activations)'].values[0]
                            baseline_q1 = baseline_models[f'{novelty} q1 (input-activations)'].values[0]
                            baseline_q3 = baseline_models[f'{novelty} q3 (input-activations)'].values[0]
                            ax.bar(x[-1] - width, baseline_mean, width*0.9,
                                   yerr=[[baseline_mean - baseline_q1], [baseline_q3 - baseline_mean]],
                                   capsize=5, label='baseline (perception)' if len(x) == 4 else "", color='gray')

                    if one_model:
                        models_real = [model for model in models if 'lstm' in model]
                    else:
                        models_real = models

                    models_real = sorted(models_real, key=get_sort_key)
                    for i, model in enumerate(models_real, start=1):
                        results = results_df[(results_df['Feature'] == feature) & (results_df['Model'] == model)]
                        model_color = colors[(i + 2 * (one_model)) % len(colors)]
                        if len(results) > 0:
                            for j, activation_type in enumerate(['all']): #all, final-layer
                                result_median = results[f'{novelty} accuracy ({activation_type}-activations)'].values[0]
                                result_q1 = results[f'{novelty} q1 ({activation_type}-activations)'].values[0]
                                result_q3 = results[f'{novelty} q3 ({activation_type}-activations)'].values[0]
                                bar_color = mcolors.to_rgba(model_color, alpha=0.7 if j == 1 else 1.0)
                                ax.bar(x[-1] + (1 * i - 1 + j) * width, result_median, width*0.9,
                                       yerr=[[result_median - result_q1], [result_q3 - result_median]],
                                       capsize=5, label=f'{model[:-7]} ({activation_type})' if len(x) == 1 else "",
                                       color=bar_color)

                group_center = (x[group_start] + x[-1] + 1) / 2
                group_labels.append((group_center, f"{feature}"))
            strat_center = (xx[strat_start] + xx[-1] + 2) / 2
            strat_labels.append((strat_center, f"{strategy}"))

        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xlim(min(x) - 0.2, max(x) + 0.7)
        ax.set_xticks([i + (len(models)*2)*width/2 for i in x])
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        for center, label in group_labels:
            ax.text(center, -0.05, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontweight='bold')
        for center, label in strat_labels:
            ax.text(center, -0.11, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontweight='bold')

        plt.subplots_adjust(bottom=0.2, left=0.01, right=0.99, top=0.92)
        fig.suptitle(f'{novelty} Accuracy Comparison', fontsize=16)
        title = f'{save_dir}/{novelty.lower()}_accuracy_comparison.png' if not one_dataset else f'{save_dir}/{novelty.lower()}_accuracy_comparison-s2.png'
        if one_model:
            title = f'{save_dir}/{novelty.lower()}_accuracy_comparison-clstm.png'
        plt.savefig(title, bbox_inches='tight')
        plt.close(fig)

def plot_strategy_bar(df, save_dir, strategies, retrain, small_mode=False):
    models = df['Model'].unique()

    vars = ['Familiar accuracy (All)', 'Novel accuracy (All)']
    if small_mode:
        vars = ['Familiar accuracy (input)', 'Novel accuracy (input)']

    for model in models:
        model_df = df[df['Model'] == model]

        fig, ax = plt.subplots(figsize=(15, 8))
        fig.suptitle(f'Accuracy by Feature and Strategy for {model}', fontsize=16)

        model_df_melted = pd.melt(model_df,
                                  id_vars=['Feature', 'strategy'],
                                  value_vars=vars,
                                  var_name='Accuracy Type',
                                  value_name='Accuracy')

        model_df_melted['Accuracy'] = model_df_melted['Accuracy'].astype(float)

        strategies_list = list(strategies.keys())
        x = np.arange(len(strategies_list))
        width = 0.3
        for i, acc_type in enumerate(vars):
            for j, strategy in enumerate(strategies_list):
                strategy_data = model_df_melted[(model_df_melted['strategy'] == strategy) &
                                                (model_df_melted['Accuracy Type'] == acc_type)]
                features = strategies[strategy]
                positions = x[j] + np.arange(len(features)) * width - width / 2 + i * width / 2

                ax.bar(positions, strategy_data['Accuracy'], width / 2, label=f'{acc_type} - {strategy}' if j == 0 else "_nolegend_")

        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies_list)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        for j, strategy in enumerate(strategies_list):
            features = strategies[strategy]
            positions = x[j] + np.arange(len(features)) * width - width / 2
            for pos, feature in zip(positions, features):
                ax.text(pos, -0.05, feature, ha='center', va='top', rotation=45, transform=ax.get_xaxis_transform())

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model}-strategy-bar-retrain{retrain}.png'), bbox_inches='tight')
        plt.close(fig)


def save_key_param_heatmap(save_dir, key_param_stats, key_param):
    this_save_dir = os.path.join(save_dir, 'key_param')
    os.makedirs(this_save_dir, exist_ok=True)
    print('kp', key_param)
    key_param = "i-informedness"
    print('saving key param heatmap')
    n_groups = len(list(key_param_stats.keys()))
    chars1 = ['T', 'F', 'N']
    chars2 = ['t', 'f', 'n']

    '''kp stats looks like this:
    save_dict[key_val]["informedness"] = {
                            'mean': means.to_dict(),
                            'std': stds.to_dict(),
                            'q1': Q1.to_dict(),
                            'q3': Q3.to_dict(),
                            'ci': standard_errors.to_dict(),
                        }'''
    print('new heatmap')
    if False:
        df_list = []
        for key, metrics_dict in key_param_stats.items():
            for metric, value in metrics_dict[key_param].items():
                df_list.append({"key": key, "metric": metric, "value": value})
        df = pd.DataFrame(df_list)

        sns.heatmap(df.pivot("key", "metric", "value"), annot=True, cmap="coolwarm")
        plt.show()

    # Initialize an empty 3x3 matrix
    if False:
        for param in list(next(iter(key_param_stats.values())).keys()):
            heatmap_data = np.zeros((3, 3))

            # Populate the heatmap data
            for i, char_x in enumerate(chars1):
                for j, char_y in enumerate(chars2):
                    key_val = char_x + char_y.lower()
                    if key_val in key_param_stats:
                        print(key_val, param)
                        print('ddd', param, key_val, key_param_stats[key_val][key_param]['mean'][param])
                        heatmap_data[i, j] = key_param_stats[key_val][key_param]['mean'][param]
            fig, ax = plt.subplots()
            cax = ax.imshow(heatmap_data, cmap='viridis', interpolation='nearest')

            # Set the ticks
            ax.set_xticks(np.arange(len(chars1)))
            ax.set_yticks(np.arange(len(chars2)))

            # Label them with the respective list entries
            ax.set_xticklabels(chars1)
            ax.set_yticklabels(chars2)

            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(len(chars1)):
                for j in range(len(chars2)):
                    text = ax.text(j, i, heatmap_data[i, j], ha="center", va="center", color="w")

            ax.set_title(f"Accuracy by inf")
            fig.tight_layout()
            file_path = os.path.join(os.getcwd(), this_save_dir, f'heatmap_inffy_{param}.png')
            plt.savefig(file_path)
            plt.close()


def df_list_from_stat_dict(stat_dict, param):
    df_list = []
    for key_val in stat_dict.keys():
        for param_val in stat_dict[key_val][param]['mean'].keys():
            mean = stat_dict[key_val][param]['mean'][param_val]
            ci = stat_dict[key_val][param]['ci'][param_val]
            df_list.append([key_val, param_val, f"{mean}", f"{ci}"])
    return df_list

def make_ifrscores(combined_df, combined_path, act, retrain, prior):
    print('prior thing', prior, combined_df['epoch'].unique(),)
    for use_acc in ['loss', 'val_acc']:
        mean_df = combined_df.groupby(['feature', 'model'])[use_acc].mean().reset_index()
        std_df = combined_df.groupby(['feature', 'model'])[use_acc].std().reset_index()

        merged_df = mean_df.merge(std_df, on=['feature', 'model'], suffixes=('_mean', '_std'))

        formatted_values = {
            (row['feature'], row['model']): f"{row[f'{use_acc}_mean']:.2f} ({row[f'{use_acc}_std']:.2f})"
            for _, row in merged_df.iterrows()
        }
        heatmap_colors = mean_df.pivot_table(index='feature', columns='model', values=use_acc, fill_value=0)

        heatmap_data = pd.DataFrame(
            {model: [formatted_values.get((feature, model), '0.00 (0.00)') for feature in heatmap_colors.index]
             for model in heatmap_colors.columns},
            index=heatmap_colors.index
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_colors, annot=heatmap_data, cmap='coolwarm' if use_acc=='loss' else 'coolwarm_r', vmin=0, vmax=0.25 if use_acc=='loss' else 1.0, fmt="")
        plt.xlabel('Model')
        plt.ylabel('Feature')
        plt.title(f'IFR {use_acc} Scores for {act}, retrain:{retrain}, prior:{prior}')
        plt.tight_layout()

        path2 = os.path.join(combined_path, 'ifr')
        os.makedirs(path2, exist_ok=True)
        plt.savefig(os.path.join(path2, f'{act}-{retrain}-{prior}-ifr_scores-{use_acc}.png'))
        plt.close()

def make_corr_things(merged_df, combined_path, loss_type='val_acc'):
    def calculate_correlations(group, mean_col, std_col, loss_type='val_acc'):
        valid_data = group[[mean_col, std_col, loss_type]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 1:
            try:
                combined_metric = valid_data[mean_col] / valid_data[std_col]
                corr, p_value = pearsonr(combined_metric, valid_data[loss_type])
                return corr, p_value
            except ValueError:
                return np.nan, np.nan
        return np.nan, np.nan

    def process_correlations(merged_df, metric_pairs, loss_type='val_acc'):
        correlations = []
        for (model, feature, act, retrain, prior), group in merged_df.groupby(['model', 'feature', 'act', 'retrain', 'prior']):
            for metric, (mean_col, std_col) in metric_pairs.items():
                if not (group[mean_col].isna().all() or group[std_col].isna().all()):
                    corr, p_value = calculate_correlations(group, mean_col, std_col, loss_type=loss_type)
                    correlations.append({
                        'train_set': model,
                        'feature': feature,
                        'act': act,
                        'metric': metric,
                        'correlation': corr,
                        'p_value': p_value
                    })

        for (feature, act), group in merged_df.groupby(['feature', 'act']):
            for metric, (mean_col, std_col) in metric_pairs.items():
                if not (group[mean_col].isna().all() or group[std_col].isna().all()):
                    corr, p_value = calculate_correlations(group, mean_col, std_col, loss_type=loss_type)
                    correlations.append({
                        'train_set': 'all_models',
                        'feature': feature,
                        'act': act,
                        'metric': metric,
                        'correlation': corr,
                        'p_value': p_value
                    })

        return pd.DataFrame(correlations)

    def make_correlation_heatmap(correlation_df, loss_type='val_acc'):
        for act in ['all_activations', 'final_layer_activations', 'input_activations']:
            try:
                cdf = correlation_df[(correlation_df['act'] == act)]
                cdf['correlation'] = pd.to_numeric(cdf['correlation'], errors='coerce')
                cdf['p_value'] = pd.to_numeric(cdf['p_value'], errors='coerce')

                fig, axes = plt.subplots(1, 3, figsize=(30, 8))
                fig.suptitle(f'IFR-{loss_type} Correlations (p-values) over using {act}', fontsize=16)

                for idx, metric in enumerate(metric_pairs):
                    pivot_corr = cdf[cdf['metric'] == metric].pivot(index='feature', columns='train_set', values='correlation')
                    pivot_pval = cdf[cdf['metric'] == metric].pivot(index='feature', columns='train_set', values='p_value')

                    def format_cell(corr, pval):
                        if pd.isnull(corr) or pd.isnull(pval):
                            return 'NaN\n(NaN)'
                        corr_str = f'{corr:.2f}'
                        pval_str = f'{pval:.3f}'
                        if pval < 0.05:
                            return f'**{corr_str}**\n({pval_str})'
                        return f'{corr_str}\n({pval_str})'

                    combined = pivot_corr.combine(pivot_pval, format_cell)

                    cbar = True if idx == len(metric_pairs) - 1 else False
                    mask = pivot_pval >= 0.05
                    sns.heatmap(pivot_corr, annot=combined, cmap='coolwarm' if loss_type=='loss' else 'coolwarm_r', vmin=-1, vmax=1, center=0, fmt="", mask=mask, ax=axes[idx], cbar=cbar)
                    axes[idx].set_title(f'{metric.replace("_", " ").title()}')
                    axes[idx].set_xlabel('Train Set')
                    if idx == 0:
                        axes[idx].set_ylabel('Feature')
                    else:
                        axes[idx].set_ylabel('')

                plt.tight_layout()
                path2 = os.path.join(combined_path, 'corrs')
                os.makedirs(path2, exist_ok=True)
                plt.savefig(os.path.join(path2, f'{act}-ifr_{loss_type}_combined.png'))
                plt.close()

            except BaseException as e:
                print(f'failed for {act}: {e}')

    # Define pairs of mean and std columns for each metric
    metric_pairs = {
        'acc': ('acc_mean', 'acc_std'),
        'novel_acc': ('novel_acc_mean', 'novel_acc_std'),
        'familiar_acc': ('familiar_acc_mean', 'familiar_acc_std')
    }

    for loss_type in ['val_acc', 'loss']:
        correlation_df = process_correlations(merged_df, metric_pairs, loss_type)
        make_correlation_heatmap(correlation_df, loss_type)

def make_splom_aux(data, act_type, dir):
    print('making splom')
    other_keys = data['other_key'].unique()
    reshaped_data = pd.DataFrame({
        key: data[data['other_key'] == key].set_index(['id', 'regime'])['aux_task_loss']
        for key in other_keys
    }).reset_index()
    print('reshaped')
    plot_vars = [col for col in reshaped_data.columns if col not in ['id', 'regime']]

    if reshaped_data.empty or reshaped_data.isnull().all().all():
        print(f"No valid data for SPLOM: {act_type}")
        return

    try:
        g = sns.PairGrid(reshaped_data, vars=plot_vars, hue='regime', diag_sharey=False, corner=True)
        g.map_lower(sns.scatterplot, alpha=0.5, s=0.5)

        def log_hist(x, **kwargs):
            if x.min() <= 0:
                min_val = x[x > 0].min()  # Find the minimum positive value
                x = x.clip(lower=min_val)  # Clip values to be at least min_val
            bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 30)
            plt.hist(x, bins=bins, **kwargs)
            plt.xscale('log')

        g.map_diag(log_hist)
        print('started')

        def corrfunc(x, y, **kws):
            r, _ = stats.pearsonr(x, y)
            ax = plt.gca()
            ax.annotate(f'r = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

        #g.map_upper(corrfunc)

        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            g.axes[i, j].set_xscale('log')
            g.axes[i, j].set_yscale('log')

        title = f'Loss Comparison SPLOM - {act_type.capitalize()} Activations'
        g.fig.suptitle(title)
        g.add_legend(title='Regime', bbox_to_anchor=(1.05, 1), loc='upper right')

        filename = f'splom_{act_type}'
        plt.savefig(os.path.join(dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SPLOM created for {act_type}")
    except Exception as e:
        print(f"Error creating SPLOM for {act_type}, Error: {str(e)}")
        print(f"Data shape: {reshaped_data.shape}, Data types: {reshaped_data.dtypes}")
        print(f"Data head:\n{reshaped_data.head()}")

def make_splom(merged_df, combined_path, act, retrain, prior):
    print('splom thing', prior, merged_df['epoch'].unique())
    for use_acc in ['loss', 'val_acc']:
        g = sns.FacetGrid(merged_df, col='feature', row='model', margin_titles=True, sharex=True, sharey=True, )
        g.map_dataframe(sns.scatterplot, x=use_acc, y='acc_mean', hue='epoch', palette='viridis')

        g.set_axis_labels(f'IFR {use_acc}', 'Model accuracy')
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.fig.suptitle(f'IFR {use_acc} vs Acc, {act}, retrain:{retrain}, prior:{prior}, Models and Features', y=1.02)

        plt.tight_layout()
        path2 = os.path.join(combined_path, 'sploms')
        os.makedirs(path2, exist_ok=True)
        plt.savefig(os.path.join(path2, f'{act}-{retrain}-{prior}-splom-{use_acc}.png'))
        plt.close()

def make_scatters2(all_indy_all, all_indy_final, other_keys, combined_path):
    for layer, layername in [(all_indy_all, 'all'), (all_indy_final, 'final')]:
        for i, key1 in enumerate(other_keys):
            for j, key2 in enumerate(other_keys):
                if j < i:
                    plt.figure(figsize=(10, 8))
                    data1 = layer[layer['other_key'] == key1][['id', 'aux_task_loss', 'regime']]
                    data2 = layer[layer['other_key'] == key2][['id', 'aux_task_loss', 'regime']]

                    merged_data = pd.merge(data1, data2, on=['id', 'regime'], suffixes=(f'_{key1}', f'_{key2}'))

                    sns.scatterplot(data=merged_data, x=f'aux_task_loss_{key1}', y=f'aux_task_loss_{key2}', hue='regime', alpha=0.5, s=5)

                    plt.xscale('log')
                    plt.yscale('log')
                    plt.title(f'{key2} loss versus {key1} loss ({layername})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(combined_path, 'scatters', f'{layername}-{key1}-{key2}.png'))
                    plt.close()

def make_scatter(merged_df, save_path, act):
    print('scatter thing', merged_df['epoch'].unique(), merged_df['model'].unique())

    for use_acc in ['val_acc']:#['loss', 'val_acc']:
        for feature in merged_df['feature'].unique():
            fig, ax = plt.subplots(figsize=(14, 10))

            this_df = merged_df[(merged_df['feature'] == feature)]

            models = this_df['model'].unique()
            colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))
            model_color_map = dict(zip(models, colors))

            for rep in this_df['rep'].unique():
                rep_df = this_df[this_df['rep'] == rep]
                rep_df = rep_df.sort_values('epoch')

                for model in models:
                    model_df = rep_df[rep_df['model'] == model]

                    ax.scatter(model_df[use_acc], model_df['acc_mean'], color=model_color_map[model], alpha=0.3, label=f"{model}" if rep == model_df['rep'].unique()[0] else "")

                    ax.plot(model_df[use_acc], model_df['acc_mean'], color=model_color_map[model], alpha=0.3, linestyle='--')
            for model in models:
                # aggregate over reps
                model_df = this_df[this_df['model'] == model]

                mean_df = model_df.groupby('epoch').agg('mean').reset_index()

                ax.errorbar(mean_df[use_acc], mean_df['acc_mean'],
                             yerr=mean_df['acc_std'],
                             fmt='o', capsize=5, capthick=2,
                             color=model_color_map[model],
                             label=f"{model}")
                ax.plot(mean_df[use_acc], mean_df['acc_mean'], color=model_color_map[model], alpha=0.9)
                ax.scatter(mean_df[use_acc], mean_df['familiar_acc_mean'], marker='^', color=model_color_map[model], alpha=0.9)
                ax.scatter(mean_df[use_acc], mean_df['novel_acc_mean'], marker='s', color=model_color_map[model], alpha=0.9)

            # this_df has model, our color, and rep/epoch, our line
            plt.xlabel(use_acc)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(by_label.values(), by_label.keys(),
                      title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylabel('model accuracy')
            plt.title(f'{act} - {feature} - Scatter plot ({use_acc})')
            plt.tight_layout()
            path2 = os.path.join(save_path, 'scatters')
            os.makedirs(path2, exist_ok=True)
            plt.savefig(os.path.join(path2, f'{act}-{feature}-scatter-{use_acc}.png'))
            plt.close()


def save_key_param_figures(save_dir, key_param_stats, oracle_stats, key_param, key_param_stats_special=[]):
    save_key_param_heatmap(save_dir, key_param_stats, key_param)
    this_save_dir = os.path.join(save_dir, 'key_param')
    os.makedirs(this_save_dir, exist_ok=True)
    print('saving key param figures', this_save_dir)

    n_groups = len(list(key_param_stats.keys()))
    print('key param keys', key_param_stats.keys())

    for param in list(next(iter(key_param_stats.values())).keys()):


        # print('trying param', param)
        labels = list(key_param_stats.keys())
        param_vals = list(key_param_stats[next(iter(key_param_stats))][param]['mean'].keys())
        bar_width = 0.8 / len(param_vals)
        index = np.arange(len(labels))

        fig, ax = plt.subplots()

        for idx, (param_val, color) in enumerate(zip(param_vals, plt.cm.tab10.colors)):
            means = [key_param_stats[key_val][param]['mean'][param_val] for key_val in labels]
            cis = [key_param_stats[key_val][param]['ci'][param_val] for key_val in labels]
            ax.bar(index + idx * bar_width, means, bar_width, label=param_val, yerr=cis, alpha=0.8, capsize=5,
                   color=color)

        ax.set_xlabel(key_param)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by {} and {}'.format(param, key_param))
        ax.set_xticks(index + bar_width * (n_groups - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        file_path = os.path.join(os.getcwd(), this_save_dir, f'key_{param}.png')
        plt.savefig(file_path)

        labels = list(key_param_stats[next(iter(key_param_stats))][param]['mean'].keys())
        key_param_vals = list(key_param_stats.keys())
        bar_width = 0.8 / len(key_param_vals)
        index = np.arange(len(labels))

        fig, ax = plt.subplots()

        # Create bars for each key_param within the group of param_val
        for idx, (key_val, color) in enumerate(zip(key_param_vals, plt.cm.tab10.colors)):
            means = [key_param_stats[key_val][param]['mean'][param_val] for param_val in labels]
            cis = [key_param_stats[key_val][param]['ci'][param_val] for param_val in labels]
            ax.bar(index + idx * bar_width, means, bar_width, label=key_val, yerr=cis, alpha=0.8, capsize=5,
                   color=color)

        # Format plot
        ax.set_xlabel(param)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by {} and {}'.format(key_param, param))
        ax.set_xticks(index + bar_width * (len(key_param_vals) - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        file_path = os.path.join(os.getcwd(), this_save_dir, f'reversed_key_{param}.png')
        plt.savefig(file_path)

        all_stds = []

        for stat_dict, label in zip([key_param_stats, oracle_stats], ['accuracy']):#['accuracy', 'o_acc']):

            if stat_dict is None:
                continue
            df_list = df_list_from_stat_dict(stat_dict, param)

            df = pd.DataFrame(df_list, columns=[key_param, param, "accuracy mean", "accuracy std"])
            # if the param shape has 2+ dimensions, use only the last timestep ones and aggregate

            table_save_path = os.path.join(this_save_dir, f'{param}_{label}_table.csv')
            df.to_csv(table_save_path, index=False)

            # produce typical heatmaps for accuracy
            big_mode = True
            split_by_type = True

            if param == 'test_regime':
                for type in ['loc', 'box', 'size'] if split_by_type else ['']:
                    for use_zero_op in [True, False]:
                        if use_zero_op:
                            desired_order = ['Tt0', 'Tf0', 'Tn0', 'Ft0', 'Ff0', 'Fn0', 'Nt0', 'Nf0', 'Nn0', 'Tt1', 'Tf1',
                                             'Tn1', 'Ft1', 'Ff1', 'Fn1', 'Nt1', 'Nf1', 'Nn1']
                        else:
                            desired_order = ['Tt1', 'Tf1', 'Tn1', 'Ft1', 'Ff1', 'Fn1', 'Nt1', 'Nf1', 'Nn1']
                        # todo: don't even make these if there's only 1 repetition
                        for use_std in [True, False]:
                            df["accuracy mean"] = pd.to_numeric(df["accuracy mean"], errors='coerce')
                            if use_std:
                                df["accuracy std"] = pd.to_numeric(df["accuracy std"], errors='coerce')
                                print('df!', df['accuracy std'], df['accuracy mean'])
                                df["Accuracy mean (Accuracy std)"] = df["accuracy mean"].map("{:.2f}".format) + " (" + df["accuracy std"].map("{:.2f}".format) + ")"
                            df_filtered = df[df[param].astype(str).str.endswith('1')] if not use_zero_op else df
                            if use_std:
                                all_stds.extend(df["accuracy std"].values.tolist())
                                pivot_df = df_filtered.pivot(index=key_param, columns=param, values="Accuracy mean (Accuracy std)")
                                mean_values_df = pivot_df.applymap(lambda x: float(x.split(' ')[0]))

                            else:
                                pivot_df = df_filtered.pivot(index=key_param, columns=param, values="accuracy mean")
                                mean_values_df = pivot_df
                                pivot_df = pivot_df.applymap(lambda x: f"{x:.2f}")

                            fig = plt.figure(figsize=(8, 13 - 6 * split_by_type))
                            heatmap_ax = fig.add_axes([0.0, 0.11, 1.0, 0.9])
                            cbar_ax = fig.add_axes([0.0, 0.03, 1.0, 0.02])

                            original_row_order = pivot_df.index.tolist()

                            use_special_rows = False

                            use_many_rows = False
                            if split_by_type:
                                filtered_row_order = [row for row in original_row_order if type in row]
                                pivot_df = pivot_df.loc[filtered_row_order]
                                mean_values_df = mean_values_df.loc[filtered_row_order]

                            if use_special_rows:
                                if use_many_rows:
                                    single_rows = ['Tt0', 'Tf0', 'Tn0', 'Ft0', 'Ff0', 'Fn0', 'Nt0', 'Nf0', 'Nn0', 'Tt1', 'Tf1', 'Tn1', 'Ft1', 'Ff1', 'Fn1', 'Nt1', 'Nf1', 'Nn1']
                                    contrast_rows = ['Tt', 'Tf', 'Tn', 'Ft', 'Ff', 'Fn', 'Nt', 'Nf', 'Nn']
                                    desired_row_order = ['noOpponent', 'direct', 'everything']
                                    desired_row_order.extend(single_rows)
                                    row_name_dict = {x: 'contrast-' + x for x in contrast_rows}
                                    desired_row_order.extend(contrast_rows)
                                    row_name_dict.update({x: 'single-' + x.replace('0', '-a').replace('1', '-p') for x in single_rows})
                                    desired_row_order.extend(['homogeneous'])
                                    row_name_dict.update({'noOpponent': 'full-absent', 'direct': 'full-present', 'everything': 'full-both', 'homogeneous': 'homogeneous'})
                                else:
                                    if 'Tt' in original_row_order:
                                        desired_row_order = ['Tt', 'Tf', 'Tn', 'Ft', 'Ff', 'Fn', 'Nt', 'Nf', 'Nn']
                                        row_name_dict = {x: 'contrast-' + x for x in desired_row_order}

                                    elif 'Tt0' in original_row_order:
                                        desired_row_order = desired_order
                                        row_name_dict = {x: 'single-' + x for x in desired_row_order}
                                    else:
                                        desired_row_order = ['noOpponent', 'direct', 'everything']
                                        row_name_dict = {'noOpponent': 'full-noop', 'direct': 'full-op', 'everything': 'full-both'}

                                pivot_df = pivot_df.reindex(columns=desired_order, index=desired_row_order)
                                mean_values_df = mean_values_df.reindex(columns=desired_order, index=desired_row_order)

                                new_column_names = {x: x.replace('1', '-p').replace('0', '-a') for x in pivot_df.columns}

                                pivot_df = pivot_df.rename(columns=new_column_names)
                                mean_values_df = mean_values_df.rename(columns=new_column_names)

                                pivot_df = pivot_df.rename(index=row_name_dict)
                                mean_values_df = mean_values_df.rename(index=row_name_dict)

                            # row_minima = mean_values_df.min(axis=1)
                            # mean_values_df['min'] = row_minima
                            # pivot_df['min'] = row_minima.map("{:.2f}".format)
                            print('mean values df', mean_values_df)

                            quadmesh = sns.heatmap(mean_values_df, annot=pivot_df, fmt='', cmap='RdBu', linewidths=0.5, linecolor='white', vmin=0, vmax=1, cbar=False, ax=heatmap_ax)
                            quadmesh.set_yticklabels(quadmesh.get_yticklabels(), rotation=0)
                            quadmesh.set_xlabel("Test regime", fontsize=10)
                            quadmesh.set_ylabel("Training dataset", fontsize=10)

                            if use_special_rows:
                                # thicker white lines manually added
                                quadmesh.hlines(3, *quadmesh.get_xlim(), color='white', linewidth=4)
                                quadmesh.hlines(3 + 9, *quadmesh.get_xlim(), color='white', linewidth=4)
                                quadmesh.hlines(3 + 18, *quadmesh.get_xlim(), color='white', linewidth=4)
                                quadmesh.vlines(9, *quadmesh.get_ylim(), color='white', linewidth=4)

                            # if not split_by_type:
                            # attempt to not draw a colorbar resulted in a bad one
                            plt.colorbar(quadmesh.collections[0], cax=cbar_ax, orientation='horizontal', cmap='RdBu')
                            plt.tight_layout()

                            plot_save_path = os.path.join(save_dir, f'{label}_{param}_{use_std}_{use_zero_op}_{type}heatmap.png')
                            print('saving fig to', plot_save_path)
                            plt.savefig(plot_save_path, bbox_inches='tight')
                            plt.close()

            if param == 'test_regime':
                regimes = df[key_param].unique()
                custom_col_order = ['t', 'f', 'n']
                custom_row_order = ['T', 'F', 'N']
                if "Nt" in regimes:
                    regimes = ['Tt', 'Tf', 'Tn', 'Ft', 'Ff', 'Fn', 'Nt', 'Nf', 'Nn']
                elif "Nt0" in regimes:
                    regimes = ['Tt0', 'Tf0', 'Tn0', 'Ft0', 'Ff0', 'Fn0', 'Nt0', 'Nf0', 'Nn0', 'Tt1', 'Tf1', 'Tn1', 'Ft1', 'Ff1', 'Fn1', 'Nt1', 'Nf1', 'Nn1']
                    print('calculating asymmetry')
                    asymmetry_dict = {}
                    for regime_A in regimes:
                        for regime_B in regimes:
                            if regime_A != regime_B:
                                accuracy_A_B = df[(df[key_param] == regime_A) & (df['test_regime'] == regime_B)]['accuracy mean'].values[0]
                                accuracy_B_A = df[(df[key_param] == regime_B) & (df['test_regime'] == regime_A)]['accuracy mean'].values[0]
                                asymmetry = accuracy_A_B - accuracy_B_A  # train a test b minus train b test a, so if positive, it's better to train a test b.
                                asymmetry_dict[(regime_A, regime_B)] = asymmetry
                    print(asymmetry_dict)
                    sorted_asymmetries = sorted(asymmetry_dict.keys(), key=lambda x: (regimes.index(x[0]), regimes.index(x[1])))
                    for key in sorted_asymmetries:
                        print(f"{key[0]}, {key[1]}: {asymmetry_dict[key]}")
                elif "lo_Nt" in regimes:
                    regimes = ['lo_' + x for x in ['Tt', 'Tf', 'Tn', 'Ft', 'Ff', 'Fn', 'Nt', 'Nf', 'Nn']]
                elif "direct" in regimes:
                    regimes = ['noOpponent', 'direct', 'everything']

                ncols = min(3, len(regimes))
                nrows = math.ceil(len(regimes) / ncols)
                for do_both_opponent_types in [True, False]:
                    all_filters = ['0', '1'] if do_both_opponent_types else ['1']
                    if big_mode:
                        fig = plt.figure(figsize=(10, 2.3 * nrows))
                    else:
                        fig = plt.figure(figsize=(15, 2 * nrows))
                    if big_mode:
                        gs = gridspec.GridSpec(nrows, ncols, wspace=0.2, hspace=0.3, top=1.0 - (0.25 / nrows), bottom=0.12, left=0.03, right=1.0)
                    else:
                        gs = gridspec.GridSpec(nrows, ncols, wspace=0.2, hspace=0.4, top=0.75, left=0.1, right=0.9)  # might mess up if ncols > 1
                    for k, key_param_val in enumerate(regimes):
                        if nrows == 1 and ncols > 1:
                            ax_main = plt.subplot(gs[k])
                        elif ncols == 1 and nrows > 1:
                            ax_main = plt.subplot(gs[k // ncols])
                        else:
                            ax_main = plt.subplot(gs[k // ncols, k % ncols])
                        if "direct" in regimes:
                            regime_name = {'noOpponent': 'no opponent', 'direct': 'opponent', 'everything': 'everything'}[key_param_val]
                        else:
                            regime_name = key_param_val
                        inner_gs = gridspec.GridSpecFromSubplotSpec(1, len(all_filters), subplot_spec=ax_main, wspace=0.2, hspace=0.6, height_ratios=[0.8])
                        for ending_idx, filter in enumerate(all_filters):
                            ax = plt.subplot(inner_gs[ending_idx])
                            df_filtered = df[df[param].str.endswith(filter)]
                            subset_df = df_filtered[(df_filtered[key_param] == key_param_val)]
                            subset_df['param_char1'] = subset_df[param].str[0]
                            subset_df['param_char2'] = subset_df[param].str[1]
                            subset_pivot = subset_df.pivot(index='param_char1', columns='param_char2', values='accuracy mean')

                            #subset_pivot = subset_pivot / 5
                            subset_pivot = subset_pivot.reindex(custom_row_order)
                            subset_pivot = subset_pivot.reindex(columns=custom_col_order)

                            sns.heatmap(subset_pivot, annot=True, fmt='.2f', cmap='RdBu', linewidths=0.5, linecolor='white', vmin=0, vmax=1, ax=ax, cbar=False)

                            if len(all_filters) < 2:
                                ax.set_xlabel("")
                            elif ((k // ncols) == (len(regimes) // ncols)) or nrows < 2:
                                ax.set_xlabel(f"No Opponent" if ending_idx == 0 else "Opponent", labelpad=2)
                            else:
                                ax.set_xlabel("")
                            ax.set_ylabel("")
                            ax.xaxis.tick_top()
                            if (big_mode and (k // ncols != 0)):
                                ax.set_xticks([])
                            if ending_idx == 1 or (big_mode and (k % ncols != 0)):
                                ax.set_yticks([])

                        center_x = (ax_main.get_position().x0 + ax_main.get_position().x1) / 2
                        if not big_mode:
                            top_y = ax_main.get_position().y1 + (0.16 / ncols)
                        else:
                            if k // ncols == 0:
                                top_y = ax_main.get_position().y1 + 0.14 / ncols
                            else:
                                top_y = ax_main.get_position().y1 + 0.03 / ncols

                        fig.text(center_x, top_y, f"Training: {regime_name}",
                                 ha='center',
                                 va='bottom',
                                 fontsize=12,
                                 fontweight='bold')
                    plot_save_path = os.path.join(save_dir, f'{label}_{param}_{do_both_opponent_types}_grids_small_multiples.png')
                    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.03)
                    plt.savefig(plot_save_path)
                    plt.close()

        # get stats about stds
        all_stds_array = np.array(all_stds)
        filtered_stds = all_stds_array[~np.isnan(all_stds_array)]
        print('len', len(all_stds), len(filtered_stds), 'mean std', np.mean(filtered_stds), 'std_of_std', np.std(filtered_stds))


def save_single_param_figures(save_dir, params, avg_loss, last_epoch_df):
    this_save_dir = os.path.join(save_dir, 'singleparams')
    os.makedirs(this_save_dir, exist_ok=True)
    for param in params:
        plt.figure(figsize=(10, 6))
        unique_values = last_epoch_df[param].unique()
        if len(unique_values) > 12:
            continue
        for value in unique_values:
            print('avg loss', avg_loss)
            sub_df = avg_loss[param][avg_loss[param][param] == value]

            plt.plot(sub_df['epoch'], sub_df['mean'],
                     label=f'{param} = {value}' if not isinstance(value, str) or value[0:3] != "N/A" or value[
                                                                                                        0:2] != "na" else value)
            plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
        plt.title(f'Average accuracy vs Epoch for {param}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        file_path = os.path.join(os.getcwd(), this_save_dir, f'{param}.png')
        plt.savefig(file_path)
        plt.close()

        # Creating the histogram
        plt.figure(figsize=(10, 6))
        hist_data = []
        labels = []
        for value in unique_values:
            value_df = last_epoch_df[last_epoch_df[param] == value]
            mean_acc = value_df.groupby('param')['accuracy'].mean()
            mean_acc.index = mean_acc.index.astype('category')
            hist_data.append(mean_acc.values)
            # labels.append(f'{param} = {value}')
            hist_data.extend(mean_acc.values)
            labels.extend([f'{param} = {value}'] * len(mean_acc.values))

        hist_data = [np.array(data) for data in hist_data]

        plt.hist(np.array(hist_data), bins=np.arange(0, 1.01, 0.05), stacked=True, label=np.array(labels))

        plt.title(f'Histogram of accuracy for last epoch for {param}')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.legend(loc='upper left')
        file_path = os.path.join(os.getcwd(), this_save_dir, f'hist_{param}.png')
        plt.savefig(file_path)
        plt.close()


def load_checkpoints(directory, prefix):
    checkpoints = {}
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.pt'):
            epoch = int(filename.split('-')[-1].split('.')[0])
            checkpoint_path = os.path.join(directory, filename)
            checkpoints[(prefix, epoch)] = torch.load(checkpoint_path)
    return checkpoints


def extract_weights(checkpoints):
    weights = []
    labels = []
    sources = []
    for (prefix, epoch), checkpoint in checkpoints.items():
        # Checkpoint structure: [model.kwargs, model.state_dict()]
        state_dict = checkpoint[1] if isinstance(checkpoint, list) and len(checkpoint) > 1 else None

        if state_dict is None:
            raise ValueError(f"Unexpected checkpoint structure: {type(checkpoint)}")

        flat_weights = np.concatenate([param.cpu().numpy().flatten() for param in state_dict.values()])

        weights.append(flat_weights)
        labels.append(f"{prefix}-{epoch}")
        sources.append(prefix)
    return np.array(weights), labels, sources


def parse_filename(filename):
    parts = filename.split('\\')
    experiment = parts[-2].split('-')[-1]
    if experiment == 'retrain':
        experiment = parts[-2].split('-')[-2]
    repetition = int(re.search(r'losses-(\d+)', parts[-1]).group(1))
    is_retrain = 'retrain' in filename
    return experiment, repetition, is_retrain

def read_and_combine_data(original_file, retrain_file=None):
    df = pd.read_csv(original_file)
    if retrain_file:
        df_retrain = pd.read_csv(retrain_file)
        df_retrain['Batch'] += 10000  # Adjust batch numbers for retrain data
        df = pd.concat([df, df_retrain], ignore_index=True)
    return df

def plot_learning_curves(save_dir, lp_list):
    print('doing tsne stuff')
    print('XXXXXXXXXXXXXXX')

    print(lp_list)

    experiments = {}

    for group in lp_list:
        for filename in group:
            experiment, repetition, is_retrain = parse_filename(filename)
            if not os.path.exists(filename):
                continue
            if experiment not in experiments:
                experiments[experiment] = {}
            if repetition not in experiments[experiment]:
                experiments[experiment][repetition] = {'original': None, 'retrain': None}

            if is_retrain:
                retrain_dir = os.path.dirname(filename)
                retrain_file = os.path.join(retrain_dir, f'adam-losses-{repetition}.csv')
                experiments[experiment][repetition]['retrain'] = retrain_file
            else:
                experiments[experiment][repetition]['original'] = filename

    plt.figure(figsize=(14, 6))
    colors = plt.cm.rainbow(np.linspace(0.1, 0.8, len(experiments)))


    for (experiment, repetitions), color in zip(experiments.items(), colors):
        original_data = []
        retrain_data = []
        for rep, rep_data in repetitions.items():
            if rep_data['original']:
                df_original = pd.read_csv(rep_data['original'])
                original_data.append(df_original)
            if rep_data['retrain']:
                df_retrain = pd.read_csv(rep_data['retrain'])
                retrain_data.append(df_retrain)

        if original_data:
            for k, df in enumerate(original_data):
                plt.plot(df['Batch'], df['Accuracy'], color=color, alpha=0.33*(k+1), label=f'{experiment}' if k == 2 else None)

        if retrain_data:
            for k, df in enumerate(retrain_data):
                plt.plot(df['Batch'] + 10000, df['Accuracy'], color=color, alpha=0.33*(k+1))

    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.savefig(os.path.join(save_dir, 'learncurves.png'))
    plt.close()


    '''
    for l in lp_list:
        if os.path.exists(l[0]):
            head, tail = os.path.split(l[0])
            # head is sup\exp\model

            print(head, tail)
            checkpoints1 = load_checkpoints(head, "checkpoint")
            checkpoints2 = load_checkpoints(head + '-retrain', "checkpoint")
            #checkpoints3 = load_checkpoints(f'supervised\\exp_101-L\\{os.path.basename(head)}', "0-checkpoint")
            #checkpoints4 = load_checkpoints(f'supervised\\exp_101-L\\{os.path.basename(head)}', "momentum09-rt-model")
            checkpoints = {**checkpoints1, **checkpoints2}
            #checkpointsb = {**checkpoints3, **checkpoints4}
            print(len(checkpoints))

            weights, labels, sources = extract_weights(checkpoints)
            #weights2, labels2, sources2 = extract_weights(checkpointsb)  # the direct training ones for comparison

            #all_weights = np.concatenate([weights, weights2])
            #all_labels = labels + labels2
            #all_sources = sources + sources2

            umap_model = umap.UMAP(n_neighbors=5, min_dist=0.5, metric='euclidean')
            #embedding2 = umap_model.fit_transform(weights2)
            embedding = umap_model.transform(weights)

            order1 = np.argsort([int(label.split('-')[-1]) for label in labels])
            sorted_embedding = embedding[order1]
            #order2 = np.argsort([int(label.split('-')[-1]) for label in labels2])
            #sorted_embedding2 = embedding2[order2]

            plt.figure(figsize=(12, 8))
            colors = {'0-checkpoint': 'blue', 'momentum09-rt-model': 'red'}
            names = {'0-checkpoint': 'pretraining', 'momentum09-rt-model': "transfer"}

            seen_labels = set()

            def add_to_legend(source, name):
                if name not in seen_labels:
                    seen_labels.add(name)
                    return name
                return "_nolegend_"

            for i, (label, source) in enumerate(zip(labels, sources)):
                plt.scatter(embedding[i, 0], embedding[i, 1], color=colors[source], s=50, label=add_to_legend(source, names[source]))
                plt.text(embedding[i, 0], embedding[i, 1], label.split('-')[-1], fontsize=8)
                #if i < len(labels) - 1:
                #    plt.plot([embedding[i, 0], embedding[i + 1, 0]], [embedding[i, 1], embedding[i + 1, 1]], color=colors[source], linestyle='-', linewidth=0.5)

            #for i, (label, source) in enumerate(zip(labels2, sources2)):
            #    plt.scatter(embedding2[i, 0], embedding2[i, 1], color=colors[source], s=50, marker='x', label=add_to_legend(source, names[source]))
            #    plt.text(embedding2[i, 0], embedding2[i, 1], label.split('-')[-1], fontsize=8)

                #if i < len(labels2) - 1:
                #    plt.plot([embedding2[i, 0], embedding2[i + 1, 0]], [embedding2[i, 1], embedding2[i + 1, 1]], color=colors[source], linestyle='-', linewidth=0.5)

            #for i in range(len(labels) - 1):
            #    plt.plot([sorted_embedding[i, 0], sorted_embedding[i + 1, 0]], [sorted_embedding[i, 1], sorted_embedding[i + 1, 1]], color='black', linestyle='-', linewidth=0.5)

            #for i in range(len(labels2) - 1):
            #    plt.plot([sorted_embedding2[i, 0], sorted_embedding2[i + 1, 0]], [sorted_embedding2[i, 1], sorted_embedding2[i + 1, 1]], color='black', linestyle='-', linewidth=0.5)

            plt.title(f'UMAP of Model Weights ({os.path.basename(head)})')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{tail}-umap.png'))'''

    print('plotting learning curves')
    for type in ['loc', 'box', 'size']:
        plt.figure(figsize=(10, 6))
        breaking = False
        for l in lp_list:
            if os.path.exists(l[0]):
                head, tail = os.path.split(l[0])
                name = os.path.basename(head)
                if type in name:
                    df = pd.read_csv(l[0])
                    plt.plot(df['Batch'], df['Accuracy'], label=name)
            #else:
            #    breaking = True
            #    break
        #if breaking:
        #    continue
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1.0])
        plt.title('Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'loss-{type}2.png'))
        plt.close()

    print('plotting rt learning curves')
    grouped_paths = {}
    for l in lp_list:
        for l2 in l[1:]:
            head, tail = os.path.split(l2)
            if head not in grouped_paths:
                grouped_paths[head] = []
            grouped_paths[head].append(l2)

    for head, paths in grouped_paths.items():
        plt.figure(figsize=(10, 6))
        print('paths', paths)
        for path in paths:
            if os.path.exists(path):
                name = os.path.basename(path)
                if 'rt' in name:
                    print('found a path', name)
                    df = pd.read_csv(path)
                    plt.plot(df['Batch'], df['Loss'], label=name[:-len('-rt-losses.csv')])
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.ylim([0, 0.6])
        plt.title(f'Validation Loss for {os.path.basename(head)}')
        plt.legend()
        plt.savefig(os.path.join(head, f'loss-{os.path.basename(head)}.png'))
        plt.close()


def save_fixed_double_param_figures(save_dir, top_n_ranges, df, avg_loss, last_epoch_df):
    this_save_dir = os.path.join(save_dir, 'fixeddoubleparams')
    os.makedirs(this_save_dir, exist_ok=True)
    for combo in top_n_ranges:
        param1, value1, param2 = combo
        subset = df[df[param1] == value1]
        plt.figure(figsize=(10, 6))
        unique_values = subset[param2].unique()
        if len(unique_values) > 12:
            continue
        for value2 in unique_values:
            sub_df = avg_loss[(param1, param2)][
                (avg_loss[(param1, param2)][param2] == value2) & (avg_loss[(param1, param2)][param1] == value1)]
            plt.plot(sub_df['epoch'], sub_df['mean'],
                     label=f'{param2} = {value2}' if not isinstance(value2, str) or value2[0:3] != "N/A" or value2[
                                                                                                            0:2] != "na" else value2)
            plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
        plt.title(f'Average accuracy vs Epoch for {param2} given {param1} = {value1}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        name = f'{param1}{str(value1)[:3]}{param2}'.replace('/', '-')
        plt.savefig(os.path.join(os.getcwd(), os.path.join(this_save_dir, f'{name}.png')))
        plt.close()

        # Creating the histogram
        plt.figure(figsize=(10, 6))
        hist_data = []
        labels = []
        for value2 in unique_values:
            value_df = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == value1)]
            mean_acc = value_df.groupby('param')['accuracy'].mean()
            mean_acc.index = mean_acc.index.astype('category')
            hist_data.append(mean_acc)
            labels.append(f'{param2} = {value2}')
        hist_data = [np.array(data) for data in hist_data]
        plt.hist(hist_data, bins=np.arange(0, 1.01, 0.05), stacked=True, label=labels, alpha=0.5)

        plt.title(f'Histogram of accuracy for {param2} given {param1} = {value1}')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(os.getcwd(), os.path.join(this_save_dir, f'hist_{name}.png')))
        plt.close()


def save_fixed_triple_param_figures(save_dir, top_n_ranges, df, avg_loss, last_epoch_df):
    this_save_dir = os.path.join(save_dir, 'fixedtripleparams')
    os.makedirs(this_save_dir, exist_ok=True)
    for combo in top_n_ranges:
        param1, value1, param2, value2, param3 = combo
        subset = df[(df[param1] == value1) & (df[param2] == value2)]
        plt.figure(figsize=(10, 6))
        unique_values = subset[param3].unique()
        if len(unique_values) > 12:
            continue
        for value3 in unique_values:
            sub_df = avg_loss[(param1, param2, param3)][(avg_loss[(param1, param2, param3)][param1] == value1) &
                                                        (avg_loss[(param1, param2, param3)][param2] == value2) &
                                                        (avg_loss[(param1, param2, param3)][param3] == value3)]
            plt.plot(sub_df['epoch'], sub_df['mean'],
                     label=f'{param3} = {value3}' if not isinstance(value3, str) or value3[0:3] != "N/A" else value3)
            plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
        plt.title(f'Average accuracy vs Epoch for {param3} given {param1}={value1} and {param2}={value2}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        name = f'{param1}{str(value1)[:3]}{param2}{str(value2)[:3]}{param3}'.replace('/', '-')
        plt.savefig(os.path.join(os.getcwd(), os.path.join(save_dir, f'{name}.png')))
        plt.close()

        # Creating the histogram
        plt.figure(figsize=(10, 6))
        hist_data = []
        labels = []
        for value3 in unique_values:
            value_df = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == value1) & (
                    last_epoch_df[param3] == value3)]
            mean_acc = value_df.groupby('param')['accuracy'].mean()
            mean_acc.index = mean_acc.index.astype('category')
            hist_data.append(mean_acc)
            labels.append(f'{param3} = {value3}')
        hist_data = [np.array(data) for data in hist_data]
        plt.hist(hist_data, bins=np.arange(0, 1.01, 0.05), stacked=True, label=labels, alpha=0.5)

        plt.title(f'Histogram of accuracy for {param3} given {param1} = {value1} and {param2} = {value2}')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(os.getcwd(), os.path.join(save_dir, f'hist_{name}.png')))
        plt.close()


def plot_losses(data_name, label, train_losses, val_losses, val_set_names, specific_name=None):
    plt.figure()
    plt.plot(train_losses, label=data_name + ' train loss')
    for val_set_name, val_loss in zip(val_set_names, val_losses):
        if specific_name == None or val_set_name == specific_name:
            plt.plot(val_loss, label=val_set_name + ' val loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f'supervised/{data_name}-{label}-losses.png')


def create_combined_histogram(df, combined_avg, param, folder):
    plt.figure(figsize=(10, 6))
    for value in combined_avg[param].unique():
        value_df = combined_avg[combined_avg[param] == value]
        mean_acc_per_epoch = value_df.groupby('epoch')['accuracy'].mean()

        plt.plot(mean_acc_per_epoch.index, mean_acc_per_epoch.values,
                 label=f'{param} = {value}' if not isinstance(value, str) or value[0:3] != "N/A" or value[
                                                                                                    0:2] != "na" else value)
        # plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
    plt.title(f'Average accuracy vs Epoch for {param}')
    plt.xlabel('Epoch')
    plt.ylabel('Average accuracy')
    plt.legend()
    plt.ylim(0, 1)
    file_path = os.path.join(os.getcwd(), folder, f'{param}.png')
    plt.savefig(file_path)
    plt.close()

    # Creating the histogram
    plt.figure(figsize=(10, 6))
    hist_data = []
    labels = []
    for value in df[param].unique():
        value_df = df[df[param] == value]
        mean_acc = value_df.groupby('param')['accuracy'].mean()
        mean_acc.index = mean_acc.index.astype('category')
        hist_data.append(mean_acc)
        labels.append(f'{param} = {value}')

    # hist_data = np.asarray(hist_data, dtype=object)
    plt.hist(hist_data, bins=np.arange(0, 1.01, 0.05), stacked=True, label=labels, alpha=0.5)

    plt.title(f'Histogram of accuracy for last epoch for {param}')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.legend(loc='upper left')
    file_path = os.path.join(os.getcwd(), folder, f'hist_{param}.png')
    plt.savefig(file_path)
    plt.close()
