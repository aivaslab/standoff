import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import pandas as pd
import json

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
        bar_width = max(df[indexer]) / (2 * len(values) + 1)
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
    print(monitor_files)

    for file_name in monitor_files:
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
                plt.figure()

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
            maxx = max(flatx) # maxx is a numpy float64, convert to int:

            mean_x_axis = [i for i in range(int(maxx))]
            ys_interp = [np.interp(mean_x_axis, x[i], y[i]) for i in range(len(x))]
            mean_y_axis = np.mean(ys_interp, axis=0)

            #y = np.asarray(y).astype(float)
            mean_y_axis = moving_average(mean_y_axis, window=window*repetitions)
            #print('results', x, y)
            mean_x_axis = mean_x_axis[len(mean_x_axis) - len(mean_y_axis):]
            plt.plot(mean_x_axis, mean_y_axis, label= str(modelname)[26:29]+str(policyname)[0:5])
            
    plt.xlabel('Timesteps (window=' + str(window) + ')')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed, mean of " + str(repetitions))
    plt.legend()
    plt.show()



