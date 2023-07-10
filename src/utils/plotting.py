import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import pandas as pd
import json
import seaborn as sns
from stable_baselines3.common.monitor import get_monitor_files


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
                    #df['index_col'] = df.index + max_episode  # shift the episode numbers
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
                    plt.scatter(df.index  + max_episode, df.r, marker='.', alpha=0.05, s=0.1, label=os.path.basename(log_folder))
                    df_combined['yrolling'] = df_combined['r'].rolling(window=window).mean()#.iloc[1000:]
                    plt.plot(df_combined[realcol] + max_episode, df_combined.yrolling, label=os.path.basename(log_folder))

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


def save_double_param_figures(top_pairs, df, avg_loss, last_epoch_df):
    os.makedirs('supervised/doubleparams', exist_ok=True)
    for (param1, param2), _ in top_pairs:
        plt.figure(figsize=(10, 6))
        for value1 in df[param1].unique():
            for value2 in df[param2].unique():
                sub_df = avg_loss[(param1, param2)][
                    (avg_loss[(param1, param2)][param1] == value1) & (
                                avg_loss[(param1, param2)][param2] == value2)]

                str1 = f'{param1} = {value1}' if not isinstance(value1, str) or value1[0:3] != "N/A" else value1
                str2 = f'{param2} = {value2}' if not isinstance(value2, str) or value2[0:3] != "N/A" else value2
                plt.plot(sub_df['epoch'], sub_df['mean'],
                         label=f'{str1}, {str2}')
                plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)


        plt.title(f'Average accuracy vs Epoch for {param1} and {param2}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(os.path.join(os.getcwd(), f'supervised/doubleparams/{param1}{param2}.png'))
        plt.close()

        # Creating the histogram
        plt.figure(figsize=(10, 6))
        hist_data = []
        labels = []
        for value1 in df[param1].unique():
            for value2 in df[param2].unique():
                value_df = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == value1)]
                mean_acc = value_df.groupby('param')['accuracy'].mean()
                hist_data.append(mean_acc.values)
                labels.append(f'{param2} = {value2}, {param1} = {value1}')
        plt.hist(hist_data, bins=np.arange(0, 1, 0.05), stacked=True, label=labels, alpha=0.5)

        plt.title(f'Histogram of accuracy for {param2} and {param1}')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(os.getcwd(), f'supervised/fixeddoubleparams/hist_{param1}{param2}.png'))
        plt.close()



def save_single_param_figures(params, df, avg_loss, last_epoch_df):
    os.makedirs('supervised/singleparams', exist_ok=True)
    for param in params:
        plt.figure(figsize=(10, 6))
        for value in df[param].unique():
            sub_df = avg_loss[param][avg_loss[param][param] == value]

            plt.plot(sub_df['epoch'], sub_df['mean'], label=f'{param} = {value}' if not isinstance(value, str) or value[0:3] != "N/A" else value)
            plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
        plt.title(f'Average accuracy vs Epoch for {param}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        file_path = os.path.join(os.getcwd(), 'supervised', 'singleparams', f'{param}.png')
        plt.savefig(file_path)
        plt.close()

        # Creating the histogram
        plt.figure(figsize=(10, 6))
        hist_data = []
        labels = []
        for value in df[param].unique():
            value_df = last_epoch_df[last_epoch_df[param] == value]
            mean_acc = value_df.groupby('param')['accuracy'].mean()
            hist_data.append(mean_acc.values)
            labels.append(f'{param} = {value}')
        plt.hist(hist_data, bins=np.arange(0, 1, 0.05), stacked=True, label=labels, alpha=0.5)

        plt.title(f'Histogram of accuracy for last epoch for {param}')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(os.getcwd(), f'supervised/singleparams/hist_{param}.png'))
        plt.close()


def save_fixed_double_param_figures(top_n_ranges, df, avg_loss, last_epoch_df):
    os.makedirs('supervised/fixeddoubleparams', exist_ok=True)
    for combo in top_n_ranges:
        param1, value1, param2 = combo
        subset = df[df[param1] == value1]
        plt.figure(figsize=(10, 6))
        for value2 in subset[param2].unique():
            sub_df = avg_loss[(param1, param2)][(avg_loss[(param1, param2)][param2] == value2) & (avg_loss[(param1, param2)][param1] == value1)]
            plt.plot(sub_df['epoch'], sub_df['mean'],
                     label=f'{param2} = {value2}' if not isinstance(value2, str) or value2[0:3] != "N/A" else value2)
            plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
        plt.title(f'Average accuracy vs Epoch for {param2} given {param1} = {value1}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        name = f'{param1}{str(value1)[:3]}{param2}'.replace('/', '-')
        plt.savefig(os.path.join(os.getcwd(), f'supervised/fixeddoubleparams/{name}.png'))
        plt.close()

        # Creating the histogram
        plt.figure(figsize=(10, 6))
        hist_data = []
        labels = []
        for value2 in subset[param2].unique():
            value_df = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == value1)]
            mean_acc = value_df.groupby('param')['accuracy'].mean()
            hist_data.append(mean_acc.values)
            labels.append(f'{param2} = {value2}')
        plt.hist(hist_data, bins=np.arange(0, 1, 0.05), stacked=True, label=labels, alpha=0.5)

        plt.title(f'Histogram of accuracy for {param2} given {param1} = {value1}')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(os.getcwd(), f'supervised/fixeddoubleparams/hist_{name}.png'))
        plt.close()


def save_fixed_triple_param_figures(top_n_ranges, df, avg_loss, last_epoch_df):
    os.makedirs('supervised/fixedtripleparams', exist_ok=True)
    for combo in top_n_ranges:
        param1, value1, param2, value2, param3 = combo
        subset = df[(df[param1] == value1) & (df[param2] == value2)]
        plt.figure(figsize=(10, 6))
        for value3 in subset[param3].unique():
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
        plt.savefig(os.path.join(os.getcwd(), f'supervised/fixedtripleparams/{name}.png'))
        plt.close()

        # Creating the histogram
        plt.figure(figsize=(10, 6))
        hist_data = []
        labels = []
        for value3 in subset[param3].unique():
            value_df = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == value1) & (last_epoch_df[param3] == value3)]
            mean_acc = value_df.groupby('param')['accuracy'].mean()
            hist_data.append(mean_acc.values)
            labels.append(f'{param3} = {value3}')
        plt.hist(hist_data, bins=np.arange(0, 1, 0.05), stacked=True, label=labels, alpha=0.5)

        plt.title(f'Histogram of accuracy for {param3} given {param1} = {value1} and {param2} = {value2}')
        plt.xlabel('Accuracy')
        plt.ylabel('Count')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(os.getcwd(), f'supervised/fixeddoubleparams/hist_{name}.png'))
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
