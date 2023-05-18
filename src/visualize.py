import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn import decomposition
import umap
import matplotlib.pylab as pl

from src.utils.display import process_csv, get_transfer_matrix_row
from src.utils.plotting import plot_merged, plot_selection, plot_split, plot_train, plot_transfer_matrix, plot_tsne, \
    plot_train_many, plot_train_curriculum

from matplotlib import pyplot as plt


def get_matrix_data(paths, only_last=False, metric='accuracy_mean', prefix='rand_rand', ordering=None,
                    ordering_old=None):
    train_paths_list = [[os.path.join(f.path) for f in os.scandir(path) if f.is_dir()] for path in paths]
    matrix_data = []
    for train_paths, dir_name in zip(train_paths_list, paths):
        for train_path in train_paths:
            train_path = os.path.join(train_path, 'evaluations')
            if os.path.exists(os.path.join(train_path, prefix + '_data.csv')):
                matrix_rows, timesteps, train_name = get_transfer_matrix_row(
                    os.path.join(train_path, prefix + '_data.csv'), metric, ordering, ordering_old, only_last=only_last)

                for index, timestep in enumerate(timesteps):
                    row_data = {
                        "this_dir": dir_name,
                        "train_name": train_name,
                        "timestep": timestep,
                        "values": matrix_rows.iloc[index].tolist()
                    }
                    print('train_name', train_name, "values", matrix_rows.iloc[index].tolist())
                    matrix_data.append(row_data)
    return matrix_data


def find_series_indices(labels, matrix_data):
    series_indices = []
    start = 0

    for i in range(1, len(labels)):
        current_train_name = labels[i][:labels[i].rfind(str(matrix_data[i]["timestep"]))]
        previous_train_name = labels[start][:labels[start].rfind(str(matrix_data[start]["timestep"]))]

        if current_train_name != previous_train_name:
            end = i - 1
            series_indices.append((start, end, previous_train_name))
            start = i

    final_train_name = labels[start][:labels[start].rfind(str(matrix_data[start]["timestep"]))]
    series_indices.append((start, len(labels) - 1, final_train_name))
    return series_indices


def make_transfer_matrix_new(save_path, prefix, make_matrix=False, make_tsne=False, metric='accuracy_mean'):
    eval_ordering = ['swapped', 'misinformed', 'partiallyUninformed', 'replaced',
                     'removedUninformed2', 'informedControl', 'removedInformed2', 'moved',]

    if make_matrix:
        matrix_data = get_matrix_data(
            [save_path],
            only_last=True,
            metric=metric,
            ordering=eval_ordering,
            prefix=prefix)
        matrix_data_sorted = sorted(matrix_data, key=lambda x: eval_ordering.index(x["train_name"]) if x[
            "train_name"] in eval_ordering else float(
            'inf'))
        lines = [x["values"] for x in matrix_data_sorted]
        row_names = [x["train_name"] for x in matrix_data_sorted]
        print(lines)
        plot_transfer_matrix(matrix_data=lines,
                             row_names=row_names,
                             col_names=eval_ordering,
                             output_file=os.path.join(save_path, metric + 'matrix.png'))

    if make_tsne:
        matrix_data = get_matrix_data([save_path], only_last=False, metric=metric, prefix='rand_rand',
                                      ordering=eval_ordering,
                                      ordering_old=None)

        lines = np.vstack([np.array(x["values"]) for x in matrix_data])
        labels = [x["train_name"] + str(x["timestep"]) for x in matrix_data]

        timesteps = [str(x["timestep"] // 1000) + 'k' for x in matrix_data]

        indices = find_series_indices(labels, matrix_data)

        for reducer, name in [(umap.UMAP(), 'umap'), (decomposition.PCA(n_components=2), 'pca'),
                              (TSNE(n_components=2, random_state=22, perplexity=min(15, lines.shape[0] - 1)), 'tsne')]:
            if name == 'pca':
                reducer.fit(lines)
                tsne_results = reducer.transform(lines)
            else:
                tsne_results = reducer.fit_transform(lines)

            output_file = os.path.join(save_path, metric + '_' + name + '_plot.png')

            plt.figure()

            colors = pl.cm.jet(np.linspace(0, 1, len(indices) + 1))
            for color, index in zip(colors, indices):
                plot_tsne(tsne_results, timesteps, index, color)

            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(name + ' plot of model performance')
            plt.legend([x[2] for x in indices], bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()


def make_eval_figures(path, figures_path, window=1, prefix=''):
    """
    plot the results found in one csv file.
    """

    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
    figures_path_combined = os.path.join(figures_path, 'combined')
    if not os.path.exists(figures_path_combined):
        os.mkdir(figures_path_combined)

    # load csv, process data
    grouped_df, grouped_df_small, grouped_df_noname_abs, grouped_df_noname = process_csv(path, prefix == 'gtr')

    title = prefix + '-' + path[path.find('S3-') + 3: path[path.find('S3-') + 3:].find('/')]

    plotted = ['Plotted']
    not_plotted = ['Could not plot']

    for rows in [grouped_df, grouped_df_small, grouped_df_noname, grouped_df_noname_abs]:
        rows["normed_r_mean"] = (rows["r_mean"] - rows["r_mean"].min()) / (
                rows["r_mean"].max() - rows["r_mean"].min())
        rows["normed_r_std"] = rows["r_std"] / (
                rows["r_mean"].max() - rows["r_mean"].min())

    if len(grouped_df_small):
        title2 = title + 'individual'
        # print(len(grouped_df_small), grouped_df_small.columns)
        plotted += ['grouped_df_small ' + title2]

        for uname in grouped_df_small.configName.unique():

            rows_big = grouped_df[grouped_df['configName'] == uname]
            print(uname, len(rows_big))
            rows_small = grouped_df_small[grouped_df_small['configName'] == uname]

            extended_path = os.path.join(figures_path, uname)
            if not os.path.exists(extended_path):
                os.mkdir(extended_path)

            # normalize certain values after all the filtering

            new_rows_big = rows_big.copy()
            new_rows_small = rows_small.copy()

            for val in ['r', 'episode_timesteps']:
                for rows, new_rows in [(rows_big, new_rows_big), (rows_small, new_rows_small)]:
                    new_rows[f"normed_{val}_mean"] = \
                        (rows[f"{val}_mean"] - rows[f"{val}_mean"].min()) \
                        / (rows[f"{val}_mean"].max() - rows[f"{val}_mean"].min())
                    new_rows[f"normed_{val}_std"] = \
                        (rows[f"{val}_std"]) / \
                        (rows[f"{val}_mean"].max() - rows[f"{val}_mean"].min())

            rows_big = new_rows_big
            rows_small = new_rows_small

            plot_merged(indexer='model_ep', df=rows_small, mypath=extended_path, title=title + 'validAccuracy',
                        window=window,
                        values=['weakAccuracy', 'accuracy'],
                        labels=['selected any treat', 'selected correct treat'])
            plot_merged(indexer='model_ep', df=rows_big, mypath=extended_path, title=title + 'absoluteAccuracy',
                        window=window,
                        values=['weakAccuracy', 'accuracy', 'valid', 'selectedSame'],
                        labels=['selected any treat', 'selected correct treat',
                                'selected any box', 'selected same as opponent'])
            plot_merged(indexer='model_ep', df=rows_small, mypath=extended_path, title=title + 'validLocation',
                        window=window,
                        values=['sel0', 'sel1', 'sel2', 'sel3', 'sel4'],
                        labels=['box 1', 'box 2', 'box 3', 'box 4', 'box 5'])
            plot_merged(indexer='model_ep', df=rows_big, mypath=extended_path, title=title + 'absoluteLocation',
                        window=window,
                        values=['sel0', 'sel1', 'sel2', 'sel3', 'sel4', 'valid'],
                        labels=['box 1', 'box 2', 'box 3', 'box 4', 'box 5', 'selected any box'])
            plot_merged(indexer='model_ep', df=rows_big, mypath=extended_path, title=title + 'absoluteLocationBars',
                        window=window, stacked_bar=True,
                        values=['sel0', 'sel1', 'sel2', 'sel3', 'sel4'],
                        labels=['box 1', 'box 2', 'box 3', 'box 4', 'box 5'])
            plot_selection(indexer='model_ep', df=rows_small, mypath=extended_path, title=title + 'selection',
                           window=window)
            plot_selection(indexer='model_ep', df=rows_small, mypath=extended_path, title=title + 'selectionBars',
                           window=window, bars=True)

            plot_merged(indexer='model_ep', df=rows_big, mypath=extended_path, title=title + 'rewardTimesteps',
                        window=window,
                        values=['r', 'episode_timesteps'],
                        labels=['reward', 'timesteps'])
            plot_merged(indexer='model_ep', df=rows_small, mypath=extended_path, title=title + 'avoid', window=window,
                        values=["avoidCorrect"], labels=["avoided correct box"])
    else:
        not_plotted += ['grouped_df ' + title2]

    if len(grouped_df_noname_abs):
        title2 = title + '-big'
        plot_merged(indexer='model_ep', df=grouped_df_noname_abs, mypath=figures_path_combined,
                    title=title2 + 'absoluteAccuracy', window=window,
                    values=['weakAccuracy', 'accuracy', 'normed_r', 'valid', 'selectedSame'],
                    labels=['selected any treat', 'selected correct treat',
                            'reward (normalized)', 'selected any box', 'selected same as opponent'])
    if len(grouped_df_small):
        title2 = title + '-small'
        # this graph is the same as above, but only taking into account valid samples
        # avoidcorrect should be identical to weakaccuracy when no opponent is present

        for value in ['accuracy', 'weakAccuracy', 'normed_r', 'avoidCorrect']:
            plot_split(indexer='model_ep', df=grouped_df_small, mypath=figures_path_combined, title=title2,
                       window=window,
                       values=[value])
        plotted += ['grouped_df_small ' + title2]
    else:
        not_plotted += ['grouped_df_small ' + title2]

    if len(grouped_df_noname):
        title2 = title + '-mixed'
        plot_merged(indexer='model_ep', df=grouped_df_noname, mypath=figures_path_combined, title=title2 + 'ac',
                    window=window, values=["avoidCorrect"], labels=["avoided correct box"])
        plot_selection(indexer='model_ep', df=grouped_df_noname, mypath=figures_path_combined, title=title2,
                       window=window)
        plot_merged(indexer='model_ep', df=grouped_df_noname, mypath=figures_path_combined,
                    title=title2 + 'validAccuracy',
                    window=window,
                    values=['weakAccuracy', 'accuracy', 'normed_r', 'selectedSame'],
                    labels=['selected any treat', 'selected correct treat',
                            'reward (normalized)', 'selected same as opponent'])
        plot_merged(indexer='model_ep', df=grouped_df_noname, mypath=figures_path_combined,
                    title=title + 'absoluteLocation',
                    window=window, values=['sel0', 'sel1', 'sel2', 'sel3', 'sel4'],
                    labels=['box 1', 'box 2', 'box 3', 'box 4', 'box 5'])
        plotted += ['grouped_df_noname ' + title2]

    else:
        not_plotted += ['grouped_df_noname ' + title2]
    if len(plotted) > 1:
        print(' '.join(plotted))
    if len(not_plotted) > 1:
        print(' '.join(not_plotted))


def main(args):
    parser = argparse.ArgumentParser(description='Train and evaluate models.')
    parser.add_argument('--path', type=str, default='drive/MyDrive/springExperiments/recurrent-real',
                        help='Path to training data directory.')
    parser.add_argument('--window', type=int, default=1000,
                        help='Size of evaluation window.')
    parser.add_argument('--plotting', action='store_true', default=False,
                        help='Enable plotting of evaluation data.')
    parser.add_argument('--det_env', action='store_true', help='Deterministic environment')
    parser.add_argument('--det_model', action='store_true', help='Deterministic model')
    parser.add_argument('--gtr', action='store_true', help='whether to plot gtr')
    parser.add_argument('--matrix', action='store_true', help='whether to plot transfer matrix')
    parser.add_argument('--tsne', action='store_true', help='whether to plot transfer matrix')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum subfolders')
    parser.add_argument('--pretrain_dir', type=str, default='', help='pretrained dir for curriculum')
    args = parser.parse_args(args)

    env_rand = 'det' if args.det_env else 'rand'
    model_rand = 'det' if args.det_model else 'rand'
    prefix = 'gtr' if args.gtr else env_rand + "_" + model_rand


    if args.curriculum:
        all_trained_folders = [p for p in os.scandir(args.path) if p.is_dir()]
        if args.pretrain_dir:
            all_starter_folders = [args.pretrain_dir]
    else:
        all_trained_folders = [args.path]

    for trained_folder in all_trained_folders:

        make_transfer_matrix_new(
            trained_folder,
            prefix,
            make_tsne=args.tsne,
            make_matrix=args.matrix)

        train_paths = [os.path.join(f.path) for f in os.scandir(trained_folder) if f.is_dir()]
        plot_train_many(train_paths, window=args.window, path=trained_folder)
        if args.pretrain_dir != '':
            start_paths = [os.path.join(f.path) for f in os.scandir(args.pretrain_dir) if f.is_dir()]

            plot_train_curriculum(start_paths, train_paths, window=args.window, path=trained_folder)

        for k, train_path in enumerate(train_paths):
            plot_train(train_path, window=args.window)

            train_path = os.path.join(train_path, 'evaluations')

            make_eval_figures(os.path.join(train_path, prefix + '_data.csv'),
                              os.path.abspath(os.path.join(train_path, '..', 'figures')), window=args.window,
                              prefix=prefix)


if __name__ == 'main':
    main(sys.argv[1:])
