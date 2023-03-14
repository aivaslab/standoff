import argparse
import os
import pandas as pd

from src.utils.display import process_csv, get_transfer_matrix_row
from src.utils.plotting import plot_merged, plot_selection, plot_split, plot_train


def make_eval_figures(path, figures_path, window=1, prefix=''):
    """
    plot the results found in one csv file.
    """
    merged_df = pd.DataFrame()
    merged_df_small = pd.DataFrame()
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
    figures_path_combined = os.path.join(figures_path, 'combined')
    if not os.path.exists(figures_path_combined):
        os.mkdir(figures_path_combined)

    # load csv, process data
    grouped_df, grouped_df_small, grouped_df_noname_abs, grouped_df_noname = process_csv(path, prefix)

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

            # normalize r after all the filtering
            for rows in [rows_big, rows_small]:
                rows["normed_r_unique_mean"] = (rows["r_mean"] - rows["r_mean"].min()) / ( \
                            rows["r_mean"].max() - rows["r_mean"].min())
                rows["normed_r_unique_std"] = (rows["r_std"]) / ( \
                            rows["r_mean"].max() - rows["r_mean"].min())

            for rows in [rows_big, rows_small]:
                rows["normed_episode_timesteps_mean"] = (rows["episode_timesteps_mean"] - rows[
                    "episode_timesteps_mean"].min()) / (
                                                                rows["episode_timesteps_mean"].max() - rows[
                                                            "episode_timesteps_mean"].min())
                rows["normed_episode_timesteps_std"] = (rows["episode_timesteps_std"]) / (
                        rows["episode_timesteps_mean"].max() - rows["episode_timesteps_mean"].min())

            plot_merged(indexer='model_ep', df=rows_small, mypath=extended_path, title=title + 'validAccuracy',
                        window=window,
                        values=['weakAccuracy', 'accuracy'],
                        labels=['selected any treat', 'selected correct treat'])
            plot_merged(indexer='model_ep', df=rows_big, mypath=extended_path, title=title + 'absoluteAccuracy',
                        window=window,
                        values=['weakAccuracy', 'accuracy', 'valid'],
                        labels=['selected any treat', 'selected correct treat',
                                'selected any box', ])
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
            plot_merged(indexer='model_ep', df=rows_small, mypath=extended_path, title=title + 'avoid', window=window, \
                        values=["avoidCorrect"], labels=["avoided correct box"])
    else:
        not_plotted += ['grouped_df ' + title2]

    if len(grouped_df_noname_abs):
        title2 = title + '-big'
        plot_merged(indexer='model_ep', df=grouped_df_noname_abs, mypath=figures_path_combined,
                    title=title2 + 'absoluteAccuracy', window=window,
                    values=['weakAccuracy', 'accuracy', 'normed_r', 'valid'],
                    labels=['selected any treat', 'selected correct treat',
                            'reward (normalized)', 'selected any box'])
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
        plot_merged(indexer='model_ep', df=grouped_df_noname, mypath=figures_path_combined, title=title2 + 'ac', window=window, \
                    values=["avoidCorrect"], labels=["avoided correct box"])
        plot_selection(indexer='model_ep', df=grouped_df_noname, mypath=figures_path_combined, title=title2, window=window)
        plot_merged(indexer='model_ep', df=grouped_df_noname, mypath=figures_path_combined, title=title2 + 'validAccuracy',
                    window=window, \
                    values=['weakAccuracy', 'accuracy', 'normed_r'], \
                    labels=['selected any treat', 'selected correct treat', \
                            'reward (normalized)'])
        plot_merged(indexer='model_ep', df=grouped_df_noname, mypath=figures_path_combined, title=title + 'absoluteLocation',
                    window=window, \
                    values=['sel0', 'sel1', 'sel2', 'sel3', 'sel4'], \
                    labels=['box 1', 'box 2', 'box 3', 'box 4', 'box 5'])
        plotted += ['grouped_df_noname ' + title2]

    else:
        not_plotted += ['grouped_df_noname ' + title2]
    if len(plotted) > 1:
        print(' '.join(plotted))
    if len(not_plotted) > 1:
        print(' '.join(not_plotted))


parser = argparse.ArgumentParser(description='Train and evaluate models.')
parser.add_argument('--path', type=str, default='drive/MyDrive/springExperiments/recurrent-real',
                    help='Path to training data directory.')
parser.add_argument('--window', type=int, default=1000,
                    help='Size of evaluation window.')
parser.add_argument('--plotting', action='store_true', default=False,
                    help='Enable plotting of evaluation data.')
parser.add_argument('--env_rand', type=str, choices=['rand', 'det'], default='rand', help='environment randomness')
parser.add_argument('--model_rand', type=str, choices=['rand', 'det'], default='rand', help='model randomness')
parser.add_argument('--gtr', action='store_true', help='whether to plot gtr')
parser.add_argument('--matrix', action='store_true', help='whether to plot transfer matrix')
args = parser.parse_args()

matrix_data = {}
for env_rand in [args.env_rand]:
    for model_rand in [args.model_rand]:
        matrix_data[env_rand + "_" + model_rand] = []
matrix_data['gtr'] = []

train_paths = [ os.path.join(f.path) for f in os.scandir(args.path) if f.is_dir() ]

for k, train_path in enumerate(train_paths):

    plot_train(train_path)

    train_path = os.path.join(train_path, 'evaluations')

    prefix = 'gtr' if args.gtr else args.env_rand + "_" + args.model_rand
    if args.matrix:
        this_matrix_data = get_transfer_matrix_row(os.path.join(train_path, prefix+'_data.csv'))
        this_matrix_data['start'] = train_path
        matrix_data[prefix] += [ this_matrix_data, ]
    else:
        make_eval_figures(os.path.join(train_path, prefix+'_data.csv'), os.path.abspath(os.path.join(train_path, '..', 'figures')), window=args.window, prefix=prefix)