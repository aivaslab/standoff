import argparse
import os
import sys
import pandas as pd
from sklearn.manifold import TSNE

from src.utils.display import process_csv, get_transfer_matrix_row
from src.utils.plotting import plot_merged, plot_selection, plot_split, plot_train

from matplotlib import pyplot as plt


def make_transfer_matrix(path, prefix):
    train_paths = [ os.path.join(f.path) for f in os.scandir(path) if f.is_dir() ]
    print(train_paths)

    matrix_data = {}
    for env_rand in ['rand', 'det']:
        for model_rand in ['rand', 'det']:
            matrix_data[env_rand + "_" + model_rand] = []
    matrix_data['gtr'] = []

    for k, train_path in enumerate(train_paths):


        train_path = os.path.join(train_path, 'evaluations')

        #prefix = 'gtr'
        if os.path.exists(os.path.join(train_path, prefix+'_data.csv')):
            this_matrix_data = get_transfer_matrix_row(os.path.join(train_path, prefix+'_data.csv'), prefix)
            this_matrix_data['start'] = train_path
            matrix_data[prefix]+= [ this_matrix_data , ]

        '''for env_rand in ['rand', 'det']:
            for model_rand in ['rand', 'det']:
                prefix = env_rand + "_" + model_rand
                try:
                    this_matrix_data = get_transfer_matrix_row(os.path.join(train_path, prefix+'_data.csv'), prefix)
                    this_matrix_data['start'] = train_path
                    matrix_data[prefix]+= [ this_matrix_data , ]
                except BaseException as e:
                    print(e)'''
    good_ordering = ['swapped', 'misinformed', 'partiallyuninformed', 'replaced', 'informedcontrol', 'moved', 'removeduninformed', 'removedinformed']

    for env_rand in ['rand']:
        for model_rand in ['rand']:
            prefix = env_rand + "_" + model_rand
            good_matrix = {}
            extra_matrix = {}
            extra_names = []
            for x in good_ordering:
                good_matrix[x] = [0 for _ in good_ordering]

            #print(prefix)
            trix = matrix_data[prefix]
            #print(trix)
            #eval_names = [x.replace(" ", "") for x in trix[0].configName.unique()] # let's use this order for train names too
            train_names = []
            plt.figure()
            fig, ax = plt.subplots()

            for conf_type in ['accuracy_mean']:
                eval_names = [x.replace(" ", "") for x in trix[0]['nn']]
       
                print(eval_names)
                #reordering = [eval_names.index(x) for x in good_ordering]
                for k, line in enumerate(trix):
                    #print('line', line[conf_type], line['nn'])
                    string = line['start'].unique()[0]
                    print('string', string)
                    if 'S3' in string:
                        string = string[string.index('S3')+3:string.index('-3-')]
                    elif 'S2' in string:
                        string = 'S2'
                    elif 'S1' in string:
                        string = 'S1'
                    temp = line[conf_type]
                    train_names.append(string)
                    if string not in good_ordering:
                        extra_names.append(string)
                        extra_matrix[string] = [0 for _ in good_ordering]
                        for j, eval_name in enumerate(good_ordering):
                            extra_matrix[string][j] = temp[eval_names.index(eval_name)]
                        print(string, extra_matrix[string])
                    else:
                        for j, eval_name in enumerate(good_ordering):
                            good_matrix[string][j] = temp[eval_names.index(eval_name)]
                        print(string, good_matrix[string])

                lm = []
                print('extras', extra_names)
                for x in good_ordering:
                    lm += [good_matrix[x], ]
                for x in extra_names:
                    lm += [extra_matrix[x], ]

                row_names = [''] + good_ordering + extra_names
                ax.set_xticks(range(8))
                ax.set_xticklabels([''] + good_ordering)
                ax.set_yticks(range(len(row_names)))
                ax.set_yticklabels(row_names)

                plt.tick_params(axis='x', rotation=90)
                ax.matshow(lm, cmap=plt.cm.Blues)
                

                print(len(good_ordering), len(train_names), eval_names, train_names)

                for i in range(len(good_ordering)+len(extra_names)):
                    for j in range(len(good_ordering)):
                        ax.text(j, i, str(round(lm[i][j]*100)/100), va='center', ha='center')
                        
                plt.tight_layout()
                plt.savefig(os.path.join(path, 'matrix.png'))

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

            # normalize certain values after all the filtering
            
            for val in ['r', 'episode_timesteps']:
                for rows in [rows_big, rows_small]:
                    rows.loc[:, f"normed_{val}_mean"] = \
                        (rows[f"{val}_mean"] - rows[f"{val}_mean"].min()) \
                        / (rows[f"{val}_mean"].max() - rows[f"{val}_mean"].min())
                    rows.loc[:, f"normed_{val}_std"] = \
                        (rows[f"{val}_std"]) / \
                        (rows[f"{val}_mean"].max() - rows[f"{val}_mean"].min())

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

def main(args):
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
    args = parser.parse_args(args)
    
    prefix = 'gtr' if args.gtr else args.env_rand + "_" + args.model_rand

    if args.matrix:
        make_transfer_matrix(args.path, prefix)
    else:

        train_paths = [ os.path.join(f.path) for f in os.scandir(args.path) if f.is_dir() ]

        for k, train_path in enumerate(train_paths):

            plot_train(train_path)

            train_path = os.path.join(train_path, 'evaluations')

            make_eval_figures(os.path.join(train_path, prefix+'_data.csv'), os.path.abspath(os.path.join(train_path, '..', 'figures')), window=args.window, prefix=prefix)
            
if __name__ == 'main':
    main(sys.argv[1:])
