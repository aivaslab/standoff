import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

from src.utils.display import process_csv, get_transfer_matrix_row
from src.utils.plotting import plot_merged, plot_selection, plot_split, plot_train

from matplotlib import pyplot as plt

def plot_transfer_matrix(matrix_data, row_names, col_names, output_file):
    fig, ax = plt.subplots()
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)

    plt.tick_params(axis='x', rotation=90)
    ax.imshow(matrix_data, cmap=plt.cm.Blues, aspect='auto')

    for i in range(len(row_names)):
        for j in range(len(col_names)):
            ax.text(j, i, str(round(matrix_data[i][j] * 100) / 100), va='center', ha='center')

    plt.tight_layout()
    plt.savefig(output_file)
    
def plot_tsne(tsne_results, tsne_results_s, labels, labels_s, output_file):
    plt.figure()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='blue', marker='o', label='bigs2')
    plt.scatter(tsne_results_s[:, 0], tsne_results_s[:, 1], c='red', marker='x', label='big')
    plt.legend()

    for i, name in enumerate(labels):
        plt.annotate(name, (tsne_results[i, 0], tsne_results[i, 1]), textcoords="offset points", xytext=(-10, 5), ha='center')
    for i, name in enumerate(labels_s):
        plt.annotate(name, (tsne_results_s[i, 0], tsne_results_s[i, 1]), textcoords="offset points", xytext=(-10, 5), ha='center')

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE plot of model performance')
    plt.tight_layout()
    plt.savefig(output_file)
    


def make_transfer_matrix(paths, prefix, make_tsne=False):
    good_ordering = ['swapped', 'misinformed', 'partiallyuninformed', 'replaced', 'informedcontrol', 'moved', 'removeduninformed', 'removedinformed']
    
    train_paths_list = [[os.path.join(f.path) for f in os.scandir(path) if f.is_dir()] for path in paths]

    matrix_data_list = []
    for _ in range(len(paths)):
        matrix_data = {}
        for env_rand in ['rand', 'det']:
            for model_rand in ['rand', 'det']:
                matrix_data[env_rand + "_" + model_rand] = []
        matrix_data['gtr'] = []
        matrix_data_list.append(matrix_data)

    # load all data
    for train_paths, matrix_data in zip(train_paths_list, matrix_data_list):
        for train_path in train_paths:
            train_path = os.path.join(train_path, 'evaluations')
            print('train_path', train_path)
            if os.path.exists(os.path.join(train_path, prefix+'_data.csv')):
                this_matrix_data = get_transfer_matrix_row(os.path.join(train_path, prefix+'_data.csv'), prefix)
                this_matrix_data['train_path'] = train_path
                matrix_data[prefix].append(this_matrix_data)
                
    for env_rand in ['rand']:
        for model_rand in ['rand']:
            prefix = env_rand + "_" + model_rand
            all_matrices = []

            for matrix_data in matrix_data_list:
                good_matrix = {x: [0 for _ in good_ordering] for x in good_ordering}
                extra_matrix = {}
                train_names = []

                trix = matrix_data[prefix]
                eval_names = [x.replace(" ", "") for x in trix[0]['nn']]

                for k, line in enumerate(trix):
                    train_string = line['train_path'].unique()[0]
                    if 'S3' in train_string:
                        train_string = train_string[train_string.index('S3')+3:train_string.index('-3-')]
                    elif 'S2' in train_string:
                        train_string = 'S2'
                    elif 'S1' in train_string:
                        train_string = 'S1'

                    temp = line['accuracy_mean']
                    train_names.append(train_string)

                    if train_string not in good_ordering:
                        extra_matrix[train_string] = [0 for _ in good_ordering]
                        for j, eval_name in enumerate(good_ordering):
                            extra_matrix[train_string][j] = temp[eval_names.index(eval_name)]
                    else:
                        for j, eval_name in enumerate(good_ordering):
                            good_matrix[train_string][j] = temp[eval_names.index(eval_name)]

                all_matrices.append((good_matrix, extra_matrix, train_names))

            plot_matrices_and_tsne(all_matrices, path, make_tsne, good_ordering)
            
def plot_matrices_and_tsne(all_matrices, path, make_tsne, good_ordering):
    combined_matrices = []
    all_labels = []
    for good_matrix, extra_matrix, train_names in all_matrices:
        lm = [good_matrix[x] for x in good_ordering] + [extra_matrix[x] for x in train_names if x not in good_ordering]
        combined_matrices.append(lm)
        all_labels.extend(train_names)

    plot_transfer_matrix(matrix_data=combined_matrices, row_names=all_labels, col_names=good_ordering, output_file=os.path.join(path, 'accuracy_mean' + 'matrix.png'))

    if make_tsne:
        tsne_data_list = [np.array([good_matrix[x] for x in good_ordering] + [extra_matrix[x] for x in train_names if x not in good_ordering]) for good_matrix, extra_matrix, train_names in all_matrices]
        combined_data = np.vstack(tsne_data_list)
        tsne = TSNE(n_components=2, random_state=43, perplexity=min(30, combined_data.shape[0] - 1))
        tsne_results_combined = tsne.fit_transform(combined_data)

        tsne_results_list = []
        start_idx = 0
        for tsne_data in tsne_data_list:
            end_idx = start_idx + len(tsne_data)
            tsne_results_list.append(tsne_results_combined[start_idx:end_idx, :])
            start_idx = end_idx

        output_file = os.path.join(path, 'accuracy_mean' + 'tsne_plot.png')
        plot_tsne(tsne_results_list, all_labels, output_file)
        
def get_matrix_data(paths, only_last=False, metric='accuracy_mean'):
    train_paths_list = [[os.path.join(f.path) for f in os.scandir(path) if f.is_dir()] for path in paths]
    matrix_data = []
    for train_paths, dir_name in zip(train_paths_list, paths):
        for train_path in train_paths:
            train_path = os.path.join(train_path, 'evaluations')
            print('train_path', train_path)
            if os.path.exists(os.path.join(train_path, prefix+'_data.csv')):
                train_name = train_path
                timesteps, matrix_rows = get_transfer_matrix_row(os.path.join(train_path, prefix+'_data.csv'), prefix, only_last=only_last)
                for timestep, matrix_row in zip(timesteps, matrix_rows):
                    matrix_data.append({'dir': dir_name, 'train': train_name, 'timestep': timestep, 'vec': matrix_row[metric], 'uname': dir_name+train_name+str(timestep)})
        
def make_transfer_matrix_new(paths, prefix, make_matrix=False, make_tsne=False):
    eval_ordering = ['swapped', 'misinformed', 'partiallyuninformed', 'replaced', 'informedcontrol', 'moved', 'removeduninformed', 'removedinformed']
    
    if make_matrix:
        matrix_data = get_matrix_data(paths, only_last=True, metric='accuracy_mean')
        # make_matrix(matrix_data)
        new_matrix_dict = {}
        for row in matrix_data:
            train_string = row['train']
            new_matrix_dict[row['train']] = row['vec']
        
        new_matrix_rows = []
        new_matrix_labels = []
        # first we use our eval names, then we use all other train names
        for row in eval_ordering:
            if new_matrix_dict[row]:
                new_matrix_labels.append(row)
                new_matrix_rows.append(new_matrix_dict[row])
                new_matrix_dict[row] = None
        for row in new_matrix_dict.keys():
            if new_matrix_dict[row]:
                new_matrix_labels.append(row)
                new_matrix_rows.append(new_matrix_dict[row])
                new_matrix_dict[row] = None
                
        plot_transfer_matrix(matrix_data=new_matrix_rows, 
                             row_names=new_matrix_labels, 
                             col_names=eval_ordering, 
                             output_file=os.path.join(path, conf_type + 'matrix.png'))
            
    if make_tsne:
        matrix_data = get_matrix_data(paths, only_last=False, metric='accuracy_mean')
        
        combined_data = np.vstack([row['vec'] for row in matrix_data])
        tsne = TSNE(n_components=2, random_state=43, perplexity=min(30, combined_data.shape[0] - 1))
        
        tsne_results = tsne.fit_transform(combined_data)
        
        labels = [row['uname'] for row in matrix_data]
        dirs = [row['dir'] for row in matrix_data]
        agents = [row['train'] for row in matrix_data]
        output_file = os.path.join(path, conf_type + 'tsne_plot.png')
        plot_tsne(tsne_results, labels, output_file)
                    
    
def make_transfer_matrix_legacy(path, prefix, make_tsne=False):
    good_ordering = ['swapped', 'misinformed', 'partiallyuninformed', 'replaced', 'informedcontrol', 'moved', 'removeduninformed', 'removedinformed']
    train_paths = [ os.path.join(f.path) for f in os.scandir(path) if f.is_dir() ]
    
    special_path = 'save_dir/big'
    train_paths_s = [ os.path.join(f.path) for f in os.scandir(special_path) if f.is_dir() ]
    #special path used for comparison, don't hardcode, should be a list

    matrix_data = {}
    for env_rand in ['rand', 'det']:
        for model_rand in ['rand', 'det']:
            matrix_data[env_rand + "_" + model_rand] = []
    matrix_data['gtr'] = []
    
    matrix_data_s = {}
    for env_rand in ['rand', 'det']:
        for model_rand in ['rand', 'det']:
            matrix_data_s[env_rand + "_" + model_rand] = []
    matrix_data_s['gtr'] = []

    for tp, m in zip([train_paths, train_paths_s], [matrix_data, matrix_data_s]):
        for k, train_path in enumerate(tp):
            train_path = os.path.join(train_path, 'evaluations')
            print('train_path', train_path)
            if os.path.exists(os.path.join(train_path, prefix+'_data.csv')):
                this_matrix_data = get_transfer_matrix_row(os.path.join(train_path, prefix+'_data.csv'), prefix)
                this_matrix_data['train_path'] = train_path
                m[prefix]+= [ this_matrix_data , ]
    

    for env_rand in ['rand']:
        for model_rand in ['rand']:
            prefix = env_rand + "_" + model_rand
            good_matrix = {}
            extra_matrix = {}
            s_matrix = {}
            s_names =  []
            extra_names = []
            for x in good_ordering:
                good_matrix[x] = [0 for _ in good_ordering]

            #print(prefix)
            #print(trix)
            #eval_names = [x.replace(" ", "") for x in trix[0].configName.unique()] # let's use this order for train names too
            train_names = []
            plt.figure()
            fig, ax = plt.subplots()

            for conf_type in ['accuracy_mean']:
                trix = matrix_data[prefix]
                eval_names = [x.replace(" ", "") for x in trix[0]['nn']]
       
                print(eval_names)
                #reordering = [eval_names.index(x) for x in good_ordering]
                for k, line in enumerate(trix):
                    #print('line', line[conf_type], line['nn'])
                    train_string = line['train_path'].unique()[0]
                    if 'S3' in train_string:
                        train_string = train_string[train_string.index('S3')+3:train_string.index('-3-')]
                    elif 'S2' in train_string:
                        train_string = 'S2'
                    elif 'S1' in train_string:
                        train_string = 'S1'
                    temp = line[conf_type]
                    train_names.append(train_string)
                    if train_string not in good_ordering:
                        extra_names.append(train_string)
                        extra_matrix[train_string] = [0 for _ in good_ordering]
                        for j, eval_name in enumerate(good_ordering):
                            extra_matrix[train_string][j] = temp[eval_names.index(eval_name)]
                        print(train_string, extra_matrix[train_string])
                    else:
                        for j, eval_name in enumerate(good_ordering):
                            good_matrix[train_string][j] = temp[eval_names.index(eval_name)]
                        print(train_string, good_matrix[train_string])
                
                trix_s = matrix_data_s[prefix]
                for k, line in enumerate(trix_s):
                    #print('line', line[conf_type], line['nn'])
                    train_string = line['train_path'].unique()[0]
                    if 'S3' in train_string:
                        train_string = train_string[train_string.index('S3')+3:train_string.index('-3-')]
                    elif 'S2' in train_string:
                        train_string = 'S2'
                    elif 'S1' in train_string:
                        train_string = 'S1'
                    temp = line[conf_type]
                    s_names.append(train_string)
                    s_matrix[train_string] = [0 for _ in good_ordering]
                    for j, eval_name in enumerate(good_ordering):
                        s_matrix[train_string][j] = temp[eval_names.index(eval_name)]

                lm = []
                print('extras', extra_names)
                for x in good_ordering:
                    lm += [good_matrix[x], ]
                for x in extra_names:
                    lm += [extra_matrix[x], ]

                row_names = good_ordering + extra_names
                
                plot_transfer_matrix(matrix_data=lm, row_names=row_names, col_names=good_ordering, output_file=os.path.join(path, conf_type + 'matrix.png'))
                
                
                if make_tsne:
                    tsne_data = np.array([good_matrix[x] for x in good_ordering] + [extra_matrix[x] for x in extra_names])
                    tsne_data_s = np.array([s_matrix[x] for x in s_names])
                    combined_data = np.vstack([tsne_data, tsne_data_s])
                    tsne = TSNE(n_components=2, random_state=43, perplexity=min(30, tsne_data.shape[0] - 1))
                    tsne_results_combined = tsne.fit_transform(combined_data)
                    
                    tsne_results = tsne_results_combined[:len(tsne_data), :]
                    tsne_results_s = tsne_results_combined[len(tsne_data):, :]
                    
                    labels = good_ordering + extra_names
                    labels_s = s_names
                    output_file = os.path.join(path, conf_type + 'tsne_plot.png')
                    plot_tsne(tsne_results, tsne_results_s, labels, labels_s, output_file)
                    

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
    parser.add_argument('--tsne', action='store_true', help='whether to plot transfer matrix')
    args = parser.parse_args(args)
    
    prefix = 'gtr' if args.gtr else args.env_rand + "_" + args.model_rand

    if args.matrix or args.tsne:
        make_transfer_matrix_new([args.path, 'save_dir/big'], prefix, make_tsne=args.tsne)
    else:

        train_paths = [ os.path.join(f.path) for f in os.scandir(args.path) if f.is_dir() ]

        for k, train_path in enumerate(train_paths):

            plot_train(train_path)

            train_path = os.path.join(train_path, 'evaluations')

            make_eval_figures(os.path.join(train_path, prefix+'_data.csv'), os.path.abspath(os.path.join(train_path, '..', 'figures')), window=args.window, prefix=prefix)
            
if __name__ == 'main':
    main(sys.argv[1:])
