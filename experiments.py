import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.supervised_learning import gen_data
from supervised_learning_main import run_supervised_session, calculate_statistics, write_metrics_to_file, save_figures


def add_label_and_combine_dfs(df_list, params, label):
    # Add 'regime' column to each DataFrame and combine them
    for i, df in enumerate(df_list):
        df[label] = params[i]
    combined_df = pd.concat(df_list)

    return combined_df


def create_combined_histogram(df, combined_avg, param, folder):
    plt.figure(figsize=(10, 6))
    for value in combined_avg[param].unique():
        value_df = combined_avg[combined_avg[param] == value]
        mean_acc_per_epoch = value_df.groupby('epoch')['accuracy'].mean()

        plt.plot(mean_acc_per_epoch.index, mean_acc_per_epoch.values,
                 label=f'{param} = {value}' if not isinstance(value, str) or value[0:3] != "N/A" else value)
        #plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
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
        hist_data.append(pd.Categorical(mean_acc))
        labels.append(f'{param} = {value}')


    #hist_data = np.asarray(hist_data, dtype=object)
    plt.hist(hist_data, bins=np.arange(0, 1.01, 0.05), stacked=True, label=labels, alpha=0.5)

    plt.title(f'Histogram of accuracy for last epoch for {param}')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.legend(loc='upper left')
    file_path = os.path.join(os.getcwd(), folder, f'hist_{param}.png')
    plt.savefig(file_path)
    plt.close()


def experiments(todo, repetitions, epochs, skip_train=False):
    """What is the overall performance of naive, off-the-shelf models on this task? Which parameters of competitive
    feeding settings are the most sensitive to overall model performance? To what extent are different models
    sensitive to different parameters? """

    regimes = [
        ('situational', ['a1']),
        ('informed', ['i0', 'i1']),
        ('contrastive', ['i0', 'u0', 'i1', 'u1']),
        ('complete', ['a0', 'i1']),
        ('comprehensive', ['a0', 'i1', 'u1'])
    ]
    default_regime = regimes[1]
    pref_types = [
        ('same', ''),
        #('different', 'd'),
        #('varying', 'v'),
    ]
    role_types = [
        ('subordinate', ''),
        #('dominant', 'D'),
        #('varying', 'V'),
    ]

    # generate supervised data
    labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target', 'correctSelection']
    oracles = [None] + labels
    oracle_names = [x if x is not None else "None" for x in oracles]
    if 0 in todo:
        print('Generating datasets with labels', labels)
        os.makedirs('supervised', exist_ok=True)
        for pref_type, pref_suffix in pref_types:
            for role_type, role_suffix in role_types:
                gen_data(labels, path='supervised', pref_type=pref_suffix, role_type=role_suffix)

    if 'h' in todo:
        print('Running hyperparameter search on all regimes, pref_types, role_types')


    # Experiment 1
    if 1 in todo:
        print('Running experiment 1: varied models training directly on the test set')

        # todo: add hparam search for many models, comparison between them?
        run_supervised_session(save_path=os.path.join('supervised', 'exp_1'),
                               repetitions=repetitions,
                               epochs=epochs,
                               train_sets=['a1'],
                               )

    if 2 in todo:
        print('Running experiment 2: varied oracle modules')
        combined_path_list = []
        last_path_list = []
        #os.makedirs(os.path.join('supervised', 'exp_2'), exist_ok=True)

        for single_oracle, oracle_name in zip(oracles[:3], oracle_names):
            print('oracle:', single_oracle)
            combined_paths, last_epoch_paths = run_supervised_session(save_path=os.path.join('supervised', 'exp_2', oracle_name),
                                   repetitions=repetitions,
                                   epochs=epochs,
                                   train_sets=regimes[4][1], # complete train regime, should be 3 for final
                                   oracle_labels=[single_oracle],
                                   skip_train=skip_train)
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)

        replace_dict = {'1': 1, '0': 0}

        print('loading dataframes')

        df_list = []
        for df_paths, oracle_name in zip(combined_path_list, oracle_names):
            for df_path in df_paths:
                chunks = pd.read_csv(df_path, chunksize=10000)
                for chunk in chunks:
                    chunk.replace(replace_dict, inplace=True)
                    chunk = chunk.assign(oracle=oracle_name)
                    df_list.append(chunk)
        combined_df = pd.concat(df_list, ignore_index=True)

        last_df_list = []
        for df_paths, oracle_name in zip(last_path_list, oracle_names):
            for df_path in df_paths:
                chunks = pd.read_csv(df_path, chunksize=10000)
                for chunk in chunks:
                    chunk.replace(replace_dict, inplace=True)
                    chunk = chunk.assign(oracle=oracle_name)
                    last_df_list.append(chunk)
        last_epoch_df = pd.concat(last_df_list, ignore_index=True)

        params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
                  'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
                  'uninformed_bait', 'uninformed_swap', 'first_swap', 'oracle']  # added 'oracle' here

        avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats = calculate_statistics(
            combined_df, last_epoch_df, params, skip_3x=True) #todo: make it definitely save one fixed param eg oracle

        create_combined_histogram(last_epoch_df, combined_df, 'oracle', os.path.join('supervised', 'exp_2'))

        combined_path = os.path.join('supervised', 'exp_2', 'c')
        os.makedirs(combined_path, exist_ok=True)
        write_metrics_to_file(os.path.join(combined_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats)
        save_figures(os.path.join(combined_path, 'figs'), combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                     params, last_epoch_df, num=12)



    # Experiment 7
    if 7 in todo:
        print('Running experiment 2: varied train regimes')
        df_list = []
        avg_list = []
        os.makedirs(os.path.join('supervised', 'exp_7'), exist_ok=True)
        for regime, train_sets in regimes:
            print('regime:', regime)
            combined_df, df = run_supervised_session(save_path=os.path.join('supervised', 'exp_7', regime),
                                        repetitions=repetitions,
                                        epochs=epochs,
                                        train_sets=train_sets)
            df_list.append(df)
            avg_list.append(combined_df)
        combined_df = add_label_and_combine_dfs(df_list, [regime for regime, _ in regimes], 'regime')
        combined_avg = add_label_and_combine_dfs(avg_list, [regime for regime, _ in regimes], 'regime')
        create_combined_histogram(combined_df, combined_avg, 'regime', os.path.join('supervised', 'exp_2'))

    # Experiment 3
    if 3 in todo:
        print('Running experiment 3: varied preferences')
        for pref_type, pref_suffix in pref_types:
            for regime, train_sets in [default_regime]:
                new_train_sets = [x + pref_suffix for x in train_sets]
                run_supervised_session(save_path=os.path.join('supervised', 'exp_3', pref_type, regime),
                                       repetitions=repetitions,
                                       epochs=epochs,
                                       train_sets=new_train_sets)

    # Experiment 4
    if 4 in todo:
        print('Running experiment 4: varied role')
        for role_type, role_suffix in role_types:
            for pref_type, pref_suffix in pref_types:
                for regime, train_sets in [default_regime]:
                    new_train_sets = [x + pref_suffix + role_suffix for x in train_sets]
                    run_supervised_session(save_path=os.path.join('supervised', 'exp_4', role_type, pref_type, regime),
                                           repetitions=repetitions,
                                           epochs=epochs,
                                           train_sets=new_train_sets)

    # Experiment 5
    if 5 in todo:
        print('Running experiment 5: varied collaboration')


    # Experiment 100
    if 100 in todo:
        print('Running experiment -1: testing effect of dense vs sparse inputs')
        # todo: add hparam search for many models, comparison between them?
        run_supervised_session(save_path=os.path.join('supervised', 'exp_100'),
                               repetitions=repetitions,
                               epochs=epochs,
                               train_sets=['a1'])


if __name__ == '__main__':
    experiments([2], 1, 20, skip_train=True)
