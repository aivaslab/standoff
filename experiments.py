import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.supervised_learning import gen_data
from supervised_learning_main import run_supervised_session


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
        hist_data.append(list(mean_acc))
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


def experiments(todo, repetitions, epochs):
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
        ('different', 'd'),
        ('varying', 'v'),
    ]
    role_types = [
        ('subordinate', ''),
        ('dominant', 'D'),
        ('varying', 'V'),
    ]

    # generate supervised data
    if 0 in todo:
        print('Generating datasets')
        for pref_type, pref_suffix in pref_types:
            for role_type, role_suffix in role_types:
                gen_data(['correctSelection'], path='supervised', pref_type=pref_suffix, role_type=role_suffix)

    if 'h' in todo:
        print('Running hyperparameter search on all regimes, pref_types, role_types')

    # Experiment 1
    if 1 in todo:
        print('Running experiment 1: varied models training directly on the test set')

        # todo: add hparam search for many models, comparison between them?
        run_supervised_session(save_path=os.path.join('supervised', 'exp_1'),
                               repetitions=repetitions,
                               epochs=epochs,
                               train_sets=['a1'])

    # Experiment 2
    if 2 in todo:
        print('Running experiment 2: varied train regimes')
        df_list = []
        avg_list = []
        for regime, train_sets in regimes[1:]:
            print('regime:', regime)
            combined_df, df = run_supervised_session(save_path=os.path.join('supervised', 'exp_2', regime),
                                        repetitions=repetitions,
                                        epochs=epochs,
                                        train_sets=train_sets)
            df_list.append(df)
            avg_list.append(combined_df)
        combined_df = add_label_and_combine_dfs(df_list, [regime for regime, _ in regimes[1:]], 'regime')
        combined_avg = add_label_and_combine_dfs(avg_list, [regime for regime, _ in regimes[1:]], 'regime')
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
    experiments([2], 1, 2)
