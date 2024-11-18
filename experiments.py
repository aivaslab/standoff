import argparse
import ast
import json
import os
import traceback
import multiprocessing
from itertools import product

import h5py

import pandas as pd
import tqdm
from scipy import stats

from src.pz_envs import ScenarioConfigs
from src.supervised_learning import gen_data
from src.utils.plotting import create_combined_histogram, plot_progression, save_key_param_figures, plot_learning_curves, make_splom, make_ifrscores, make_scatter, make_corr_things, make_splom_aux, plot_strategy_bar, \
    create_faceted_heatmap, plot_bar_graphs, plot_bar_graphs_special, plot_bar_graphs_new, plot_dependency_bar_graphs, plot_dependency_bar_graphs_new
from supervised_learning_main import run_supervised_session, calculate_statistics, write_metrics_to_file, save_figures, \
    train_model
import numpy as np
import random

import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_hparam_search(trials=64, repetitions=3, log_file='hparam_search_log.txt', train_sets=None, epochs=5):
    model_kwargs_base = {'hidden_size': [6, 8, 12, 16, 32],
                         'num_layers': [1, 2, 3],
                         'kernels': [4, 8, 16, 24, 32],
                         'kernel_size1': [1, 3, 5],
                         'kernel_size2': [1, 3, 5],
                         'stride1': [1, 2],
                         'pool_kernel_size': [2, 3],
                         'pool_stride': [1, 2],
                         'padding1': [0, 1],
                         'padding2': [0, 1],
                         'use_pool': [True, False],
                         'use_conv2': [True, False],
                         'kernels2': [8, 16, 32, 48],
                         'batch_size': [64, 128, 256],
                         'lr': [0.0005, 0.001, 0.002, 0.005]
                         }

    best_val_loss = float('inf')
    best_model_kwargs = None

    tried_models = 0

    while True:

        try:
            model_kwargs = {x: random.choice(model_kwargs_base[x]) for x in model_kwargs_base.keys()}
            print('kwargs', model_kwargs)
            cumulative_val = 0.0
            for repetition in range(repetitions):
                val_loss = train_model(train_sets, 'correct-loc', epochs=epochs, repetition=repetition,
                                       save_models=False, record_loss=True, model_kwargs=model_kwargs)
                cumulative_val += val_loss
            avg_val = cumulative_val / repetitions
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_kwargs = model_kwargs

            with open(log_file, 'a') as f:
                log_entry = {
                    'trial': tried_models,
                    'model_kwargs': model_kwargs,
                    'avg_val_loss': avg_val,
                    'best_model_kwargs': best_model_kwargs,
                    'best_val_loss': best_val_loss
                }
                f.write(json.dumps(log_entry) + '\n')
            tried_models += 1
            if tried_models > trials:
                break
        except BaseException as e:
            print(e)
            traceback.print_exc()


def string_to_list(s):
    """Converts the string format [2 2] to a list [2, 2]."""
    s = s.replace(' ', ',')
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return None

def informedness_to_str(informedness_list):
    mapping = {0: 'N', 1: 'F', 2: 'T'}
    if isinstance(informedness_list[0], list):
        # for some reason something in the Tt regime has an informedness list of length 5
        informedness_list = informedness_list[-1]
    first_val = mapping[informedness_list[0]]
    second_val = mapping[informedness_list[1]].lower()
    return first_val + second_val

def load_dataframes(combined_path_list, value_names, key_param):
    df_list = []

    print('loading and assigning key param', key_param)

    replace_dict = {'1': 1, '0': 0}
    tq = tqdm.trange(len(combined_path_list))
    print(combined_path_list)
    for df_paths, value_name in zip(combined_path_list, value_names):
        print('value name', value_name)

        for df_path in df_paths:
            #print(df_path, ('prior' in df_path) )
            file_name = os.path.basename(df_path)
            repetition = int(file_name.split('_')[-1].split('.')[0])

            epoch = file_name.split('_')[-2]
            if epoch == 'prior':
                epoch = 0
            else:
                epoch = int(epoch)
            retrain = ('retrain' in df_path)
            prior = ('prior' in df_path)
            if prior:
                epoch = 0
            #note that this only supports single digits
            try:
                chunks = pd.read_csv(df_path, chunksize=10000, compression='gzip')
            except:
                chunks = pd.read_csv(df_path, chunksize=10000)

            for chunk in chunks:
                chunk.replace(replace_dict, inplace=True)
                chunk['test_regime'] = chunk.apply(
                    lambda row: informedness_to_str(string_to_list(row['i-informedness'])) + str(row['opponents']),
                    axis=1
                )
                chunk = chunk.assign(**{key_param: value_name, 'repetition': repetition, 'retrain': retrain, 'prior': prior, 'epoch': epoch})
                df_list.append(chunk)

        tq.update(1)
    if len(df_list):
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df['i-informedness'] = combined_df['i-informedness'].fillna('none')

        return combined_df
    return None

def special_heatmap(df_path_list2, df_path_list, key_param='regime', key_param_list=[], exp_names=[], save_dir=None, params=None, used_regimes=None):
    print('special heatmap', save_dir)
    dfs = load_dataframes(df_path_list, key_param_list, key_param)
    #dfs2 = load_dataframes(df_path_list2, key_param_list, key_param)

    # next thing to check: print dfs and make sure it's same
    print('loaded, calculating stats...')
    _, _, _, _, _, _, _, key_param_stats, _, _, _ = calculate_statistics(
        None, dfs, list(set(params + [key_param])), skip_3x=True, skip_1x=True, key_param=key_param, used_regimes=used_regimes)
    print(key_param_stats)
    save_key_param_figures(save_dir, key_param_stats, None, key_param)
    print('done special')

def is_novel_task(row):
    sub_regime_keys = [
        "Nn", "Fn", "Nf", "Tn", "Nt", "Ff", "Tf", "Ft", "Tt"
    ]
    train_map = {
        's3': [x + '0' for x in sub_regime_keys] + [x + '1' for x in sub_regime_keys],
        's2': [x + '0' for x in sub_regime_keys] + ['Tt1'],
        's1': [x + '0' for x in sub_regime_keys],
        'homogeneous': ['Tt0', 'Ff0', 'Nn0', 'Tt1', 'Ff1', 'Nn1']
    }
    train_regime = row['regime'].split('-')[2]
    return not row['test_regime'] in train_map[train_regime]


def load_h5_data(h5_path, regime, retrain, prior):
    all_data = []
    with h5py.File(h5_path, 'r') as f:
        for dataset_name in f.keys():
            if dataset_name.endswith('_indices'):
                base_name = dataset_name[:-8]  # Remove '_indices'

                new_name = base_name.replace('_act', '-act').replace('_layer', '-layer').replace('acc_values', 'acc-values')

                #act_key, other_key, epoch, repetition = base_name.split('_')

                act_key, other_key, epoch, repetition, ifr_rep, model_type = new_name.split('_')
                indices = f[f"{base_name}_indices"][:]
                values = f[f"{base_name}_values"][:]
                acc_values = f[f"{base_name}_acc_values"][:]

                for idx, value, acc_value in zip(indices, values, acc_values):
                    all_data.append({
                        'act_key': act_key,
                        'other_key': other_key,
                        'epoch': int(epoch),
                        'repetition': int(repetition),
                        'ifr_rep': int(ifr_rep),
                        'model_type': model_type,
                        'id': idx,
                        'aux_task_loss': value,
                        'regime': regime,
                        'retrain': retrain,
                        'prior': prior,
                        'acc': acc_value,
                    })
    return pd.DataFrame(all_data)

def adjust_epochs(df):
    df = df.copy()
    df.loc[df['prior'] == True, 'epoch'] = 0
    df.loc[df['retrain'] == True, 'epoch'] += 100

    return df

def group_eval_df(all_epochs_df):
    grouped = all_epochs_df.groupby(['repetition', 'regime', 'epoch', 'retrain', 'prior'])  # ['accuracy'].agg(['mean', 'std']).reset_index()

    results = []
    for name, group in grouped:
        repetition, regime, epoch, retrain, prior = name
        key = '-'.join(map(str, name))

        acc = group['accuracy']
        acc_mean = acc.mean()
        acc_std = acc.std()

        novel_acc = group[group['is_novel_task']]['accuracy']
        mean_novel_accuracy = novel_acc.mean()
        std_novel_accuracy = novel_acc.std()

        non_novel_acc = group[~group['is_novel_task']]['accuracy']
        mean_xnovel_accuracy = non_novel_acc.mean()
        std_xnovel_accuracy = non_novel_acc.std()

        results.append({
            # 'key': key,
            'repetition': repetition,
            'regime': regime.replace('-retrain', '').replace('-prior', ''),
            'epoch': epoch,
            'retrain': retrain,
            'prior': prior,
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'novel_acc_mean': mean_novel_accuracy,
            'novel_acc_std': std_novel_accuracy,
            'familiar_acc_mean': mean_xnovel_accuracy,
            'familiar_acc_std': std_xnovel_accuracy,
        })

    grouped_df = pd.DataFrame(results)

    grouped_df = grouped_df.dropna(subset=['acc_mean'])
    grouped_df = grouped_df.rename(columns={'regime': 'model', 'repetition': 'rep'})

    print('length', len(grouped_df), )
    return grouped_df


def do_prediction_dependency(df, acc_name, output_path, retrain):
    results = []
    for model in df['regime'].unique():
        print('model', model)
        for act in ['all-activations', 'final-layer-activations']:

            # print(f"Unique values in 'regime': {df['regime'].unique()}")
            # print(f"Unique values in 'act_key': {df['act_key'].unique()}")

            act_data = df[(df['regime'] == model) & (df['act_key'] == act)]
            for is_novel in [False, True]:
                task_data = act_data[act_data['is_novel_task'] == is_novel]
                grouped = task_data.groupby(['id', 'repetition'])
                for feature in task_data['other_key'].unique():
                    #print(feature, act, is_novel)
                    has_matches = grouped.apply(lambda x: len(x[x['other_key'] == feature]) > 0)
                    if all(has_matches):
                        feature_acc = grouped.apply(lambda x: x[x['other_key'] == feature]['acc'].values[0]).astype(bool)
                        pred_acc = grouped.apply(lambda x: x['accuracy'].values[0]).astype(bool)

                        feature_acc_mean_per_id = feature_acc.groupby('id').mean().mean()
                        feature_acc_std_per_id = feature_acc.groupby('id').mean().std()

                        f_implies_p = np.logical_or(~feature_acc, pred_acc)
                        notf_implies_p = np.logical_or(feature_acc, pred_acc)

                        f_implies_p_mean_per_id = f_implies_p.groupby('id').mean()
                        notf_implies_p_mean_per_id = notf_implies_p.groupby('id').mean()

                        #n_trials = len(f_implies_p_mean_per_id)
                        #n_successes_f = np.sum(f_implies_p_mean_per_id > 0.5)
                        #n_successes_notf = np.sum(notf_implies_notp_mean_per_id > 0.5)

                        #p_value_f = stats.binom_test(n_successes_f, n_trials, p=0.5, alternative='greater')
                        #p_value_notf = stats.binom_test(n_successes_notf, n_trials, p=0.5, alternative='greater')

                        new_result = {
                            'retrain': retrain,
                            'model': model,
                            'activation': act,
                            'is_novel': is_novel,
                            'feature': feature,
                            'feature_acc': feature_acc_mean_per_id,
                            'feature_acc_std': feature_acc_std_per_id,
                            'f_implies_p_mean': f_implies_p_mean_per_id.mean(),
                            'f_implies_p_std': f_implies_p_mean_per_id.std(),
                            #'f_implies_p_p_value': p_value_f,
                            'notf_implies_p_mean': notf_implies_p_mean_per_id.mean(),
                            'notf_implies_p_std': notf_implies_p_mean_per_id.std(),
                            #'notf_implies_notp_p_value': p_value_notf,
                        }

                        results.append(new_result)
    #headers = ['retrain', 'model', 'activation', 'is_novel', 'feature', 'f_implies_p_mean', 'f_implies_p_std', 'f_implies_p_p_value', 'notf_implies_notp_mean', 'notf_implies_notp_std', 'notf_implies_notp_p_value']
    headers = ['retrain', 'model', 'activation', 'is_novel', 'feature', 'feature_acc', 'feature_acc_std', 'f_implies_p_mean', 'f_implies_p_std', 'notf_implies_p_mean', 'notf_implies_p_std']
    df_output = pd.DataFrame(results, columns=headers)
    filename = f"{acc_name}_dependencies_retrain_{retrain}.csv"
    df_output.to_csv(os.path.join(output_path, filename), index=False)


def generate_accuracy_tables(df, output_path, is_baseline=False, retrain=False):
    table_data = []
    combinations = df[['other_key', 'regime', 'model_type']].drop_duplicates()
    act_types = ['all-activations', 'final-layer-activations'] if not is_baseline else ['input-activations']

    for _, combo in tqdm.tqdm(combinations.iterrows()):
        feature, model, model_type = combo['other_key'], combo['regime'], combo['model_type']
        row = [feature, model, model_type]


        for act in act_types:
            act_data = df[(df['other_key'] == feature) &
                          (df['regime'] == model) &
                          (df['act_key'] == act) &
                          (df['model_type'] == model_type)]

            for is_novel in [False, True]:
                task_data = act_data[act_data['is_novel_task'] == is_novel]
                grouped = task_data.groupby(['repetition', 'ifr_rep'])

                combination_means = grouped['acc'].mean()

                overall_mean = combination_means.mean()
                between_std = combination_means.std()
                q1 = combination_means.quantile(0.25)
                q3 = combination_means.quantile(0.75)

                within_stds = grouped.apply(lambda x: x['acc'].std())
                avg_within_std = within_stds.mean() if len(within_stds) > 0 else np.nan

                row.extend([
                    overall_mean,
                    between_std,
                    avg_within_std,
                    q1,
                    q3
                ])

        table_data.append(row)

    headers = ['Feature', 'Model', 'Model_Type']
    for act in act_types:
        for task in ['Familiar', 'Novel']:
            headers.extend([
                f'{task} accuracy ({act})',
                f'{task} between-model std ({act})',
                f'{task} within-model std ({act})',
                f'{task} q1 ({act})',
                f'{task} q3 ({act})',
            ])

    df_output = pd.DataFrame(table_data, columns=headers)
    df_output['data_Type'] = 'baseline' if is_baseline else 'result'
    filename = f"all_table_retrain_{retrain}.csv" if not is_baseline else f"base_all_table_retrain_{retrain}.csv"
    df_output.to_csv(os.path.join(output_path, filename), index=False)

    return df_output

    for retrain in [True, False]:
        for prior in [True, False]:
            for act in ['all-activations', 'input-activations', 'final-layer_activations']:
                make_ifrscores(combined_df[(combined_df['act_key'] == act) & (combined_df['retrain'] == retrain) & (combined_df['prior'] == prior)], output_path, act, retrain, prior)

def do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list, used_regimes=None):
    #print('loading dataframes for final comparison')

    #print('kp list', key_param_list) # has all 9 values

    combined_path = os.path.join('supervised', exp_name, 'c')
    os.makedirs(combined_path, exist_ok=True)

    print('loading IFRs')
    folder_paths = set()
    for sublist in lp_list:
        for file_path in sublist:
            folder_path = os.path.dirname(file_path)
            folder_paths.add(folder_path)

    dataframes = []
    for folder in folder_paths:
        csv_path = os.path.join(folder, 'ifr_df.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                base_name = os.path.basename(folder)
                base_name = base_name.replace('-retrain', '')
                df = df.dropna(subset=['epoch'])
                #print('uniques', df['epoch'].unique())
                df['prior'] = df['epoch'] == 0
                df['model'] = base_name
                df['retrain'] = 'retrain' in folder
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    if len(dataframes) > 0:
        combined_ifr_df = pd.concat(dataframes, ignore_index=True)
        combined_ifr_df.to_csv(os.path.join(combined_path, 'combined_ifr_df.csv'), index=False)
    columns_to_ignore = ['loss']
    columns_to_consider = [col for col in combined_ifr_df.columns if col not in columns_to_ignore]
    current_length = len(combined_ifr_df)
    combined_ifr_df = combined_ifr_df.drop_duplicates(subset=columns_to_consider)
    combined_ifr_df = adjust_epochs(combined_ifr_df)
    print('duplicates removed', current_length, len(combined_ifr_df))

    if True:
        print('loading baseline h5s!!!!!')
        all_baseline_data = []
        base_path = os.path.dirname(list(folder_paths)[0])
        for regime in ['s1', 's2', 's3']:
            individual_data = load_h5_data(os.path.join(base_path, f'indy_ifr_base-{regime}-0.h5'), regime=regime, retrain=False, prior=False)
            all_baseline_data.append(individual_data)
        all_baseline_df = pd.concat(all_baseline_data, ignore_index=True)
        #filtered = all_baseline_df[all_baseline_df['model_type'] == 'mlp1']
        #if len(filtered):
        #    all_baseline_df = filtered
        #else:
        #    print("couldn't find mlp1 baselines")
        #print(all_baseline_df['act_key'].unique())


        ### Get individual data
        print('loading model h5s')
        all_indy_data = []
        for folder in folder_paths:
            regime = os.path.basename(folder).replace('-retrain', '')
            retrain = 'retrain' in folder
            prior = 'prior' in folder
            for rep in [0, 1, 2]:
                individual_data = load_h5_data(os.path.join(folder, f'indy_ifr{rep}.h5'), regime=regime, retrain=retrain, prior=prior)
                all_indy_data.append(individual_data)
        all_indy_df = pd.concat(all_indy_data, ignore_index=True)



        if False:
            all_indy_all = all_indy_df[all_indy_df['act_key'] == 'all-activations']
            all_indy_final = all_indy_df[all_indy_df['act_key'] == 'final-layer-activations']
            make_splom_aux(all_indy_all, 'all', os.path.join(combined_path, 'sploms'))
            make_splom_aux(all_indy_final, 'final', os.path.join(combined_path, 'sploms'))
            make_scatters2(all_indy_all, all_indy_final, other_keys, combined_path)

        print('loading evaluation results')


        all_epochs_df = load_dataframes(combined_path_list, key_param_list, key_param)
        all_epochs_df['is_novel_task'] = all_epochs_df.apply(is_novel_task, axis=1)
        print('columns and epochs', all_epochs_df.columns, all_epochs_df['epoch'].unique())

        # run experiment 3:
        if True:
            print('running experiment 3')
            print(all_epochs_df.columns)
            print(all_indy_df.columns)
            for retrain in [True, False]:
                merged_df = pd.merge(all_indy_df[(all_indy_df['model_type'] == 'mlp1') & (all_indy_df['retrain'] == retrain)], all_epochs_df[['id', 'regime', 'is_novel_task', 'accuracy', 'repetition']], on=['id', 'regime', 'repetition'], how='left')
                merged_df = merged_df.drop_duplicates()
                print(len(merged_df))
                print(merged_df.head())
                print(merged_df.dtypes)
                print(merged_df['accuracy'].describe())
                print(merged_df['other_key'].value_counts())
                do_prediction_dependency(merged_df, 'accuracy', combined_path, retrain)


        all_epochs_df['regime_short'] = all_epochs_df['regime'].str[-2:]
        merged_baselines = pd.merge(all_baseline_df, all_epochs_df[['id', 'regime_short', 'is_novel_task', 'retrain']], left_on=['id', 'regime', 'retrain'], right_on=['id', 'regime_short', 'retrain'], how='left')
        all_baselines_df = merged_baselines
        all_baselines_df = all_baselines_df.drop_duplicates()

        # figure out whether each item in all_indy_df is novel:
        merged_df = pd.merge(all_indy_df, all_epochs_df[['id', 'regime', 'is_novel_task', 'retrain']], on=['id', 'regime', 'retrain'], how='left')
        all_indy_df = merged_df
        all_indy_df = all_indy_df.drop_duplicates()

        print('generating accuracy tables')
        if False:
            for retrain in [True, False]:
                baseline_tables = generate_accuracy_tables(all_baselines_df[(all_baselines_df['retrain'] == retrain) & (all_baselines_df['prior'] == False)], combined_path, is_baseline=True, retrain=retrain)
                result_tables = generate_accuracy_tables(all_indy_df[(all_indy_df['retrain'] == retrain) & (all_indy_df['prior'] == False)], combined_path, retrain=retrain)

    # make pred dep heatmaps
    if True:
        print('doing dependency heatmaps')
        for strat in ['normal', 'box']:
            if strat == "normal":
                strategies = {
                    'No-Mindreading': ['opponents', 'big-loc', 'small-loc'],
                    'Low-Mindreading': ['vision', 'fb-exist'],
                    'High-Mindreading': ['fb-loc', 'b-loc', 'target-loc']
                }
            else:
                strategies = {
                    'No-Mindreading': ['big-loc', 'big-box', 'small-loc', 'small-box'],
                    'High-Mindreading': ['fb-loc', 'fb-box', 'b-loc', 'b-box', 'target-loc', 'target-box',]
                }


            for retrain in [False, True]:
                dep_df = pd.read_csv(os.path.join(combined_path, f'accuracy_dependencies_retrain_{retrain}.csv'))

                for layer in ['all-activations', 'final-layer-activations']:
                        plot_dependency_bar_graphs_new(dep_df, combined_path, strategies, True, retrain=retrain, strat=strat, layer=layer)
            #create_faceted_heatmap(dep_df, True, 'final-layer-activations', os.path.join(combined_path, 'test.png'), strategies)

    strategies_short = {
        'Opponents': ['opponents'],
        'Location Beliefs': ['b-loc']
    }
    strategies_long = {
        'No-Mindreading': ['pred', 'opponents', 'big-loc', 'small-loc'],
        'Low-Mindreading': ['vision', 'fb-exist'],
        'High-Mindreading': ['fb-loc', 'b-loc', 'target-loc', 'labels']
    }
    strategies_both = {
        'No-Mindreading': ['big-loc', 'big-box', 'small-loc', 'small-box'],
        'High-Mindreading': ['fb-loc', 'fb-box', 'b-loc', 'b-box', 'target-loc', 'target-box', ]
    }
    for retrain in [True, False]:
        baseline_tables = pd.read_csv(os.path.join(combined_path, f'base_all_table_retrain_False.csv'))
        result_tables = pd.read_csv(os.path.join(combined_path, f'all_table_retrain_{retrain}.csv'))

        print('made acc tables', len(baseline_tables), len(result_tables))
        print(baseline_tables.columns, result_tables.columns)

        this_path = os.path.join(combined_path, f'strats-rt-{retrain}')

        for result_type in ['mlp1', 'linear']:
            for layer in ['all', 'final-layer']:
                plot_bar_graphs_new(baseline_tables[(baseline_tables['Model_Type'] == result_type)], result_tables[(result_tables['Model_Type'] == result_type)], this_path, strategies_short, layer=layer, r_type=f'spec-{result_type}')
                plot_bar_graphs_new(baseline_tables[(baseline_tables['Model_Type'] == result_type)], result_tables[(result_tables['Model_Type'] == result_type)], this_path, strategies_long, layer=layer, r_type=result_type)
                plot_bar_graphs_new(baseline_tables[(baseline_tables['Model_Type'] == result_type)], result_tables[(result_tables['Model_Type'] == result_type)], this_path, strategies_both, layer=layer, r_type=f'both-{result_type}')

        #plot_bar_graphs_special(baseline_tables, result_tables, os.path.join(combined_path, 'strats'), strategies)
        #plot_bar_graphs(baseline_tables[(baseline_tables['Model_Type'] == 'mlp1')], result_tables[(result_tables['Model_Type'] == 'mlp1')], os.path.join(combined_path, f'strats-rt-{retrain}'), strategies)

    print('Original columns and epochs:', all_epochs_df.columns, all_epochs_df['epoch'].unique())

    all_epochs_df = adjust_epochs(all_epochs_df)

    grouped_df = group_eval_df(all_epochs_df)

    print('merging dfs')
    merged_df = pd.merge(combined_ifr_df, grouped_df, on=['rep', 'model', 'epoch', 'retrain', 'prior'])
    print('after merge', merged_df['epoch'].unique(), merged_df['retrain'].unique(), merged_df['prior'].unique())

    # SPLOM
    for act in ['all_activations', 'final_layer_activations', 'input_activations']:
        try:
            make_splom(merged_df[(merged_df['act'] == act)], combined_path, act, False, True)
            make_scatter(merged_df[merged_df['act'] == act], combined_path, act)
        except BaseException as e:
            print('failed a splom', e)

    make_corr_things(merged_df, combined_path)

    #mean_correlation_df = correlation_df.groupby('feature')['correlation'].mean().reset_index()

    last_epoch_df = load_dataframes(last_path_list, key_param_list, key_param)

    print('last epoch df columns', last_epoch_df.columns)

    if all_epochs_df is not None and last_epoch_df is not None:
        create_combined_histogram(last_epoch_df, all_epochs_df, key_param, os.path.join('supervised', exp_name))

        avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats, oracle_stats, delta_sum, delta_x = calculate_statistics(
            all_epochs_df, last_epoch_df, list(set(params + prior_metrics + [key_param])),
            skip_3x=True, skip_1x=True, key_param=key_param, used_regimes=used_regimes, savepath=os.path.join('supervised', exp_name), last_timestep=True)



    write_metrics_to_file(os.path.join(combined_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats,
                          key_param=key_param, d_s=delta_sum, d_x=delta_x)
    save_figures(combined_path, combined_ifr_df, avg_loss, ranges_2, range_dict, range_dict3,
                 params, last_epoch_df, num=12, key_param_stats=key_param_stats, oracle_stats=oracle_stats,
                 key_param=key_param, delta_sum=delta_sum, delta_x=delta_x)


def experiments(todo, repetitions, epochs=50, batches=5000, skip_train=False, skip_calc=False, batch_size=64, desired_evals=5,
                skip_eval=False, skip_activations=False, last_timestep=True, retrain=False, current_model_type=None, current_label=None, current_label_name=None,
                comparison=False):
    """What is the overall performance of naive, off-the-shelf models on this task? Which parameters of competitive
    feeding settings are the most sensitive to overall model performance? To what extent are different models
    sensitive to different parameters? """
    save_every = max(1, epochs // desired_evals)

    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap', 'test_regime']
    prior_metrics = ['shouldAvoidSmall', 'correct-loc', 'incorrect-loc',
                     'shouldGetBig', 'informedness', 'p-b-0', 'p-b-1', 'p-s-0', 'p-s-1', 'delay', 'opponents']

    sub_regime_keys = [
        "Nn","Fn", "Nf","Tn", "Nt","Ff","Tf", "Ft","Tt"
    ]
    all_regimes = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-' + x + '1' for x in sub_regime_keys]
    mixed_regimes = {k: ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-' + k + '1'] for k in sub_regime_keys}

    regimes = {}
    regimes['direct'] = ['sl-' + x + '1' for x in sub_regime_keys]
    regimes['noOpponent'] = ['sl-' + x + '0' for x in sub_regime_keys]
    regimes['everything'] = all_regimes
    hregime = {}
    hregime['homogeneous'] = ['sl-Tt0', 'sl-Ff0', 'sl-Nn0', 'sl-Tt1', 'sl-Ff1', 'sl-Nn1']
    #hregime['identity'] = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-Tt1', 'sl-Ff1', 'sl-Nn1']
    sregime = {}
    sregime['special'] = ['sl-Tt0', 'sl-Tt1', 'sl-Nt0', 'sl-Nt1', 'sl-Nf0', 'sl-Nf1', 'sl-Nn0', 'sl-Nn1']

    fregimes = {}
    fregimes['s1'] = regimes['noOpponent']
    fregimes['s3'] = regimes['everything']
    fregimes['s2'] = mixed_regimes['Tt']
    #fregimes['homogeneous'] = hregime['homogeneous']

    single_regimes = {k[3:]: [k] for k in all_regimes}
    leave_one_out_regimes = {}
    for i in range(len(sub_regime_keys)):
        regime_name = "lo_" + sub_regime_keys[i]
        leave_one_out_regimes[regime_name] = ['sl-' + x + '0' for x in sub_regime_keys]
        ones = ['sl-' + x + '1' for j, x in enumerate(sub_regime_keys) if j != i]
        leave_one_out_regimes[regime_name].extend(ones)

    pref_types = [
        ('same', ''), # ('different', 'd'), # ('varying', 'v'),
    ]
    role_types = [
        ('subordinate', ''), # ('dominant', 'D'), # ('varying', 'V'),
    ]

    # labels for ICLR, including size just in case

    model_type = "loc" # or box
    labels = [
        'id', 'i-informedness', # must have these or it will break
        'opponents',
        'big-loc',
        'small-loc',
        'target-loc',
        'b-loc',
        'fb-loc',
        'fb-exist',
        'vision',
        'big-box',
        'small-box',
        'target-box',
        'b-box',
        'fb-box',
        'box-locations'
              ]

    # box-locations could be added to check for type conversions

    '''labels = ['loc', 'vision', 'b-loc', 'b-exist', 'exist', 'box-updated',
              'saw-last-update', 'target-loc',  'opponents', 'id',
              'informedness', 'swap-treat', 'swap-loc',
              'bait-loc', 'i-informedness', 'i-b-loc',
              'i-b-exist', 'i-target-loc', #'i-target-size',
              #"treat-box", "b-treat-box", "i-b-treat-box",
              #"target-box", "i-target-box",
              #"box-locations", "b-box-locations", "i-b-box-locations",
              #"correct-box", 'b-correct-box', 'i-b-correct-box',
              #'target-size',
              "b-correct-loc", "i-b-correct-loc", 'shouldGetBig',
              "b-loc-diff", "i-b-loc-diff", "b-exist-diff", "i-b-exist-diff",
              "fb", "i-fb"
              ] #'last-vision-span',

    ### Removed these to speed it up
    for name in ["loc", "b-loc", "i-b-loc", "b-loc-diff", "i-b-loc-diff"]:
        labels += ["big-" + name, "small-" + name]#, , "any-" + name,] "scalar-" + name,'''

    oracles = labels + [None]
    conf = ScenarioConfigs()
    exp_name = f'exp_{todo[0]}'
    if last_timestep:
        exp_name += "-L"

    session_params = {
        'repetitions': repetitions,
        'epochs': epochs,
        'batches': batches,
        'skip_train': skip_train,
        'skip_eval': skip_eval,
        'batch_size': batch_size,
        'prior_metrics': list(set(prior_metrics + labels)),
        'save_every': save_every,
        'skip_calc': skip_calc,
        'act_label_names': labels,
        'skip_activations': skip_activations,
        #'oracle_is_target': False,
        'last_timestep': last_timestep,
    }
    if 0 in todo:
        print('Generating datasets with labels', labels)
        os.makedirs('supervised', exist_ok=True)
        for pref_type, pref_suffix in pref_types:
            for role_type, role_suffix in role_types:
                gen_data(labels, path='supervised', pref_type=pref_suffix, role_type=role_suffix,
                         prior_metrics=prior_metrics, conf=conf)

    if 'h' in todo:
        print('Running hyperparameter search on all regimes, pref_types, role_types')
        run_hparam_search(trials=100, repetitions=3, log_file='hparam_file.txt', train_sets=regimes['direct'], epochs=20)

    if 2 in todo:
        print('Running experiment 1: base, different models and answers')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = False

        if current_model_type != None:
            model_types = [current_model_type]
            label_tuples = [(current_label, current_label_name)]
        else:
            model_types = ['cnn', 'smlp', 'clstm']
            label_tuples = [('correct-loc', 'loc')]

        for label, label_name in label_tuples: #[('correct-loc', 'loc'), ('correct-box', 'box'), ('shouldGetBig', 'size')]:
            for model_type in model_types:#['smlp', 'cnn', 'clstm', ]:
                for regime in list(fregimes.keys()):
                    kpname = f'{model_type}-{label_name}-{regime}'
                    print(model_type + '-' + label_name, 'regime:', regime, 'train_sets:', fregimes[regime])
                    combined_paths, last_epoch_paths, lp = run_supervised_session(
                        save_path=os.path.join('supervised', exp_name, kpname),
                        train_sets=fregimes[regime],
                        eval_sets=fregimes['s3'],
                        oracle_labels=[None],
                        key_param=key_param,
                        key_param_value=kpname,
                        label=label,
                        model_type=model_type,
                        do_retrain_model=retrain,
                        **session_params
                    )
                    conditions = [
                        (lambda x: 'prior' not in x and 'retrain' not in x, ''),
                        #(lambda x: 'prior' in x and 'retrain' not in x, '-prior'),
                        #(lambda x: 'prior' not in x and 'retrain' in x, '-retrain')
                    ]

                    print('paths found', combined_paths, last_epoch_paths)

                    for condition, suffix in conditions:
                        last_path_list.append([x for x in last_epoch_paths if condition(x)])
                        combined_path_list.append([x for x in combined_paths if condition(x)])
                        key_param_list.append(kpname + suffix)
                    lp_list.append(lp) # has x, x-retrain currently

        if comparison:
            do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 22 in todo:
        print('Running experiment 22: ablate')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = False

        if current_model_type != None:
            model_types = [current_model_type]
            label_tuples = [(current_label, current_label_name)]
        else:
            model_types = ['ablate-hardcoded']
            label_tuples = [('correct-loc', 'loc')]

        for label, label_name in label_tuples:
            for model_type in model_types:
                for regime in list(fregimes.keys()):
                    kpname = f'{model_type}-{label_name}-{regime}'
                    print(model_type + '-' + label_name, 'regime:', regime, 'train_sets:', fregimes[regime])
                    combined_paths, last_epoch_paths, lp = run_supervised_session(
                        save_path=os.path.join('supervised', exp_name, kpname),
                        train_sets=fregimes[regime],
                        eval_sets=fregimes['s3'],
                        oracle_labels=[None],
                        key_param=key_param,
                        key_param_value=kpname,
                        label=label,
                        model_type=model_type,
                        do_retrain_model=retrain,
                        **session_params
                    )
                    conditions = [
                        (lambda x: 'prior' not in x and 'retrain' not in x, ''),
                        #(lambda x: 'prior' in x and 'retrain' not in x, '-prior'),
                        #(lambda x: 'prior' not in x and 'retrain' in x, '-retrain')
                    ]

                    print('paths found', combined_paths, last_epoch_paths)

                    for condition, suffix in conditions:
                        last_path_list.append([x for x in last_epoch_paths if condition(x)])
                        combined_path_list.append([x for x in combined_paths if condition(x)])
                        key_param_list.append(kpname + suffix)
                    lp_list.append(lp)

        if comparison:
            do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)


def run_single_experiment(args_tuple):
    model_type, (label, label_name), args = args_tuple
    print(f"Running experiment with model_type: {model_type}, label: {label}, label_name: {label_name}")
    print(f"Process: {multiprocessing.current_process().name}")

    experiments([22],
                repetitions=3,
                batches=10000,
                skip_train=not args.t,
                skip_eval=not args.e,
                skip_calc=not args.c,
                skip_activations=not args.a,
                retrain=args.r,
                batch_size=256,
                desired_evals=1,
                last_timestep=True,
                current_model_type=model_type,
                current_label=label,
                current_label_name=label_name,
                comparison=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments with various options.")

    parser.add_argument('-t', action='store_true', help='training')
    parser.add_argument('-e', action='store_true', help='evaluation')
    parser.add_argument('-c', action='store_true', help='calculation')
    parser.add_argument('-a', action='store_true', help='activations')
    parser.add_argument('-r', action='store_true', help='Retrain the model')
    parser.add_argument('-d', action='store_true', help='dont run in parallel and do the end thing')
    parser.add_argument('-g', action='store_true', help='generate dataset')

    args = parser.parse_args()

    #model_types = [ 'cnn', 'smlp', 'clstm']
    model_types = ['clstm']
    labels = [('correct-loc', 'loc')]

    if (not args.d) and (not args.g):
        experiment_args = [(model_type, label, args) for model_type, label in product(model_types, labels)]

        with multiprocessing.Pool() as pool:
            pool.map(run_single_experiment, experiment_args)
    else:
        experiments([22] if not args.g else [0],
                repetitions=3,
                batches=10000,
                skip_train=not args.t,
                skip_eval=not args.e,
                skip_calc=not args.c,
                skip_activations=not args.a,
                retrain=args.r,
                batch_size=256,
                desired_evals=1,
                last_timestep=True,
                comparison=args.d)

    print("finished")
