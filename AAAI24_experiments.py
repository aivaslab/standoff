import json
import os
import traceback

import pandas as pd

from src.pz_envs import ScenarioConfigs
from src.supervised_learning import gen_data
from src.utils.plotting import create_combined_histogram
from supervised_learning_main import run_supervised_session, calculate_statistics, write_metrics_to_file, save_figures, \
    train_model
import numpy as np
import random

import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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
            # produce a random set of kwargs, from our base set above
            model_kwargs = {x: random.choice(model_kwargs_base[x]) for x in model_kwargs_base.keys()}
            print('kwargs', model_kwargs)
            cumulative_val = 0.0
            for repetition in range(repetitions):
                val_loss = train_model(train_sets, 'correctSelection', epochs=epochs, repetition=repetition,
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


def add_label_and_combine_dfs(df_list, params, label):
    # Add 'regime' column to each DataFrame and combine them
    for i, df in enumerate(df_list):
        df[label] = params[i]
    combined_df = pd.concat(df_list)

    return combined_df


def load_dataframes(combined_path_list, value_names, key_param):
    df_list = []

    replace_dict = {'1': 1, '0': 0}
    for df_paths, value_name in zip(combined_path_list, value_names):
        for df_path in df_paths:
            chunks = pd.read_csv(df_path, chunksize=10000)
            for chunk in chunks:
                chunk.replace(replace_dict, inplace=True)
                chunk = chunk.assign(**{key_param: value_name})
                df_list.append(chunk)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['informedness'] = combined_df['informedness'].fillna('none')
    return combined_df


def experiments(todo, repetitions, epochs, skip_train=False, skip_calc=False, batch_size=64, desired_evals=5,
                use_ff=False, skip_eval=False):

    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap']
    prior_metrics = ['shouldAvoidSmall', 'correctSelection', 'incorrectSelection',
                     'shouldGetBig', 'informedness', 'p-b-0', 'p-b-1', 'p-s-0', 'p-b-1', 'delay']

    sub_regime_keys = [
        "",
        "eb", "es",
        "eb-lb", "es-ls",
        "eb-es",
        "eb-es-lb", "eb-es-ls",
        "eb-es-lb-ls"
    ]
    sub_regime_mapping = {
        'noInfo': '',
        'bigExist': 'eb',
        'smallExist': 'es',
        'bothExist': 'eb-es',
        'bigExistBigLoc': 'eb-lb',
        'smallExistSmallLoc': 'es-ls',
        'bothExistBigLoc': 'eb-es-lb',
        'bothExistSmallLoc': 'eb-es-ls',
        'bothExistBothLoc': 'eb-es-lb-ls'
    }
    regimes = {k: ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-' + v + '1'] for k, v in sub_regime_mapping.items()}
    regimes['direct'] = ['sl-' + x + '1' for x in sub_regime_keys]
    regimes['noOpponent'] = ['sl-' + x + '0' for x in sub_regime_keys]

    default_regime = regimes['bothExistBothLoc'] # the non noOpponent regime used for standard training
    pref_types = [
        ('same', ''),
    ]
    role_types = [
        ('subordinate', ''),
    ]

    labels = ['loc', 'vision', 'b-loc', 'target']
    oracles = labels + [None]
    oracle_names = [x if x is not None else "None" for x in oracles]
    conf = ScenarioConfigs()

    # generate supervised data
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

    # Experiment 2
    if 2 in todo:
        print('Running experiment 1: varied models training directly on the test set')

        save_every = max(1, epochs // desired_evals)
        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        exp_name = 'exp_2' if not use_ff else 'exp_2-f'

        for regime in list(regimes.keys()):
            print('regime:', regime)
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                repetitions=repetitions,
                epochs=epochs,
                train_sets=regimes[regime],
                eval_sets=regimes['direct'],
                oracle_labels=[None],
                skip_train=skip_train,
                skip_eval=skip_eval,
                batch_size=batch_size,
                prior_metrics=list(set(prior_metrics + labels)),
                key_param=key_param,
                key_param_value=regime,
                save_every=save_every,
                skip_calc=skip_calc,
                use_ff=use_ff
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)

        print('loading dataframes for final comparison')

        combined_df = load_dataframes(combined_path_list, regimes.keys(), key_param)
        last_epoch_df = load_dataframes(last_path_list, regimes.keys(), key_param)

        create_combined_histogram(last_epoch_df, combined_df, key_param, os.path.join('supervised', exp_name))

        avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats, oracle_stats, delta_sum, delta_x = calculate_statistics(
            combined_df, last_epoch_df, list(set(params + prior_metrics + [key_param])),
            skip_3x=True, key_param=key_param)  # todo: make it definitely save one fixed param eg oracle

        combined_path = os.path.join('supervised', exp_name, 'c')
        os.makedirs(combined_path, exist_ok=True)
        write_metrics_to_file(os.path.join(combined_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats,
                              key_param=key_param, d_s=delta_sum, d_x=delta_x)
        save_figures(os.path.join(combined_path, 'figs'), combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                     params, last_epoch_df, num=12, key_param_stats=key_param_stats, oracle_stats=oracle_stats, key_param=key_param, delta_sum=delta_sum)

    if 3 in todo:
        print('Running experiment 3: train oracle label ')

        save_every = max(1, epochs // desired_evals)
        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        oracle = 'b-loc'
        exp_name = 'exp_3' if not use_ff else 'exp_3-f'

        for regime in regimes.keys():
            print('regime:', regime, 'oracle:', oracle)
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                repetitions=repetitions,
                epochs=epochs,
                train_sets=regimes[regime],
                eval_sets=regimes['direct'],
                oracle_labels=[oracle],
                oracle_is_target=True,
                skip_train=skip_train,
                skip_eval=skip_eval,
                batch_size=batch_size,
                prior_metrics=list(set(prior_metrics + labels)),
                key_param=key_param,
                key_param_value=regime,
                save_every=save_every,
                skip_calc=skip_calc,
                use_ff=use_ff
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)

        print('loading dataframes for final comparison')

        combined_df = load_dataframes(combined_path_list, regimes.keys(), key_param)
        last_epoch_df = load_dataframes(last_path_list, regimes.keys(), key_param)

        create_combined_histogram(last_epoch_df, combined_df, key_param, os.path.join('supervised', exp_name))

        avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats, oracle_stats, delta_sum, delta_x = calculate_statistics(
            combined_df, last_epoch_df, list(set(params + prior_metrics + [key_param])),
            skip_3x=True, key_param=key_param) 

        combined_path = os.path.join('supervised', exp_name, 'c')
        os.makedirs(combined_path, exist_ok=True)
        write_metrics_to_file(os.path.join(combined_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats,
                              key_param=key_param, d_s=delta_sum, d_x=delta_x)
        save_figures(os.path.join(combined_path, 'figs'), combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                     params, last_epoch_df, num=12, key_param_stats=key_param_stats,  oracle_stats=oracle_stats, key_param=key_param, delta_sum=delta_sum)



if __name__ == '__main__':
    experiments([0, 2, 3], repetitions=3, epochs=50, skip_train=False, skip_eval=False, skip_calc=False, batch_size=256, desired_evals=1, use_ff=False)
