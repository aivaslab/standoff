import ast
import copy
import json
import os
import pickle
import traceback

import pandas as pd

from src.pz_envs import ScenarioConfigs
from src.supervised_learning import gen_data
from src.utils.plotting import create_combined_histogram, plot_progression
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

def string_to_list(s):
    """Converts the string format [2 2] to a list [2, 2]."""
    s = s.replace(' ', ',')
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return None

def informedness_to_str(informedness_list):
    mapping = {0: 'N', 1: 'F', 2: 'T'}
    first_val = mapping[informedness_list[0]]
    second_val = mapping[informedness_list[1]].lower()
    return first_val + second_val

def load_dataframes(combined_path_list, value_names, key_param):
    df_list = []

    replace_dict = {'1': 1, '0': 0}
    for df_paths, value_name in zip(combined_path_list, value_names):
        for df_path in df_paths:
            repetition = int(df_path.split('_')[-1][:1])
            #note that this only supports single digits
            chunks = pd.read_csv(df_path, chunksize=10000)
            for chunk in chunks:
                chunk.replace(replace_dict, inplace=True)
                chunk['test_regime'] = chunk.apply(
                    lambda row: informedness_to_str(string_to_list(row['informedness'])) + str(row['opponents']),
                    axis=1
                )
                chunk = chunk.assign(**{key_param: value_name, 'repetition': repetition})
                df_list.append(chunk)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['informedness'] = combined_df['informedness'].fillna('none')
    #print('combined df cols', combined_df.columns)
    return combined_df

def do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, used_regimes=None):
    print('loading dataframes for final comparison')

    combined_df = load_dataframes(combined_path_list, key_param_list, key_param)
    last_epoch_df = load_dataframes(last_path_list, key_param_list, key_param)

    create_combined_histogram(last_epoch_df, combined_df, key_param, os.path.join('supervised', exp_name))

    avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats, oracle_stats, delta_sum, delta_x = calculate_statistics(
        combined_df, last_epoch_df, list(set(params + prior_metrics + [key_param])),
        skip_3x=True, skip_1x=True, key_param=key_param, used_regimes=used_regimes, savepath=os.path.join('supervised', exp_name))

    combined_path = os.path.join('supervised', exp_name, 'c')
    os.makedirs(combined_path, exist_ok=True)
    write_metrics_to_file(os.path.join(combined_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats,
                          key_param=key_param, d_s=delta_sum, d_x=delta_x)
    save_figures(combined_path, combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                 params, last_epoch_df, num=12, key_param_stats=key_param_stats, oracle_stats=oracle_stats,
                 key_param=key_param, delta_sum=delta_sum, delta_x=delta_x)


def experiments(todo, repetitions, epochs, skip_train=False, skip_calc=False, batch_size=64, desired_evals=5,
                use_ff=False, skip_eval=False, skip_activations=False):
    """What is the overall performance of naive, off-the-shelf models on this task? Which parameters of competitive
    feeding settings are the most sensitive to overall model performance? To what extent are different models
    sensitive to different parameters? """
    save_every = max(1, epochs // desired_evals)

    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap', 'test_regime']
    prior_metrics = ['shouldAvoidSmall', 'correctSelection', 'incorrectSelection',
                     'shouldGetBig', 'informedness', 'p-b-0', 'p-b-1', 'p-s-0', 'p-s-1', 'delay', 'opponents']

    sub_regime_keys = [
        "Nn",
        "Fn", "Nf",
        "Tn", "Nt",
        "Ff",
        "Tf", "Ft",
        "Tt"
    ]
    all_regimes = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-' + x + '1' for x in sub_regime_keys]
    mixed_regimes = {k: ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-' + k + '1'] for k in sub_regime_keys}
    #print('regimes:', regimes)
    regimes = {}
    regimes['direct'] = ['sl-' + x + '1' for x in sub_regime_keys]
    regimes['noOpponent'] = ['sl-' + x + '0' for x in sub_regime_keys]
    regimes['everything'] = all_regimes
    regimes['identity'] = ['sl-' + x + '0' for x in sub_regime_keys] + ['Tt1', 'Ff1', 'Nn1']

    single_regimes = {k[3:]: [k] for k in all_regimes}
    leave_one_out_regimes = {}
    for i in range(len(sub_regime_keys)):
        regime_name = "lo_" + sub_regime_keys[i]
        leave_one_out_regimes[regime_name] = ['sl-' + x + '0' for x in sub_regime_keys]
        ones = ['sl-' + x + '1' for j, x in enumerate(sub_regime_keys) if j != i]
        leave_one_out_regimes[regime_name].extend(ones)

    pref_types = [
        ('same', ''),
        # ('different', 'd'),
        # ('varying', 'v'),
    ]
    role_types = [
        ('subordinate', ''),
        # ('dominant', 'D'),
        # ('varying', 'V'),
    ]

    labels = ['loc', 'vision', 'b-loc', 'target']
    oracles = labels + [None]
    oracle_names = [x if x is not None else "None" for x in oracles]
    conf = ScenarioConfigs()
    exp_name = f'exp_{todo[0]}' if not use_ff else f'exp_{todo[0]}-f'
    session_params = {
        'repetitions': repetitions,
        'epochs': epochs,
        'skip_train': skip_train,
        'skip_eval': skip_eval,
        'batch_size': batch_size,
        'prior_metrics': list(set(prior_metrics + labels)),
        'save_every': save_every,
        'skip_calc': skip_calc,
        'use_ff': use_ff,
        'act_label_names': labels,
        'skip_activations': skip_activations,
        'oracle_is_target': False,
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

    if 1 in todo:
        print('Running experiment 1: varied models training directly on the test set')

        combined_path_list = []
        last_path_list = []
        key_param = 'regime'

        for regime in list(regimes.keys()):
            print('regime:', regime)
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)

        do_comparison(combined_path_list, last_path_list, regimes, key_param, exp_name, params, prior_metrics)

    if 51 in todo:
        print('Running experiment 51: single models do not generalize')

        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        key_param_list = []
        used_regimes = []

        for regime in list(single_regimes.keys()):
            print('regime:', regime)
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=single_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            used_regimes.append(single_regimes[regime])

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, used_regimes)

    if 52 in todo:
        print('Running experiment 52: (small version of 51) single models do not generalize')

        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(single_regimes.keys())[:2]:
            print('regime:', regime, 'train_sets:', single_regimes[regime])
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=single_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics)

    if 53 in todo:
        print('Running experiment 53: (small version of 51 on many) multi models do generalize')

        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(single_regimes.keys())[-3:]:
            print('regime:', regime, 'train_sets:', single_regimes[regime])
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=single_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics)

    if 54 in todo:
        print('Running experiment 54: mixed models maybe generalize')

        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(mixed_regimes.keys()):
            print('regime:', regime, 'train_sets:', mixed_regimes[regime])
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=mixed_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics)

    if 55 in todo:
        print('Running experiment 55: mixed models with oracle training maybe generalize')

        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = True

        for regime in ['Nn']:
            print('regime:', regime, 'train_sets:', mixed_regimes[regime])
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=mixed_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=['b-loc'],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics)

    if 56 in todo:
        print('Running experiment 56: odd-one-out')

        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(leave_one_out_regimes.keys()):
            print('regime:', regime, 'train_sets:', leave_one_out_regimes[regime])
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=leave_one_out_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics)

    if 57 in todo:
        print('Running experiment 57: progression')

        key_param = 'regime'
        all_accuracies = {}
        save_file = os.path.join('supervised', exp_name, 'prog.pkl')
        image_file = os.path.join('supervised', exp_name, 'prog.png')

        session_params['skip_calc'] = True
        session_params['skip_activations'] = True

        for progression_trial in range(3):
            base_regimes = ['sl-' + x + '0' for x in sub_regime_keys]
            add_regimes = ['sl-' + x + '1' for x in sub_regime_keys]
            random.shuffle(add_regimes)

            prog_regimes = [[x for x in base_regimes]]
            for idx in range(9):
                prog_regimes.append(copy.copy(prog_regimes[idx]))
                prog_regimes[idx + 1].append(add_regimes.pop())

            for oracle in [0, 1]:
                prog_accuracies = []
                last_path_list = []
                key_param_list = []
                session_params['oracle_is_target'] = bool(oracle)
                for regime in range(10):
                    print('regime:', regime, 'train_sets:', prog_regimes[regime])
                    _, last_epoch_paths = run_supervised_session(
                        save_path=os.path.join('supervised', exp_name, str(oracle) + '_' + str(progression_trial) + '_' + str(regime)),
                        train_sets=prog_regimes[regime],
                        eval_sets=regimes['direct'],
                        oracle_labels=[None if not oracle else 'b-loc'],
                        key_param=key_param,
                        key_param_value=str(regime),
                        **session_params
                    )
                    last_path_list.append(last_epoch_paths)
                    key_param_list.append(str(regime))

                    replace_dict = {'1': 1, '0': 0}
                    #print('last path', last_epoch_paths)
                    df_list = []
                    if len(last_epoch_paths):
                        for df_path in last_epoch_paths:
                            chunks = pd.read_csv(df_path, chunksize=10000)
                            for chunk in chunks:
                                chunk.replace(replace_dict, inplace=True)
                                df_list.append(chunk)
                        combined_df = pd.concat(df_list, ignore_index=True)
                        avg_accuracy = combined_df['accuracy'].mean()
                        print('avg_accuracy', avg_accuracy)
                        prog_accuracies.append(avg_accuracy)

                all_accuracies[str(oracle) + '_' + str(progression_trial)] = prog_accuracies

            with open(save_file, 'wb') as f:
                pickle.dump(all_accuracies, f)

            plot_progression(save_file, image_file)



        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics)



if __name__ == '__main__':
    experiments([51], repetitions=1, epochs=50, skip_train=True, skip_eval=True, skip_calc=True, skip_activations=True,
                batch_size=256, desired_evals=1, use_ff=False)
