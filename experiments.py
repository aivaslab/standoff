import ast
import copy
import json
import os
import pickle
import traceback

import pandas as pd
import tqdm

from src.pz_envs import ScenarioConfigs
from src.supervised_learning import gen_data
from src.utils.plotting import create_combined_histogram, plot_progression, save_key_param_figures, plot_learning_curves
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
            repetition = int(df_path.split('_')[-1][:1])
            if repetition > 0:
                continue
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
                chunk = chunk.assign(**{key_param: value_name, 'repetition': repetition})
                df_list.append(chunk)
        tq.update(1)
    if len(df_list):
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df['i-informedness'] = combined_df['i-informedness'].fillna('none')
        #print('combined df cols', combined_df.columns)
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



def do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list, used_regimes=None):
    print('loading dataframes for final comparison')

    combined_path = os.path.join('supervised', exp_name, 'c')
    os.makedirs(combined_path, exist_ok=True)
    plot_learning_curves(combined_path, lp_list)

    combined_df = load_dataframes(combined_path_list, key_param_list, key_param)
    last_epoch_df = load_dataframes(last_path_list, key_param_list, key_param)

    if combined_df is not None and last_epoch_df is not None:
        create_combined_histogram(last_epoch_df, combined_df, key_param, os.path.join('supervised', exp_name))

        avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats, oracle_stats, delta_sum, delta_x = calculate_statistics(
            combined_df, last_epoch_df, list(set(params + prior_metrics + [key_param])),
            skip_3x=True, skip_1x=True, key_param=key_param, used_regimes=used_regimes, savepath=os.path.join('supervised', exp_name))


    write_metrics_to_file(os.path.join(combined_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats,
                          key_param=key_param, d_s=delta_sum, d_x=delta_x)
    save_figures(combined_path, combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                 params, last_epoch_df, num=12, key_param_stats=key_param_stats, oracle_stats=oracle_stats,
                 key_param=key_param, delta_sum=delta_sum, delta_x=delta_x)


def experiments(todo, repetitions, epochs=50, batches=5000, skip_train=False, skip_calc=False, batch_size=64, desired_evals=5,
                skip_eval=False, skip_activations=False, last_timestep=True):
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
        "Nn",
        "Fn", "Nf",
        "Tn", "Nt",
        "Ff",
        "Tf", "Ft",
        "Tt"
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
    fregimes['everything'] = regimes['everything']
    fregimes['Tt'] = mixed_regimes['Tt']
    fregimes['homogeneous'] = hregime['homogeneous']

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

    labels = ['loc', 'vision', 'b-loc', 'b-exist', 'exist', 'box-updated',
              'saw-last-update', 'target-loc', 'target-size', 'opponents',
              'informedness', 'swap-treat', 'swap-loc',
              'bait-loc', 'i-informedness', 'i-b-loc',
              'i-b-exist', 'i-target-loc', 'i-target-size',
              "treat-box", "b-treat-box", "i-b-treat-box",
              "target-box", "i-target-box",
              "box-locations", "b-box-locations", "i-b-box-locations",
              "correct-box", 'b-correct-box', 'i-b-correct-box',
              "b-correct-loc", "i-b-correct-loc", 'shouldGetBig'
              ] #'last-vision-span',
    for name in ["loc", "b-loc", "i-b-loc"]:
        labels += ["scalar-" + name, "big-" + name]#, "small-" + name, "any-" + name,]

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

    if 999 in todo:
        print('Running experiment 999: base, different models and answers')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = False

        for label, label_name in [('correct-loc', 'loc')]:#[('correct-loc', 'loc'), ('correct-box', 'box'), ('shouldGetBig', 'size')]:
            for model_type in ['clstm', ]:#['smlp', 'cnn', 'clstm', ]
                for regime in fregimes.keys():
                    kpname = f'{model_type}-{label_name}-{regime}'
                    print(model_type + '-' + label_name, 'regime:', regime, 'train_sets:', fregimes[regime])
                    combined_paths, last_epoch_paths, lp = run_supervised_session(
                        save_path=os.path.join('supervised', exp_name, kpname),
                        train_sets=fregimes[regime],
                        eval_sets=fregimes['everything'],
                        oracle_labels=[None],
                        key_param=key_param,
                        key_param_value=kpname,
                        label=label,
                        model_type=model_type,
                        **session_params
                    )
                    last_path_list.append(last_epoch_paths)
                    combined_path_list.append(combined_paths)
                    key_param_list.append(kpname)
                    lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 101 in todo:
        print('Running experiment 101: base, different models and answers')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = False

        for label, label_name in [('correct-loc', 'loc')]:
            for model_type in ['smlp', 'cnn', 'clstm', ]:
                for regime in ['everything']:
                    kpname = model_type + '-' + label_name
                    print(model_type + '-' + label_name, 'regime:', regime, 'train_sets:', regimes['everything'])
                    combined_paths, last_epoch_paths, lp = run_supervised_session(
                        save_path=os.path.join('supervised', exp_name, kpname),
                        train_sets=regimes['everything'],
                        eval_sets=regimes['everything'],
                        oracle_labels=[None],
                        key_param=key_param,
                        key_param_value=kpname,
                        label=label,
                        model_type=model_type,
                        **session_params
                    )
                    last_path_list.append(last_epoch_paths)
                    combined_path_list.append(combined_paths)
                    key_param_list.append(kpname)
                    lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)


if __name__ == '__main__':
    experiments([999], repetitions=3, batches=10000, skip_train=True, skip_eval=False, skip_calc=True, skip_activations=True,
                batch_size=256, desired_evals=1, last_timestep=True)
