import os

import pandas as pd

from src.pz_envs import ScenarioConfigs
from src.supervised_learning import gen_data
from src.utils.plotting import create_combined_histogram
from supervised_learning_main import run_supervised_session, calculate_statistics, write_metrics_to_file, save_figures
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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

def experiments(todo, repetitions, epochs, skip_train=False, skip_calc=False, batch_size=64, desired_evals=5, use_ff=False):
    """What is the overall performance of naive, off-the-shelf models on this task? Which parameters of competitive
    feeding settings are the most sensitive to overall model performance? To what extent are different models
    sensitive to different parameters? """

    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap']
    prior_metrics = ['shouldAvoidSmall', 'correctSelection', 'incorrectSelection',
                     'firstBaitReward', 'shouldGetBig', 'informedness']

    sub_regime_keys = [
        "",
        "eb", "es",
        "eb-lb", "es-ls",
        "eb-es",
        "eb-es-lb", "eb-es-ls",
        "eb-es-lb-ls"
    ]
    old_regimes = [
        ('situational', ['a1']),
        ('informed', ['i0', 'i1']),
        ('contrastive', ['i0', 'u0', 'i1', 'u1']),
        ('complete', ['a0', 'i1']),
        ('comprehensive', ['a0', 'i1', 'u1']),
        ('tiny', ['sl-' + x + '0' for x in ["eb-es"]])
    ]
    sub_regime_keys = [
        "",
        "eb", "es",
        "eb-lb", "es-ls",
        "eb-es",
        "eb-es-lb", "eb-es-ls",
        "eb-es-lb-ls"
    ]
    regimes = {
        'situational': ['sl-' + x + '1' for x in sub_regime_keys],
        'informed': ['sl-' + x + '0' for x in ["eb-es-lb-ls"]] + ['sl-' + x + '1' for x in ["eb-es-lb-ls"]],
        'contrastive': ['sl-' + x + '0' for x in ["eb-es-lb-ls", ""]] + ['sl-' + x + '1' for x in ["eb-es-lb-ls", ""]],
        'complete': ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-' + x + '1' for x in ["eb-es-lb-ls"]],
        'comprehensive': ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-' + x + '1' for x in ["eb-es-lb-ls", ""]],
    }
    default_regime = regimes['complete']
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
    labels = ['loc', 'vision', 'b-loc', 'b-exist', 'target', 'correctSelection']
    oracles = labels + [None]
    oracle_names = [x if x is not None else "None" for x in oracles]
    if 0 in todo:
        print('Generating datasets with labels', labels)
        os.makedirs('supervised', exist_ok=True)
        for pref_type, pref_suffix in pref_types:
            for role_type, role_suffix in role_types:
                gen_data(labels, path='supervised', pref_type=pref_suffix, role_type=role_suffix, prior_metrics=prior_metrics, ScenarioConfigs=ScenarioConfigs)

    if 'h' in todo:
        print('Running hyperparameter search on all regimes, pref_types, role_types')


    # Experiment 1
    if 1 in todo:
        print('Running experiment 1: varied models training directly on the test set')

        # todo: add hparam search for many models, comparison between them?
        save_every = max(1, epochs // desired_evals)
        combined_path_list = []
        last_path_list = []
        key_param = 'regime'
        exp_name = 'exp_1' if not use_ff else 'exp_1-f'

        for regime in regimes.keys():
            print('regime:', regime)
            combined_paths, last_epoch_paths = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                repetitions=repetitions,
                epochs=epochs,
                train_sets=regimes[regime],
                eval_sets=regimes['situational'],
                oracle_labels=[None],
                skip_train=skip_train,
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
        # todo: add specific cell plots here

        avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats = calculate_statistics(
            combined_df, last_epoch_df, list(set(params + prior_metrics + [key_param])),
            skip_3x=True, key_param=key_param)  # todo: make it definitely save one fixed param eg oracle

        combined_path = os.path.join('supervised', exp_name, 'c')
        os.makedirs(combined_path, exist_ok=True)
        write_metrics_to_file(os.path.join(combined_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats, key_param=key_param)
        save_figures(os.path.join(combined_path, 'figs'), combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                     params, last_epoch_df, num=12, key_param_stats=key_param_stats, key_param=key_param)

    if 2 in todo:
        save_every = max(1, epochs // desired_evals)
        print('Running experiment 2: varied oracle modules, saving every', save_every)
        combined_path_list = []
        last_path_list = []
        key_param = 'oracle'
        oracle_layer = 0
        exp_name = 'exp_2' if not use_ff else 'exp_2-f'
        if oracle_layer != 0:
            exp_name = exp_name + str(oracle_layer)
        #os.makedirs(os.path.join('supervised', 'exp_2'), exist_ok=True)

        for single_oracle, oracle_name in zip(oracles, oracle_names):
            print('oracle:', single_oracle)
            combined_paths, last_epoch_paths = run_supervised_session(save_path=os.path.join('supervised', exp_name, oracle_name),
                                                repetitions=repetitions,
                                                epochs=epochs,
                                                train_sets=regimes['complete'],
                                                eval_sets=regimes['situational'],
                                                oracle_labels=[single_oracle],
                                                skip_train=skip_train,
                                                batch_size=batch_size,
                                                prior_metrics=list(set(prior_metrics+labels)),
                                                key_param=key_param,
                                                key_param_value=oracle_name,
                                                save_every=save_every,
                                                skip_calc=skip_calc,
                                                use_ff=use_ff,
                                                oracle_layer=oracle_layer,
                                                )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)

        print('loading dataframes for final comparison')

        combined_df = load_dataframes(combined_path_list, oracle_names, key_param)
        last_epoch_df = load_dataframes(last_path_list, oracle_names, key_param)

        create_combined_histogram(last_epoch_df, combined_df, key_param, os.path.join('supervised', exp_name))
        # todo: add specific cell plots here

        avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats = calculate_statistics(
            combined_df, last_epoch_df, list(set(params + prior_metrics + [key_param])), skip_3x=True, key_param=key_param) #todo: make it definitely save one fixed param eg oracle


        combined_path = os.path.join('supervised', exp_name, 'c')
        os.makedirs(combined_path, exist_ok=True)
        write_metrics_to_file(os.path.join(combined_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats, key_param=key_param)
        save_figures(os.path.join(combined_path, 'figs'), combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                     params, last_epoch_df, num=12, key_param_stats=key_param_stats, key_param=key_param)



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
                               train_sets=regimes['situational'])


if __name__ == '__main__':
    experiments([2], 1, 50, skip_train=False, skip_calc=False, batch_size=256, desired_evals=1, use_ff=False)
