import argparse
import os
import multiprocessing
from itertools import product
import pandas as pd

from src.experiments_utils import run_hparam_search, do_comparison
from src.pz_envs import ScenarioConfigs
from src.supervised_learning import gen_data
from supervised_learning_main import run_supervised_session
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



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
            model_types = ['smlp', 'cnn', 'clstm']
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

        #print('comparison time')
        if comparison:
            #print('doing comparison')
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
            model_types = ['a-mixed-n-belief', 'a-mixed-n-output', 'a-neural-split', 'a-neural-shared', 'a-mixed-n-decision'] #'''a-hardcoded','''
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
    parser.add_argument('-p', action='store_false', help='run in parallel, dont do end')
    parser.add_argument('-g', action='store_true', help='generate dataset')

    args = parser.parse_args()

    #model_types = [ 'cnn', 'smlp', 'clstm']
    model_types = 'a-mixed-n-belief', 'a-mixed-n-output', 'a-neural-split', 'a-neural-shared', 'a-mixed-n-decision'
    labels = [('correct-loc', 'loc')]

    if (not args.p) and (not args.g):
        experiment_args = [(model_type, label, args) for model_type, label in product(model_types, labels)]

        with multiprocessing.Pool() as pool:
            pool.map(run_single_experiment, experiment_args)
    else:
        experiments([22] if not args.g else [0],
                repetitions=3,
                batches=500,
                skip_train=not args.t,
                skip_eval=not args.e,
                skip_calc=not args.c,
                skip_activations=not args.a,
                retrain=args.r,
                batch_size=256,
                desired_evals=1,
                last_timestep=True,
                comparison=args.p)

    print("finished")
