import argparse
import os
import multiprocessing
from itertools import product
import pandas as pd

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.experiments_utils import run_hparam_search, do_comparison
from src.pz_envs import ScenarioConfigs
from src.gen_data import gen_data
from supervised_learning_main import run_supervised_session
from tqdm.auto import tqdm


def get_model_types(num, use_eval):

    experiment_2_models = [
                    'a-mix-n-perception-my',
                    'a-mix-n-perception-op',
                    'a-mix-n-treat-my',
                    'a-mix-n-treat-op',
                    'a-mix-n-vision-op',
                    'a-mix-n-belief-op',
                    'a-mix-n-belief-my',
                    'a-mix-n-decision-op',
                    'a-mix-n-decision-my',
                    'a-mix-n-all-my',
                    'a-mix-n-all-op',
                    'a-mix-n-all-split',
                    ]

    # real exp3/4
    experiment_12_models = [
                    'a-mix-n-perception-split',
                    'a-mix-n-perception-shared',
                    'a-mix-n-perception-detach',
                    'a-mix-n-treat-split',
                    'a-mix-n-treat-shared',
                    'a-mix-n-treat-detach',
                    'a-mix-n-belief-split',
                    'a-mix-n-belief-shared',
                    'a-mix-n-belief-detach',
                    'a-mix-n-decision-split',
                    'a-mix-n-decision-shared',
                    'a-mix-n-decision-detach',
                    'a-mix-n-all-split',
                    'a-mix-n-all-shared',
                    'a-mix-n-all-detach',

                    'a-mix-n-perception-shared-v50-b5',
                    'a-mix-n-perception-detach-v50-b5',
                    'a-mix-n-treat-shared-v50-b5',
                    'a-mix-n-treat-detach-v50-b5',
                    'a-mix-n-belief-shared-v50-b5',
                    'a-mix-n-belief-detach-v50-b5',
                    'a-mix-n-decision-shared-v50-b5',
                    'a-mix-n-decision-detach-v50-b5',
                    'a-mix-n-combiner-split-v50-b5',
                    'a-mix-n-combiner-shared-v50-b5',
                    'a-mix-n-combiner-detach-v50-b5',
                    'a-mix-n-all-shared-v50-b5',
                    'a-mix-n-all-detach-v50-b5',
                    ]
    experiment_122_models = [
                    'a-mix-n-perception-split',
                    'a-mix-n-perception-shared',
                    'a-mix-n-perception-detach',
                    'a-mix-n-treat-split',
                    'a-mix-n-treat-shared',
                    'a-mix-n-treat-detach',
                    'a-mix-n-all-split',
                    'a-mix-n-all-shared',
                    'a-mix-n-all-detach',

                    'a-mix-n-perception-shared-v50-b5',
                    'a-mix-n-perception-detach-v50-b5',
                    'a-mix-n-treat-shared-v50-b5',
                    'a-mix-n-treat-detach-v50-b5',
                    'a-mix-n-all-shared-v50-b5',
                    'a-mix-n-all-detach-v50-b5',
                    ]

    # real exp2
    experiment_13_models = [
                    'a-mix-n-treat-my',
                    'a-mix-n-treat-op',
                    #'a-mix-n-vision-my',
                    'a-mix-n-vision-op',
                    #'a-mix-n-presence-my',
                    'a-mix-n-presence-op',
                    'a-mix-n-perception-my',
                    'a-mix-n-perception-op',
                    'a-mix-n-belief-op',
                    'a-mix-n-belief-my',
                    'a-mix-n-decision-op',
                    'a-mix-n-decision-my',
                    'a-mix-n-all-my',
                    'a-mix-n-all-op',]

    experiment_133_models = [
                    'a-mix-n-treat-my',
                    'a-mix-n-treat-op',
                    'a-mix-n-perception-my',
                    'a-mix-n-perception-op',
                    'a-mix-n-all-my',
                    'a-mix-n-all-op',]

    experiment_1_models = [
                    'a-hardcoded',
                    'a-mix-r-treat-my-100',
                    'a-mix-r-treat-op-100',
                    'a-mix-r-vision-op-100',
                    'a-mix-r-presence-op-100',
                    'a-mix-r-belief-op-100',
                    'a-mix-r-belief-my-100',
                    'a-mix-r-decision-op-100',
                    'a-mix-r-decision-my-100',
                    ]

    experiment_3_models = [
                   'a-mix-n-vision-shared',
                   'a-mix-n-perception-shared',
                   'a-mix-n-treat-shared',
                   'a-mix-n-belief-shared',
                   'a-mix-n-decision-shared',
                   'a-mix-n-all-shared',
                   'a-neural-shared',
                   'a-neural-belief-shared',
                   'a-neural-decision-shared',
                   'a-neural-treat-shared',]

    experiment_6_models = [
                   'a-mix-n-treat-detach',
                   'a-mix-n-belief-detach',
                   'a-mix-n-decision-detach',
                   'a-mix-n-all-detach',
                   'a-neural-detach',
                   'a-neural-belief-detach',
                   'a-neural-decision-detach',
                   'a-neural-treat-detach',
                   'a-mix-n-treat-detach-v50-b5',
                   'a-mix-n-belief-detach-v50-b5',
                   'a-mix-n-decision-detach-v50-b5',
                   'a-mix-n-all-detach-v50-b5',
                   'a-neural-detach-v50-b5',
                   'a-mix-n-combiner-detach-v50-b5',
                   #'a-neural-belief-detach-v50-b5',
                   #'a-neural-decision-detach-v50-b5',
                   #'a-neural-treat-detach-v50-b5',
                   ]

    experiment_4_models = [
                   #'a-hardcoded-v50-b5', # remember to eval this!
                   'a-mix-n-treat-shared-v50-b5',
                   'a-mix-n-treat-detach-v50-b5',
                   'a-mix-n-belief-shared-v50-b5',
                   'a-mix-n-decision-shared-v50-b5',
                   'a-mix-n-belief-detach-v50-b5',
                   'a-mix-n-combiner-detach-v50-b5',
                   'a-mix-n-decision-detach-v50-b5',
                   'a-mix-n-combiner-v50-b5',
                   'a-mix-n-combiner-shared-v50-b5',
                   'a-mix-n-all-shared-v50-b5',
                   'a-mix-n-all-detach-v50-b5',
                   'a-neural-shared-v50-b5',]

    if use_eval:
        experiment_4_models.append('a-hardcoded-v50-b5')

    if num == 1:
        return experiment_1_models
    elif num == 2:
        return experiment_2_models
    elif num == 3: 
        return experiment_3_models
    elif num == 4: 
        return experiment_4_models
    elif num == 6: 
        return experiment_6_models
    elif num == 12:
        return experiment_12_models
    elif num == 13: 
        return experiment_13_models
    elif num == 133: 
        return experiment_133_models
    elif num == 122: 
        return experiment_122_models


def init_regimes(rational_train_only=True):
    sub_regime_keys = [
        "Fn", "Nf", "Tn", "Ff", "Tf", "Ft", "Tt", "Gg", "Gn", "Ng"
    ] # removed Nn and Nt

    all_regimes = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-' + x + '1' for x in sub_regime_keys] + ['sl-Nn1', 'sl-Nt1a', 'sl-Nt1b', 'sl-Nn0', 'sl-Nt0'] #['sl-Nn1a', 'sl-Nn1b', 'sl-Nt1a', 'sl-Nt1b', 'sl-Nn0', 'sl-Nt0'] 
    mixed_regimes = {k: ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-Nn1', 'sl-Nt1a', 'sl-Nt1b', 'sl-Nn0', 'sl-Nt0'] + ['sl-' + k + '1'] for k in sub_regime_keys} 

    regimes = {}
    regimes['direct'] = ['sl-' + x + '1' for x in sub_regime_keys]
    regimes['noOpponent'] = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-Nn0', 'sl-Nt0']
    regimes['everything'] = all_regimes
    hregime = {}
    hregime['homogeneous'] = ['sl-Tt0', 'sl-Ff0', 'sl-Nn0', 'sl-Tt1', 'sl-Ff1', 'sl-Nn1']
    # hregime['identity'] = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-Tt1', 'sl-Ff1', 'sl-Nn1']
    sregime = {}
    sregime['special'] = ['sl-Tt0', 'sl-Tt1', 'sl-Nt0', 'sl-Nt1', 'sl-Nf0', 'sl-Nf1', 'sl-Nn0', 'sl-Nn1']

    fregimes = {}
    fregimes['s1'] = regimes['noOpponent']
    fregimes['s2-x'] = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-Tt1', 'sl-Nn0', 'sl-Nt0', 'sl-Gg1']
    fregimes['s2'] = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-Tt1', 'sl-Nn0', 'sl-Nt0']
    #a has no swaps, b has swaps
    fregimes['s21-x'] = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-Tt1', 'sl-Nn0', 'sl-Nt0', 'sl-Nn1a', 'sl-Nt1a', 'sl-Gg1', 'sl-Gn1', 'sl-Ng1'] 
    fregimes['s21'] = ['sl-' + x + '0' for x in sub_regime_keys] + ['sl-Tt1'] + ['sl-Nn0', 'sl-Nt0'] + ['sl-Nn1a'] + ['sl-Nt1a'] 
    fregimes['s3'] = all_regimes
    # fregimes['homogeneous'] = hregime['homogeneous']

    single_regimes = {k[3:]: [k] for k in all_regimes}
    leave_one_out_regimes = {}
    for i in range(len(sub_regime_keys)):
        regime_name = "lo_" + sub_regime_keys[i]
        leave_one_out_regimes[regime_name] = ['sl-' + x + '0' for x in sub_regime_keys]
        ones = ['sl-' + x + '1' for j, x in enumerate(sub_regime_keys) if j != i]
        leave_one_out_regimes[regime_name].extend(ones)

    pref_types = [
        ('same', ''),  # ('different', 'd'), # ('varying', 'v'),
    ]
    role_types = [
        ('subordinate', ''),  # ('dominant', 'D'), # ('varying', 'V'),
    ]

    labels = [
        'id', 'i-informedness',  # must have these or it will break
        'opponents',
        'loc-large',
        'loc-small',
        'target-loc',
        'b-loc-large',
        'b-loc-small',
        'fb-loc', # describes whether actual and belief are different, while belief contains the actual somewhere else
        'fb-exist',
        # 'vision',
        # 'big-box',
        # 'small-box',
        # 'target-box',
        # 'b-box',
        # 'fb-box',
        # 'box-locations'
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


    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap', 'test_regime']

    # shortening for faster iteration
    params = ['test_regime']
    prior_metrics = [ 'correct-loc', 'incorrect-loc',  'informedness', 'opponents']  # 'shouldGetBig', 'shouldAvoidSmall', 'p-b-0', 'p-b-1', 'p-s-0', 'p-s-1', 'delay',

    return regimes, hregime, sregime, fregimes, leave_one_out_regimes, pref_types, role_types, labels, params, prior_metrics

def experiments(todo, repetitions, epochs=50, batches=5000, skip_train=False, skip_calc=False, batch_size=64, desired_evals=5,
                skip_eval=False, skip_activations=False, last_timestep=True, retrain=False, current_model_type=None, current_label=None, current_label_name=None,
                comparison=False, model_types=None, curriculum_name=None):
    """What is the overall performance of naive, off-the-shelf models on this task? Which parameters of competitive
    feeding settings are the most sensitive to overall model performance? To what extent are different models
    sensitive to different parameters? """
    save_every = max(1, epochs // desired_evals)

    regimes, hregime, sregime, fregimes, leave_one_out_regimes, pref_types, role_types, labels, params, prior_metrics = init_regimes()

    model_type = "loc" # or box


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


    

    print('Running experiment')

    combined_path_list = []
    last_path_list = []
    lp_list = []
    key_param = 'regime'
    key_param_list = []
    session_params['oracle_is_target'] = False

    print(current_model_type, model_types)

    if current_model_type != None:
        model_types = [current_model_type]
        label_tuples = [(current_label, current_label_name)]
    elif model_types is not None:
        label_tuples = [('correct-loc', 'loc')]
        if model_types == 'neural':
            model_types = neural_models
        if model_types == 'random':
            model_types = non_neural_models
    else:
        model_types = neural_models
        label_tuples = [('correct-loc', 'loc')]

    if 2 in todo or 13 in todo or 133 in todo:
        real_regimes = {'s21': fregimes['s21'],  's2': fregimes['s2'], 's21-r': fregimes['s21-r'],  's2-r': fregimes['s2-r']}
    elif 30 in todo:
        real_regimes = {'s3': fregimes['s3']}
    elif 20 in todo:
        real_regimes = {'s2': fregimes['s2']}
    elif 21 in todo:
        real_regimes = {'s21': fregimes['s21']}
    elif 10 in todo:
        real_regimes = {'s1': fregimes['s1']}
    else:
        real_regimes = {'s21': fregimes['s21'], 's2': fregimes['s2'], 's1': fregimes['s1']}

    print('eval regimes:', fregimes['s3'])

    for label, label_name in label_tuples:
        for model_type in model_types:
            for regime in list(real_regimes.keys()):
                kpname = f'{model_type}-{label_name}-{regime}'
                print(model_type + '-' + label_name, 'regime:', regime, 'train_sets:', real_regimes[regime])
                combined_paths, last_epoch_paths, lp = run_supervised_session(
                    save_path=os.path.join('supervised', exp_name, kpname) if curriculum_name is None else os.path.join('supervised', exp_name, curriculum_name),
                    train_sets=real_regimes[regime],
                    eval_sets=fregimes['s3'],
                    oracle_labels=[None],
                    key_param=key_param,
                    key_param_value=kpname,
                    label=label,
                    model_type=model_type,
                    do_retrain_model=retrain,
                    train_sets_dict={'s21': fregimes['s21'], 's2': fregimes['s2'], 's1': fregimes['s1'], 's3': fregimes['s3']},
                    curriculum_name=curriculum_name,
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

    else:
        print('no experiment')


def run_single_experiment(args_tuple):
    model_type, (label, label_name), args, curriculum_name = args_tuple
    print(f"Running experiment with model_type: {model_type}, label: {label}, label_name: {label_name}, curriculum: {curriculum_name}")
    print(f"Process: {multiprocessing.current_process().name}")


    experiments([args.exp_num],
                repetitions=5,
                batches=4000,
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
                comparison=args.p,
                curriculum_name=curriculum_name)

def tqdm_pool_map(pool, func, args, total=None):
    total = total or len(args)
    pbar = tqdm(total=total, desc="Experiments")
    
    def update(*args, **kwargs):
        pbar.update()
        return func(*args, **kwargs)
    
    results = pool.map(update, args)
    pbar.close()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments with various options.")

    parser.add_argument('exp_num', type=int, help='which experiment number, 1-4')
    parser.add_argument('-t', action='store_true', help='training')
    parser.add_argument('-e', action='store_true', help='evaluation')
    parser.add_argument('-c', action='store_true', help='calculation')
    parser.add_argument('-a', action='store_true', help='activations')
    parser.add_argument('-r', action='store_true', help='Retrain the model')
    parser.add_argument('-p', action='store_false', help='run in parallel, dont do end')
    parser.add_argument('-g', action='store_true', help='generate dataset')

    args = parser.parse_args()

    exp_num = args.exp_num

    model_types = get_model_types(exp_num, args.e)

    #model_types = ['a-mix-n-treat-split', 'a-mix-n-perception-detach', 'a-mix-n-decision-shared-v50-b5', ]
    #model_types = ['a-mix-n-treat-split',] #['a-mixed-n-belief-my', 'a-mixed-n-belief-op', 'a-mixed-n-decision-my', 'a-mixed-n-decision-op']

    #model_types = ['a-mix-n-perception-op', 'a-mix-n-vision-op', 'a-mix-n-treat-op', 'a-mix-n-belief-op',]
    model_types = ['a-neural-split',]
    #print('number of model types:', len(model_types))

    labels = [('correct-loc', 'loc')]
    curriculum_names = [
        "everything", 
        ]
    base_names = [
        "perception_my", 
        "perception_op",
        "belief_my",
        "belief_op",
        "decision_my",
        "decision_op",
    ]
    single_curriculum_names = [name + "_s21" for name in base_names]
    early_curriculum_names = [name + "_then_all_s21" for name in base_names]
    late_curriculum_names = ["all_then_" + name + "_s21" for name in base_names]
    early_frozen_curriculum_names = [name + "_then_else_s3" for name in base_names]
    late_frozen_curriculum_names = ["else_then_" + name + "_s3" for name in base_names]

    curriculum_names = early_frozen_curriculum_names + late_frozen_curriculum_names

    if (not args.p) and (not args.g):
        experiment_args = [(model_type, label, args, curriculum_name) for model_type, label, curriculum_name in product(model_types, labels, curriculum_names)]

        total_tasks = len(experiment_args)
        
        with multiprocessing.Pool(processes=5) as pool:
            list(tqdm(
                pool.imap(run_single_experiment, experiment_args),
                total=total_tasks,
                desc="Running experiments"
            ))
    else:
        print('running single')
        experiments([args.exp_num] if not args.g else [0],
                repetitions=3,
                batches=2500,
                skip_train=not args.t,
                skip_eval=not args.e,
                skip_calc=not args.c,
                skip_activations=not args.a,
                retrain=args.r,
                batch_size=256,
                desired_evals=1,
                last_timestep=True,
                comparison=args.p,
                model_types=model_types,
                curriculum_name=curriculum_names[0])

    print("finished")


# i both multiplied by weights and also