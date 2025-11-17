import hashlib
import pickle
from functools import lru_cache

import h5py
import sys
import os

import pandas as pd

from .utils.conversion import calculate_informedness
from .utils.evaluation import get_relative_direction
from torch.utils.data._utils.collate import default_collate

sys.path.append(os.getcwd())

from .objects import *
from .agents import GridAgentInterface
from .pz_envs import env_from_config
# import src.pz_envs
from torch.utils.data import Dataset
import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torch
from src.supervised_learning import one_hot, serialize_data, identify_mismatches

def save_dataset(base_path, indices, data_obs, data_params, data_labels, all_labels):
    if not indices or len(indices) == 0:
        return
    
    os.makedirs(base_path, exist_ok=True)
    np.savez_compressed(os.path.join(base_path, 'obs'), np.array([data_obs[i] for i in indices]))
    np.savez_compressed(os.path.join(base_path, 'params'), np.array([data_params[i] for i in indices]))
    for label in all_labels:
        if len(data_labels[label]) > 0:
            np.savez_compressed(os.path.join(base_path, 'label-' + label), np.array([data_labels[label][i] for i in indices]))

def modify_informedness_string(informedness_str, replacements):
    new_str = list(informedness_str)
    
    if isinstance(replacements[0], int):
        pos, char = replacements
        if pos < len(new_str):
            new_str[pos] = char
    else:
        for pos, char in replacements:
            if pos < len(new_str):
                new_str[pos] = char
                
    return ''.join(new_str)

def save_all_datasets(path, config_name, data_obs, data_params, data_labels, all_labels, has_swaps):
    prefix_end = 3
    suffix_start = len(config_name) - 1
    
    prefix = config_name[:prefix_end]
    informedness_str = config_name[prefix_end:suffix_start]
    suffix = config_name[suffix_start:]
    
    gettier_big_indices = [i for i, vals in enumerate(data_labels['gettier_big']) if np.any(vals)]
    gettier_small_indices = [i for i, vals in enumerate(data_labels['gettier_small']) if np.any(vals)]
    both_gettier_indices = [i for i in gettier_big_indices if i in gettier_small_indices]
    only_big_indices = [i for i in gettier_big_indices if i not in gettier_small_indices]
    only_small_indices = [i for i in gettier_small_indices if i not in gettier_big_indices]
    
    all_gettier_indices = set(gettier_big_indices + gettier_small_indices)
    non_gettier_indices = [i for i in range(len(data_obs)) if i not in all_gettier_indices]
    
    conditions = [
        (None, non_gettier_indices),
        ((0, 'G'), only_big_indices),
        ((1, 'g'), only_small_indices),
        (((0, 'G'), (1, 'g')), both_gettier_indices)
    ]
    
    for replacements, indices in conditions:
        if not indices:
            continue
        
        if replacements is None:
            modified_config = config_name
        else:
            new_informedness = modify_informedness_string(informedness_str, replacements)
            modified_config = prefix + new_informedness + suffix
        
        main_path = os.path.join(path, modified_config)
        save_dataset_append(main_path, indices, data_obs, data_params, data_labels, all_labels)
        
        with_swaps_indices = [i for i in indices if has_swaps[i]]
        without_swaps_indices = [i for i in indices if not has_swaps[i]]
        
        swaps_path = os.path.join(path, modified_config + "b")
        save_dataset_append(swaps_path, with_swaps_indices, data_obs, data_params, data_labels, all_labels)
        
        no_swaps_path = os.path.join(path, modified_config + "a")
        save_dataset_append(no_swaps_path, without_swaps_indices, data_obs, data_params, data_labels, all_labels)

def save_dataset_append(base_path, indices, data_obs, data_params, data_labels, all_labels):
    if not indices or len(indices) == 0:
        return
    
    os.makedirs(base_path, exist_ok=True)
    
    new_obs = np.array([data_obs[i] for i in indices])
    new_params = np.array([data_params[i] for i in indices])
    new_labels = {}
    for label in all_labels:
        if len(data_labels[label]) > 0:
            new_labels[label] = np.array([data_labels[label][i] for i in indices])
    
    obs_path = os.path.join(base_path, 'obs.npz')
    params_path = os.path.join(base_path, 'params.npz')
    
    if os.path.exists(obs_path):
        existing_obs = np.load(obs_path)['arr_0']
        combined_obs = np.concatenate([existing_obs, new_obs], axis=0)
    else:
        combined_obs = new_obs
    
    if os.path.exists(params_path):
        existing_params = np.load(params_path)['arr_0']
        combined_params = np.concatenate([existing_params, new_params], axis=0)
    else:
        combined_params = new_params
    
    np.savez_compressed(obs_path, combined_obs)
    np.savez_compressed(params_path, combined_params)
    
    for label in all_labels:
        if label in new_labels:
            label_path = os.path.join(base_path, f'label-{label}.npz')
            if os.path.exists(label_path):
                existing_label = np.load(label_path)['arr_0']
                combined_label = np.concatenate([existing_label, new_labels[label]], axis=0)
            else:
                combined_label = new_labels[label]
            np.savez_compressed(label_path, combined_label)


def gen_data(labels=[], path='supervised', pref_type='', role_type='', record_extra_data=False, prior_metrics=[], conf=None):
    '''
    For all relevant variants, iterates through all possible permutations of environment reset configs and simulates
    up until the release event.
    Records and saves observations, as well as data including labels and metrics.
    '''
    # labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target']

    for item in os.listdir(path):
        if item.startswith('sl-') and os.path.isdir(os.path.join(path, item)):
            shutil.rmtree(os.path.join(path, item))

    posterior_metrics = ['selection', 'selectedBig', 'selectedSmall', 'selectedNeither',
                         'selectedPrevBig', 'selectedPrevSmall', 'selectedPrevNeither',
                         'selectedSame', ]

    env_config = {
        "env_class": "MiniStandoffEnv",
        "max_steps": 25,
        "respawn": True,
        "ghost_mode": False,
        "reward_decay": False,
        "width": 9,
        "height": 9,
    }

    player_interface_config = {
        "view_size": 7,
        "view_offset": 0,
        "view_tile_size": 1,
        "observation_style": "rich",
        "see_through_walls": False,
        "color": "yellow",
        "view_type": 0,
        "move_type": 0
    }
    puppet_interface_config = {
        "view_size": 5,
        "view_offset": 3,
        "view_tile_size": 48,
        "observation_style": "rich",
        "see_through_walls": False,
        "color": "red",
        # "move_type": 1,
        # "view_type": 1,
    }

    frames = 4
    all_path_infos = pd.DataFrame()
    suffix = pref_type + role_type

    onehot_labels = ['correct-loc', 'incorrect-loc']
    extra_labels = ['opponents', 'last-vision-span', 'id', 'gettier_big', 'gettier_small'] #we get these manually outside the env

    tq = tqdm.tqdm(range(sum(len(conf.stages[cc]['events']) for cc in conf.stages)))
    # tqdm not currently working with subject dominant and subject valence lists

    unique_id = 0 # this is the id of each datapoint

    for configName in conf.stages:
        print(f"Stage: {configName}, events: {list(conf.stages[configName]['events'].keys())}")
        g_count_big = 0
        g_count_small = 0
        total_count = 0
        configs = conf.standoff
        events = conf.stages[configName]['events']
        params = configs[conf.stages[configName]['params']]
        #if params['num_puppets'] == 0:
        #    continue
        #print(configName, configs, events, params)

        _subject_is_dominant = [False] if role_type == '' else [True] if role_type == 'D' else [True, False]
        _subject_valence = [1] if pref_type == '' else [2] if pref_type == 'd' else [1, 2]
        #print('dom', _subject_is_dominant, 'val', _subject_valence)

        data_name = f'{configName}'
        informedness = data_name[3:-1]
        mapping = {'T': 2, 'F': 1, 'N': 0, 't': 2, 'f': 1, 'n': 0, '0': 0, '1': 1, 'G': 3, 'g': 3}
        # we just get opponents directly from num_puppets later
        informedness = [mapping[char] for char in informedness]

        #print('data name', data_name)
        data_obs = []
        data_labels = {}
        all_labels = list(set(labels + prior_metrics + list(posterior_metrics) + extra_labels))
        for label in all_labels:
            data_labels[label] = []
        data_params = []
        has_swaps = []
        posterior_metrics = set(posterior_metrics)

        
        for subject_is_dominant in _subject_is_dominant:
            for subject_valence in _subject_valence:
                params['subject_is_dominant'] = subject_is_dominant
                params['sub_valence'] = subject_valence

                if isinstance(params["num_agents"], list):
                    params["num_agents"] = params["num_agents"][0]
                if isinstance(params["num_puppets"], list):
                    params["num_puppets"] = params["num_puppets"][0]

                env_config['config_name'] = configName
                env_config['agents'] = [GridAgentInterface(**player_interface_config) for _ in range(params['num_agents'])]
                env_config['puppets'] = [GridAgentInterface(**puppet_interface_config) for _ in range(params['num_puppets'])]

                difficulty = 3
                env_config['opponent_visible_decs'] = (difficulty < 1)
                env_config['persistent_treat_images'] = (difficulty < 2)
                env_config['subject_visible_decs'] = (difficulty < 3)
                env_config['gaze_highlighting'] = (difficulty < 3)
                env_config['persistent_gaze_highlighting'] = (difficulty < 2)
                env_config['conf'] = conf

                env = env_from_config(env_config)
                env.record_oracle_labels = True
                env.record_info = True 

                env.target_param_group_count = 20
                env.param_groups = [ {'eLists': {n: events[n]},
                                      'params': params,
                                      'perms': {n: conf.all_event_permutations[n]},
                                      'delays': {n: conf.all_event_delays[n]}
                                      }
                                     for n in events ]
                #print('first param group', env.param_groups[0])

                prev_param_group = -1

                total_groups = len(env.param_groups)

                env.deterministic = True
                #print('total_groups', total_groups)
                check_labels = [x for x in all_labels if x not in posterior_metrics and x not in extra_labels and x not in onehot_labels]

                counts = {'total': 0, 'big': 0, 'small': 0}

                contains_swaps = False

                while True:
                    env.deterministic_seed = env.current_param_group_pos

                    obs = env.reset()
                    if env.current_param_group != prev_param_group:
                        eName = env.current_event_list_name
                        #contains_swaps = "v0" not in eName
                        print(eName, contains_swaps)
                        contains_swaps = not ("w2v2" in eName or "w1v1" in eName or "w0v0" in eName)

                        tq.update(1)
                    prev_param_group = env.current_param_group
                    this_ob = np.zeros((1 + frames, *obs['p_0'].shape))
                    pos = 0

                    temp_labels = {label: [] for label in check_labels}
                    one_labels = {label: [] for label in onehot_labels}

                    while pos <= frames:
                        obs, _, _, info = env.step({'p_0': 2})
                        this_ob[pos, :, :, :] = obs['p_0']

                        #print(info['p_0'].keys())
                        #print(onehot_labels)

                        for label in check_labels:
                            temp_labels[label].append(info['p_0'][label])
                        for label in onehot_labels:
                            data = one_hot(6, info['p_0'][label]) #changed to 6
                            one_labels[label].append(data)
                            #print(info['p_0'][label])

                        if pos == frames or env.has_released:
                            # Record whether this datapoint has swaps

                            if info['p_0']['gettier_big']:
                                g_count_big += 1
                            if info['p_0']['gettier_small']:
                                g_count_small += 1
                            total_count += 1
                            has_swaps.append(contains_swaps)
                            
                            data_obs.append(serialize_data(this_ob.astype(np.uint8)))
                            data_params.append(eName)
                            for label in check_labels:
                                data_labels[label].append(np.stack(temp_labels[label]))
                            for label in onehot_labels:
                                data_labels[label].append(np.stack(one_labels[label]))

                            informedness_str = data_name[3:-1]
                            data_labels['gettier_big'].append(np.array([info['p_0']['gettier_big'] and 'T' in informedness_str]))
                            data_labels['gettier_small'].append(np.array([info['p_0']['gettier_small'] and 't' in informedness_str]))

                            #if 'b2w2v0fs-0' in str(conf.stages[data_name]['events']) and info['p_0']['gettier_big']:
                            #    print(f"Stage {data_name} b2w2v0fs-0 has events", np.array([info['p_0']['gettier_big']]), np.array([info['p_0']['gettier_small']]))
                            #if 'b2w2v0fs-1' in str(conf.stages[data_name]['events']) and info['p_0']['gettier_big']:
                            #    print(f"Stage {data_name} b2w2v0fs-1 has events", np.array([info['p_0']['gettier_big']]), np.array([info['p_0']['gettier_small']]))

                            if info['p_0']['gettier_big'] and info['p_0']['gettier_small']:
                                print(f"Both gettier case from stage {data_name}, event {eName}")

                            data_labels['opponents'].append(params["num_puppets"])
                            data_labels['id'].append(unique_id)
                            unique_id += 1
                            #print(informedness, params["num_puppets"], info['p_0']['shouldGetBig'], info['p_0']['shouldGetSmall'])
                            #print(env.gettier_big)
                            
                            identify_mismatches(info, env, informedness, params, data_name, configName, eName, info['p_0']['loc'], info['p_0']['b-loc'], counts)

                            break

                        pos += 1

                    if record_extra_data:
                        all_paths = env.get_all_paths(env.grid.volatile, env.instance_from_name['p_0'].pos)

                        for k, path in enumerate(all_paths):

                            _env = copy.deepcopy(env)
                            a = _env.instance_from_name['p_0']
                            while True:
                                _, _, done, info = _env.step({'p_0': get_relative_direction(a, path)})
                                if done['p_0']:
                                    all_path_infos = all_path_infos.append(info['p_0'], ignore_index=True)
                                    break
                            del _env, a

                    if env.current_param_group == total_groups - 1 and env.current_param_group_pos == env.target_param_group_count - 1:
                        # normally the while loop won't break because reset uses a modulus
                        break

                print('regime', data_name, 'counts', counts)

        has_swaps = np.array(has_swaps)

        print(data_name, total_count, g_count_big, g_count_small)

        save_all_datasets(path, data_name, data_obs, data_params, data_labels, all_labels, has_swaps)