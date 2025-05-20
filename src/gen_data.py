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
import torch
from src.supervised_learning import one_hot, serialize_data, identify_mismatches


def gen_data(labels=[], path='supervised', pref_type='', role_type='', record_extra_data=False, prior_metrics=[], conf=None):
    '''
    For all relevant variants, iterates through all possible permutations of environment reset configs and simulates
    up until the release event.
    Records and saves observations, as well as data including labels and metrics.
    '''
    # labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target']
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
        g_count_big = 0
        g_count_small = 0
        total_count = 0
        configs = conf.standoff
        events = conf.stages[configName]['events']
        params = configs[conf.stages[configName]['params']]
        #print(configName, configs, events, params)

        _subject_is_dominant = [False] if role_type == '' else [True] if role_type == 'D' else [True, False]
        _subject_valence = [1] if pref_type == '' else [2] if pref_type == 'd' else [1, 2]
        #print('dom', _subject_is_dominant, 'val', _subject_valence)

        data_name = f'{configName}'
        informedness = data_name[3:-1]
        mapping = {'T': 2, 'F': 1, 'N': 0, 't': 2, 'f': 1, 'n': 0, '0': 0, '1': 1}
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
                env.record_info = True  # used for correct-loc right now

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
                        contains_swaps = "v0" not in eName

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

        this_path = os.path.join(path, data_name)
        os.makedirs(this_path, exist_ok=True)
        np.savez_compressed(os.path.join(this_path, 'obs'), np.array(data_obs))
        np.savez_compressed(os.path.join(this_path, 'params'), np.array(data_params))
        for label in all_labels:
            if len(data_labels[label]) > 0:
                #print(len(data_labels[label]))
                np.savez_compressed(os.path.join(this_path, 'label-' + label), np.array(data_labels[label]))
        
        with_swaps_path = os.path.join(path, data_name + "b")
        os.makedirs(with_swaps_path, exist_ok=True)
        
        with_swaps_indices = np.where(has_swaps)[0]
        if len(with_swaps_indices) > 0:
            #print('with', data_name, len(with_swaps_indices))
            np.savez_compressed(os.path.join(with_swaps_path, 'obs'), np.array([data_obs[i] for i in with_swaps_indices]))
            np.savez_compressed(os.path.join(with_swaps_path, 'params'), np.array([data_params[i] for i in with_swaps_indices]))
            for label in all_labels:
                if len(data_labels[label]) > 0:
                    #print(len(np.array([data_labels[label][i] for i in with_swaps_indices])))
                    np.savez_compressed(os.path.join(with_swaps_path, 'label-' + label), np.array([data_labels[label][i] for i in with_swaps_indices]))
        
        without_swaps_path = os.path.join(path, data_name + "a")
        os.makedirs(without_swaps_path, exist_ok=True)
        
        without_swaps_indices = np.where(~has_swaps)[0]
        if len(without_swaps_indices) > 0:
            #print('without', data_name, len(without_swaps_indices))
            np.savez_compressed(os.path.join(without_swaps_path, 'obs'), np.array([data_obs[i] for i in without_swaps_indices]))
            np.savez_compressed(os.path.join(without_swaps_path, 'params'), np.array([data_params[i] for i in without_swaps_indices]))
            for label in all_labels:
                if len(data_labels[label]) > 0:
                    #print(len(np.array([data_labels[label][i] for i in without_swaps_indices])))
                    np.savez_compressed(os.path.join(without_swaps_path, 'label-' + label), np.array([data_labels[label][i] for i in without_swaps_indices]))