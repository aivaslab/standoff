import copy
import math

import numpy as np
import sys
import os

import pandas as pd

from .utils.evaluation import get_relative_direction

sys.path.append(os.getcwd())

from .objects import *
from .agents import GridAgentInterface
from .pz_envs import env_from_config
from .pz_envs.scenario_configs import ScenarioConfigs
# import src.pz_envs
from torch.utils.data import Dataset, DataLoader
import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def one_hot(size, data):
    return np.eye(size)[data]


def gen_data(configNames, num_timesteps=2500, labels=[]):
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

    frames = 10
    all_path_infos = pd.DataFrame()
    record_extra_data = False

    for configName in configNames:
        configs = ScenarioConfigs().standoff

        reset_configs = {**configs["defaults"], **configs[configName]}

        if isinstance(reset_configs["num_agents"], list):
            reset_configs["num_agents"] = reset_configs["num_agents"][0]
        if isinstance(reset_configs["num_puppets"], list):
            reset_configs["num_puppets"] = reset_configs["num_puppets"][0]

        env_config['config_name'] = configName
        env_config['agents'] = [GridAgentInterface(**player_interface_config) for _ in
                                range(reset_configs['num_agents'])]
        env_config['puppets'] = [GridAgentInterface(**puppet_interface_config) for _ in
                                 range(reset_configs['num_puppets'])]
        # env_config['num_agents'] = reset_configs['num_agents']
        # env_config['num_puppets'] = reset_configs['num_puppets']

        difficulty = 3
        env_config['opponent_visible_decs'] = (difficulty < 1)
        env_config['persistent_treat_images'] = (difficulty < 2)
        env_config['subject_visible_decs'] = (difficulty < 3)
        env_config['gaze_highlighting'] = (difficulty < 3)
        env_config['persistent_gaze_highlighting'] = (difficulty < 2)

        env = env_from_config(env_config)
        env.record_supervised_labels = True
        env.record_info = True  # used for correctSelection right now
        if hasattr(env, "hard_reset"):
            env.hard_reset(reset_configs)

        # labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target']
        prior_metrics = ['eName', 'shouldAvoidBig', 'shouldAvoidSmall', 'correctSelection', 'incorrectSelection', 'firstBaitReward', 'eventVisibility']
        posterior_metrics = ['selection', 'selectedBig', 'selectedSmall', 'selectedNeither',
                   'selectedPrevBig', 'selectedPrevSmall', 'selectedPrevNeither',
                   'selectedSame', ]

        data_name = f'{configName}-{num_timesteps}'
        data_obs = []
        data_labels = {}
        for label in labels + prior_metrics + posterior_metrics:
            data_labels[label] = []


        all_event_lists = list(ScenarioConfigs.all_event_lists.items())
        all_obs_and_labels = pd.DataFrame(columns=['name', 'obs', 'label'])

        env.target_param_group_count = 2
        env.param_groups = [ {'eLists': {all_event_lists[n][0]: all_event_lists[n][1]}, 'params': ScenarioConfigs.standoff['defaults']} for n in range(len(all_event_lists)) ]
        print(env.param_groups[0])
        #while len(data_obs) < num_timesteps:
        #tq = tqdm.tqdm(range(int(num_timesteps)))
        prev_param_group = -1
        eName = ''
        tq = tqdm.tqdm(range(len(env.param_groups)))

        total_groups = len(env.param_groups)

        while env.current_param_group < total_groups:

            obs = env.reset()
            if env.current_param_group != prev_param_group:
                tq.update(1)
                eName = env.current_event_list_name
            prev_param_group = env.current_param_group
            #print(env.current_param_group, env.event_lists)
            this_ob = np.zeros((10, *obs['p_0'].shape))
            pos = 0

            while pos < frames:
                next_obs, _, _, info = env.step({'p_0': 2})
                this_ob[pos, :, :, :] = next_obs['p_0']
                #if not any([np.array_equal(this_ob, x) for x in data_obs]):
                if pos == frames - 1 or env.has_released:
                    #data_obs.append(copy.copy(this_ob))
                    for label in labels + prior_metrics:
                        if label == "correctSelection" or label == 'incorrectSelection':
                            data = one_hot(5, info['p_0'][label])
                        else:
                            data = info['p_0'][label]
                        all_obs_and_labels = all_obs_and_labels.append({'name': eName, 'obs': copy.copy(this_ob), 'label': data}, ignore_index=True)
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

            if env.current_param_group == total_groups - 1:
                # normally the while loop won't break because reset has a modulus
                break

        all_obs_and_labels.to_csv('train_data.csv', index=False)
        all_path_infos.to_csv('extra_data.csv', index=False)
        #np.save('supervised/' + data_name + '-obs', np.array(data_obs))
        #for label in labels:
        #    np.save('supervised/' + data_name + '-label-' + label, np.array(data_labels[label]))


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Flatten the labels
        flat_labels = torch.tensor(self.labels[idx].flatten(), dtype=torch.float32)
        return torch.tensor(self.data[idx], dtype=torch.float32), flat_labels


class RNNModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False):
        super(RNNModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers,
                       'output_len': output_len, 'pool_kernel_size': pool_kernel_size,
                       'pool_stride': pool_stride, 'channels': channels, 'kernels': kernels,
                       'padding': padding1, 'padding2': padding2, 'pool': use_pool, 'stride': stride1,
                       'use_conv2': use_conv2, 'kernel_size1': kernel_size1,
                       'kernels2': kernels2, 'kernel_size2': kernel_size2}

        input_size = 7

        self.use_pool = use_pool
        self.use_conv2 = use_conv2
        conv1_output_size = (input_size - kernel_size1 + 2 * padding1) // stride1 + 1
        self.conv1 = nn.Conv2d(channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)

        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride,
                                     padding=0 if pool_kernel_size < conv1_output_size else 1)
        # self.output_len = output_len

        pool1_output_size = (
                                conv1_output_size - pool_kernel_size + 2 * 0 if pool_kernel_size < conv1_output_size else 1) // pool_stride + 1 if use_pool else conv1_output_size

        print(self.kwargs, conv1_output_size, pool1_output_size)

        if use_conv2:
            conv2_output_size = (pool1_output_size - min(kernel_size2, pool1_output_size) + 2 * padding2) // stride1 + 1
            self.conv2 = nn.Conv2d(kernels, kernels2, kernel_size=min(pool1_output_size, kernel_size2),
                                   padding=padding2)
            pks = min(pool_kernel_size, conv2_output_size)
            if use_pool:
                self.pool2 = nn.MaxPool2d(kernel_size=pks, stride=pool_stride,
                                          padding=0 if pool_kernel_size < conv2_output_size else 1)
            pool2_output_size = (
                                    conv2_output_size - pks + 2 * 0 if pks < conv2_output_size else 1) // pool_stride + 1 if use_pool else conv2_output_size
        else:
            conv2_output_size = conv1_output_size
            pool2_output_size = pool1_output_size
        print(self.kwargs, conv1_output_size, pool1_output_size, conv2_output_size, pool2_output_size)

        # input_size = 16 * pool2_output_size * pool2_output_size
        input_size = kernels2 * pool2_output_size * pool2_output_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, int(output_len))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        conv_outputs = []
        for t in range(10):
            x_t = x[:, t, :, :, :]

            x_t = self.pool(F.relu(self.conv1(x_t))) if self.use_pool else F.relu(self.conv1(x_t))

            if self.use_conv2:
                x_t = self.pool(F.relu(self.conv2(x_t))) if self.use_pool else F.relu(self.conv2(x_t))
            conv_outputs.append(x_t.view(x.size(0), -1))
        x = torch.stack(conv_outputs, dim=1)

        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Use only the last time step's output
        outputs = self.fc(out)
        return outputs
