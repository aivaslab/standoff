import copy

import numpy as np
import sys
import os

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
#from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plt

def one_hot(size, data):
    return np.eye(size)[data]

def gen_data(configNames, num_timesteps=2500):
    env_config = {
        "env_class": "StandoffEnv",
        "max_steps": 15,
        "respawn": True,
        "ghost_mode": False,
        "reward_decay": False,
        "width": 9,
        "height": 9,
    }

    player_interface_config = {
        "view_size": 17,
        "view_offset": 4,
        "view_tile_size": 15,
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
    

    for configName in configNames:
        configs = ScenarioConfigs().standoff

        reset_configs = {**configs["defaults"], **configs[configName]}

        if isinstance(reset_configs["num_agents"], list):
            reset_configs["num_agents"] = reset_configs["num_agents"][0]
        if isinstance(reset_configs["num_puppets"], list):
            reset_configs["num_puppets"] = reset_configs["num_puppets"][0]

        env_config['config_name'] = configName
        env_config['agents'] = [GridAgentInterface(**player_interface_config) for _ in range(reset_configs['num_agents'])]
        env_config['puppets'] = [GridAgentInterface(**puppet_interface_config) for _ in range(reset_configs['num_puppets'])]
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
        if hasattr(env, "hard_reset"):
            env.hard_reset(reset_configs)

        labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target']
    
        data_name = f'{configName}-{num_timesteps}'
        data_obs = []
        data_labels = {}
        for label in labels:
            data_labels[label] = []
        tq = tqdm.tqdm(range(int(num_timesteps)))
        while len(data_obs) < num_timesteps:
            obs = env.reset()
            this_ob = np.zeros((10, *obs['p_0'].shape))
            pos = 0

            while True:
                next_obs, rew, done, info = env.step({'p_0': 2})
                this_ob[pos, :, :, :] = next_obs['p_0']
                if not any([np.array_equal(this_ob, x) for x in data_obs]):
                    # if True:
                    data_obs.append(copy.copy(this_ob))
                    for label in labels:
                        if label == "correctSelection":
                            data_labels[label].append(one_hot(5, info['p_0'][label]))
                        else:
                            data_labels[label].append(info['p_0'][label])

                    tq.update(1)

                pos += 1
                if done['p_0']:
                    break
        np.save('supervised/' + data_name + '-obs', np.array(data_obs))
        for label in labels:
            np.save('supervised/' + data_name + '-label-' + label, np.array(data_labels[label]))


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
    def __init__(self, hidden_size, num_layers, output_len, channels):
        super(RNNModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers, 'output_len': output_len, 'channels': channels}

        padding1 = 0
        padding2 = 0
        kernel_size1 = 3
        kernel_size2 = 3
        input_size = 17
        stride1 = 1
        stride2 = 1
        pool_kernel_size = 2
        pool_stride = 2

        self.conv1 = nn.Conv2d(channels, 8, kernel_size=kernel_size1, padding=padding1)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size2, padding=padding2)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=0)
        # self.output_len = output_len

        conv1_output_size = (input_size - kernel_size1 + 2 * padding1) // stride1 + 1
        pool1_output_size = (conv1_output_size - pool_kernel_size) // pool_stride + 1

        # conv2_output_size = (pool1_output_size - kernel_size2 + 2 * padding2) // stride2 + 1
        # pool2_output_size = (conv2_output_size - pool_kernel_size) // pool_stride + 1

        # input_size = 16 * pool2_output_size * pool2_output_size
        input_size = 8 * pool1_output_size * pool1_output_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, int(output_len))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        conv_outputs = []
        for t in range(10):
            x_t = x[:, t, :, :, :]

            x_t = self.pool(F.relu(self.conv1(x_t)))
            # x_t = self.pool(F.relu(self.conv2(x_t)))
            conv_outputs.append(x_t.view(x.size(0), -1))
        x = torch.stack(conv_outputs, dim=1)

        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Use only the last time step's output
        outputs = self.fc(out)
        return outputs


