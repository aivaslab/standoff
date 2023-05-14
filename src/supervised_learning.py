import copy

import numpy as np
import sys
import os

sys.path.append(os.getcwd())

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
            this_ob = np.zeros((10, *obs['player_0'].shape))
            pos = 0

            while True:
                next_obs, rew, done, info = env.step({'player_0': 2})
                this_ob[pos, :, :, :] = next_obs['player_0']
                if not any([np.array_equal(this_ob, x) for x in data_obs]):
                    # if True:
                    data_obs.append(copy.copy(this_ob))
                    for label in labels:
                        data_labels[label].append(info['player_0'][label])
                    tq.update(1)

                pos += 1
                if done['player_0']:
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
    def __init__(self, hidden_size, num_layers, output_len):
        super(RNNModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers, 'output_len': output_len}

        padding1 = 0
        padding2 = 0
        kernel_size1 = 3
        kernel_size2 = 3
        input_size = 17
        stride1 = 1
        stride2 = 1
        pool_kernel_size = 2
        pool_stride = 2

        self.conv1 = nn.Conv2d(6, 8, kernel_size=kernel_size1, padding=padding1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size2, padding=padding2)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=0)
        #self.output_len = output_len

        conv1_output_size = (input_size - kernel_size1 + 2 * padding1) // stride1 + 1
        pool1_output_size = (conv1_output_size - pool_kernel_size) // pool_stride + 1

        conv2_output_size = (pool1_output_size - kernel_size2 + 2 * padding2) // stride2 + 1
        pool2_output_size = (conv2_output_size - pool_kernel_size) // pool_stride + 1

        input_size = 16 * pool2_output_size * pool2_output_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, int(output_len))
     

    def forward(self, x):
        #x = x.view((-1, 10, 6, 17, 17))
        conv_outputs = []
        for t in range(10):
            x_t = x[:, t, :, :, :]
            x_t = self.pool(F.relu(self.conv1(x_t)))
            x_t = self.pool(F.relu(self.conv2(x_t)))
            conv_outputs.append(x_t.view(x.size(0), -1))
        x = torch.stack(conv_outputs, dim=1)

        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Use only the last time step's output
        outputs = self.fc(out)
        return outputs


def train_model(data_name, label, additional_val_sets, path='supervised/'):
    data = np.load(path + data_name + '-obs.npy')
    labels = np.load(path + data_name + '-label-' + label + '.npy')

    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    batch_size = 64
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    additional_val_loaders = []
    for val_set_name in additional_val_sets:
        val_data = np.load(path + val_set_name + '-2500-obs.npy')
        val_labels = np.load(path + val_set_name + '-2500-label-' + label + '.npy')
        val_dataset = CustomDataset(val_data, val_labels)
        additional_val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))

    output_len = np.prod(train_labels.shape[1:])

    input_size = 6 * 17 * 17
    hidden_size = 32
    num_layers = 1
    model = RNNModel(hidden_size, num_layers, output_len)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_val_samples = 500
    num_epochs = 100
    train_losses = []
    val_losses = [[] for _ in range(len(additional_val_loaders) + 1)]
    for epoch in tqdm.trange(num_epochs):
        train_loss = 0
        for i, (inputs, target_labels) in enumerate(train_loader):
            inputs = inputs.view(-1, 10, input_size)
            outputs = model(inputs)
            loss = criterion(outputs, target_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        model.eval()
        for idx, _val_loader in enumerate([val_loader] + additional_val_loaders):
            with torch.no_grad():
                val_loss = 0
                val_samples_processed = 0
                for inputs, labels in _val_loader:
                    inputs = inputs.view(-1, 10, input_size)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_samples_processed += inputs.size(0)
                    if val_samples_processed >= max_val_samples:
                        break
            val_loss /= val_samples_processed/batch_size
            val_losses[idx].append(val_loss)
        model.train()
    # save model
    torch.save([model.kwargs, model.state_dict()], f'{path}{data_name}-{label}-model.pt')

    return train_losses, val_losses


def plot_losses(data_name, label, train_losses, val_losses, val_set_names):
    plt.figure()
    plt.plot(train_losses, label=data_name + ' train loss')
    for val_set_name, val_loss in zip(val_set_names, val_losses):
        plt.plot(val_loss, label=val_set_name + 'val loss')
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig(f'supervised/{data_name}-{label}-losses.png')

# train_model('random-2500', 'exist')
#gen_data(['stage_2', 'all', 'random'], 2500)
if __name__ == '__main__':
    labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target']
    sets = ['stage_2', 'all', 'random']
    for data_name in ['random']:
        unused_sets = [s for s in sets if s != data_name]
        # sum losses
        t_loss_sum = []
        v_loss_sum = []
        first_v_loss = []
        for label in labels:
            t_loss, v_loss, = train_model(data_name + '-2500', label, unused_sets)
            plot_losses(data_name, label, t_loss, v_loss, [data_name] + unused_sets)
            first_v_loss.append(v_loss[0])
            # add losses elementwise
            if len(t_loss_sum) == 0:
                t_loss_sum = t_loss
                v_loss_sum = v_loss
            else:
                t_loss_sum = [x + y for x, y in zip(t_loss_sum, t_loss)]
                v_loss_sum = [x + y for x, y in zip(v_loss_sum, v_loss)]
        # plot sum
        plot_losses(data_name, 'sum', t_loss_sum, v_loss_sum, [data_name])

        plt.figure(figsize=(10, 5))
        for label, loss in zip(labels, first_v_loss):
            plt.plot(loss, label=label)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.ylim(bottom=0)
        plt.savefig(f'supervised/{data_name}-first-losses.png')

