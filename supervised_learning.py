import copy

import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.rendering import InteractivePlayerWindow
from src.agents import GridAgentInterface
from src.pz_envs import env_from_config
from src.pz_envs.scenario_configs import ScenarioConfigs
from src.utils.conversion import make_env_comp
import pyglet
import gym
#import src.pz_envs
from torch.utils.data import Dataset, DataLoader
import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def gen_data():
    env_config =  {
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
        #"move_type": 1,
        #"view_type": 1,
    }
    configs = ScenarioConfigs().standoff

    configName = 'random'
    reset_configs = {**configs["defaults"],  **configs[configName]}

    if isinstance(reset_configs["num_agents"], list):
        reset_configs["num_agents"] = reset_configs["num_agents"][0]
    if isinstance(reset_configs["num_puppets"], list):
        reset_configs["num_puppets"] = reset_configs["num_puppets"][0]

    env_config['config_name'] = configName
    env_config['agents'] = [GridAgentInterface(**player_interface_config) for _ in range(reset_configs['num_agents'])]
    env_config['puppets'] = [GridAgentInterface(**puppet_interface_config) for _ in range(reset_configs['num_puppets'])]
    #env_config['num_agents'] = reset_configs['num_agents']
    #env_config['num_puppets'] = reset_configs['num_puppets']

    difficulty = 3
    env_config['opponent_visible_decs'] = (difficulty < 1)
    env_config['persistent_treat_images'] = (difficulty < 2)
    env_config['subject_visible_decs'] = (difficulty < 3)
    env_config['gaze_highlighting'] = (difficulty < 3)
    env_config['persistent_gaze_highlighting'] = (difficulty < 2)

    env_name = 'Standoff-S3-' + configName.replace(" ", "") + '-' + str(difficulty) + '-v0'

    env = env_from_config(env_config)
    env.record_supervised_labels = True
    if hasattr(env, "hard_reset"):
        env.hard_reset(reset_configs)

    num_timesteps = int(2500)
    labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target']

    for configName in ['random']:
        data_name  = f'{configName}-{num_timesteps}'
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
                    #if True:
                    data_obs.append(copy.copy(this_ob))
                    for label in labels:
                        data_labels[label].append(info['player_0'][label])
                    tq.update(1)
                #datapoints.append((next_obs['player_0'], info['player_0'][label]))

                obs = next_obs
                pos += 1
                if done['player_0']:
                    break
        np.save('supervised/' + data_name + '-obs', np.array(data_obs))
        for label in labels:
            np.save('supervised/' + data_name + '-label-' + label, np.array(data_labels[label]))
        # np.load('file', mmap_mode='r')
          

def train_model(data_name, label):
    data = np.load('supervised/' + data_name + '-obs.npy')
    labels = np.load('supervised/' + data_name + '-label-' + label + '.npy')
    
    print(data[0].shape)
    
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    
    batch_size = 32
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #output_shape = (batch_size, train_labels.shape[1])
    #print(output_shape)
    output_len = (train_labels.shape[1])
    
    input_size = 6 * 17 * 17
    hidden_size = 32
    num_layers = 1
    model = RNNModel(hidden_size, num_layers, [output_len])
    criterion = nn.MSELoss() # or nn.CrossEntropyLoss() if it's a classification problem
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (inputs, target_labels) in enumerate(train_loader):
            #inputs = inputs.to(device)
            #target_labels = [label.to(device) for label in target_labels]

            # Forward pass
            outputs = model(inputs)
            
            loss = criterion(outputs, target_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_losses.append(train_loss / len(train_loader))
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, labels in val_loader:
                inputs = inputs.view(-1, 10, input_size)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        model.train()
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig(f'{data_name}-{label}-losses.png')

    
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
    def __init__(self, hidden_size, num_layers, output_lens):
        super(RNNModel, self).__init__()

        # Add convolutional layers before the LSTM
        self.conv1 = nn.Conv2d(6, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.output_lens = output_lens

        # Calculate the input size for the LSTM after the convolutions and pooling
        conv_output_height = (17 // 2) // 2
        conv_output_width = (17 // 2) // 2
        input_size = 16 * conv_output_height * conv_output_width


        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_lens[0])
        # Use an adaptive fully connected layer for each output type
        #self.fc_layers = nn.ModuleList([nn.Linear(hidden_size, output_len) for output_len in output_lens])

    def forward(self, x):
        # Split the input tensor into a list of tensors for each time step
        x = x.view(-1, 10, 6, 17, 17)

        # Apply the convolutional layers to each time step separately
        conv_outputs = []
        for t in range(10):  # Assuming 10 time steps
            x_t = x[:, t, :, :, :]
            x_t = self.pool(F.relu(self.conv1(x_t)))
            x_t = self.pool(F.relu(self.conv2(x_t)))
            conv_outputs.append(x_t.view(x.size(0), -1))

        # Stack the processed tensors along the temporal dimension
        x = torch.stack(conv_outputs, dim=1)

        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Use only the last time step's output
        outputs = self.fc(out)

        # Apply the fully connected layers and reshape the outputs to match the desired shapes
        #outputs = [fc(out).view(-1, output_len) for fc, output_len in zip(self.fc_layers, self.output_lens)]
        

        return outputs



train_model('random-2500', 'exist')
#gen_data()
