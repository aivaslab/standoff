import copy

import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from src.objects import *
from src.agents import GridAgentInterface
from src.pz_envs import env_from_config
from src.pz_envs.scenario_configs import ScenarioConfigs
# import src.pz_envs
from torch.utils.data import Dataset, DataLoader
import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from src.supervised_learning import RNNModel, CustomDataset, gen_data


def train_model(data_name, label, additional_val_sets, path='supervised/', epochs=100):
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
    channels = np.prod(train_data.shape[1])

    input_size = channels * 17 * 17
    hidden_size = 16
    num_layers = 1
    model = RNNModel(hidden_size, num_layers, output_len, channels)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_val_samples = 500
    num_epochs = epochs
    train_losses = []
    val_losses = [[] for _ in range(len(additional_val_loaders) + 1)]
    for epoch in tqdm.trange(num_epochs):
        train_loss = 0
        for i, (inputs, target_labels) in enumerate(train_loader):
            #inputs = inputs.view(-1, 10, input_size)
            #print(inputs.shape)
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
                    #inputs = inputs.view(-1, 10, input_size)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_samples_processed += inputs.size(0)
                    if val_samples_processed >= max_val_samples:
                        break
            val_loss /= val_samples_processed / batch_size
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
if __name__ == '__main__':
    sets = ScenarioConfigs.env_groups['3'] + ['stage_2', 'all', 'random']
    dsize = 2000
    gen_data(sets, 2000)
    #labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target', 'correctSelection']
    labels = ['correctSelection']
    for data_name in sets:
        unused_sets = [s for s in sets if s != data_name]
        # sum losses
        t_loss_sum = []
        v_loss_sum = []
        first_v_loss = []
        for label in labels:
            t_loss, v_loss, = train_model(data_name + '-' + str(dsize), label, unused_sets, epochs=20)
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
