import copy

import numpy as np
import sys
import os

import pandas as pd

sys.path.append(os.getcwd())

from src.objects import *
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch.nn as nn
import random
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from src.supervised_learning import RNNModel, CustomDataset, gen_data
import traceback


def decode_event_name(name):
    # Split the name into the main part and the numerical suffix
    main_part, numerical_suffix = name.split('-')

    # Extract individual parameters
    visible_baits = int(main_part[main_part.index('b') + 1:main_part.index('w')])
    swaps = int(main_part[main_part.index('w') + 1:main_part.index('v')])
    visible_swaps = int(main_part[main_part.index('v') + 1])
    first_swap_is_both = 1 if 'f' in main_part else 0
    second_swap_to_first_loc = 1 if 's' in main_part else 0
    delay_2nd_bait = 1 if 'd' in main_part else 0

    # Convert the numerical suffix to binary and pad with zeroes to ensure 4 bits
    binary_suffix = format(int(numerical_suffix), '04b')

    # Extract parameters from the binary suffix
    first_bait_size = int(binary_suffix[0])
    uninformed_bait = int(binary_suffix[1])
    uninformed_swap = int(binary_suffix[2])
    first_swap = int(binary_suffix[3])

    return pd.Series({
        "visible_baits": visible_baits,
        "swaps": swaps,
        "visible_swaps": visible_swaps,
        "first_swap_is_both": first_swap_is_both,
        "second_swap_to_first_loc": second_swap_to_first_loc,
        "delay_2nd_bait": delay_2nd_bait,
        "first_bait_size": first_bait_size,
        "uninformed_bait": uninformed_bait,
        "uninformed_swap": uninformed_swap,
        "first_swap": first_swap
    })

def train_model(data_name, label, additional_val_sets, path='supervised/', dsize=2500, epochs=100, model_kwargs=None):
    data = np.load(path + data_name + '-obs.npy')
    labels = np.load(path + data_name + '-label-' + label + '.npy')
    params = np.load(path + data_name + '-params.npy')

    train_data, val_data, train_labels, val_labels, train_params, val_params = train_test_split(
        data, labels, params, test_size=0.2, random_state=42
    )

    batch_size = 64
    train_dataset = CustomDataset(train_data, train_labels, train_params)
    val_dataset = CustomDataset(val_data, val_labels, val_params)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    additional_val_loaders = []
    for val_set_name in additional_val_sets:
        val_data = np.load(path + val_set_name + '-obs.npy')
        val_labels = np.load(path + val_set_name + '-label-' + label + '.npy')
        val_labels = np.load(path + val_set_name + '-params.npy')
        val_dataset = CustomDataset(val_data, val_labels, val_params)
        additional_val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))

    model_kwargs['output_len'] = np.prod(train_labels.shape[1:])
    model_kwargs['channels'] = np.prod(train_data.shape[2])

    model = RNNModel(**model_kwargs)
    criterion = nn.CrossEntropyLoss() #nn.MSELoss()
    special_criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_val_samples = 500
    num_epochs = epochs
    train_losses = []
    val_losses = [[] for _ in range(len(additional_val_loaders) + 1)]

    #param_losses = pd.DataFrame(columns=['param', 'epoch', 'loss'])
    param_losses_list = []
    for epoch in tqdm.trange(num_epochs):
        train_loss = 0
        for i, (inputs, target_labels, _) in enumerate(train_loader):
            #inputs = inputs.view(-1, 10, input_size)
            #print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(target_labels, dim=1))
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
                for inputs, labels, params in _val_loader:
                    #inputs = inputs.view(-1, 10, input_size)
                    outputs = model(inputs)
                    loss = criterion(outputs, torch.argmax(labels, dim=1))
                    val_loss += loss.item()
                    val_samples_processed += inputs.size(0)
                    #if val_samples_processed >= max_val_samples:
                    #    break

                # broken part:
                if True:
                    for inputs, labels, params in _val_loader:
                        outputs = model(inputs)
                        losses = special_criterion(outputs, torch.argmax(labels, dim=1))
                        _, predicted = torch.max(outputs, 1)
                        corrects = (predicted == torch.argmax(labels, dim=1)).float()
                        for input, label, param, loss, correct in zip(inputs, labels, params, losses, corrects):
                            param_losses_list.append(
                                {'param': param, 'epoch': epoch, 'loss': loss.item(), 'accuracy': correct.item()})
                            #param_losses = param_losses.append({'param': param, 'epoch': epoch, 'loss': loss.item(), 'accuracy': accuracy}, ignore_index=True)


            val_loss /= val_samples_processed / batch_size
            val_losses[idx].append(val_loss)
        model.train()
    # save model
    torch.save([model.kwargs, model.state_dict()], f'{path}{data_name}-{label}-model.pt')

    pd.DataFrame(param_losses_list).to_csv(path + 'param_losses.csv')


    df = pd.read_csv('supervised/param_losses.csv')
    # Apply the decoding function to each row
    df = df.join(df['param'].apply(decode_event_name))

    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap']

    avg_loss = {}
    entropy = {}
    std_dev = {}

    df['log_loss'] = np.log(df['loss'])
    # Calculate average loss and entropy for each parameter individually
    for param in params:
        avg_loss[param] = df.groupby([param, 'epoch'])['accuracy'].mean().reset_index()
        std_dev[param] = df.groupby([param, 'epoch'])['accuracy'].std().reset_index()
        entropy[param] = df.groupby([param, 'epoch'])['log_loss'].apply(
            lambda x: -np.sum(x * df.loc[x.index, 'loss'])).reset_index()

    # Now, you can plot the curves for each parameter
    for param in params:
        plt.figure(figsize=(10, 6))
        for value in df[param].unique():
            sub_df = avg_loss[param][avg_loss[param][param] == value]
            std_dev_sub_df = std_dev[param][std_dev[param][param] == value]
            plt.plot(sub_df['epoch'], sub_df['accuracy'], label=f'{param} = {value}')
            plt.fill_between(sub_df['epoch'], sub_df['accuracy'] - std_dev_sub_df['accuracy'],
                             sub_df['accuracy'] + std_dev_sub_df['accuracy'], alpha=0.2)
            plt.title(f'Average accuracy vs Epoch for {param}')
            plt.xlabel('Epoch')
            plt.ylabel('Average accuracy')
            plt.legend()
        plt.show()

    return train_losses, val_losses


def plot_losses(data_name, label, train_losses, val_losses, val_set_names, specific_name=None):
    plt.figure()
    plt.plot(train_losses, label=data_name + ' train loss')
    for val_set_name, val_loss in zip(val_set_names, val_losses):
        if specific_name == None or val_set_name == specific_name:
            plt.plot(val_loss, label=val_set_name + ' val loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f'supervised/{data_name}-{label}-losses.png')


# train_model('random-2500', 'exist')
if __name__ == '__main__':
    #sets = ScenarioConfigs.env_groups['3'] + ['all'] #use stage_2 and random for the other thing
    sets = ['296']
    dsize = 6000
    labels = ['correctSelection']
    #gen_data(sets, dsize, labels)
    #exit()
    #labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target', 'correctSelection']

    data_name = '296'
    model_kwargs_base = {'hidden_size': [6, 8, 12, 16, 32],
                    'num_layers': [1, 2, 3],
                    'kernels': [4, 8, 16, 24, 32],
                    'kernel_size1': [1, 3, 5],
                    'kernel_size2': [1, 3, 5],
                    'stride1': [1, 2],
                    'pool_kernel_size': [2, 3],
                    'pool_stride': [1, 2],
                    'padding1': [0, 1],
                    'padding2': [0, 1],
                    'use_pool': [True, False],
                    'use_conv2': [True, False],
                    'kernels2': [8, 16, 32, 48],
                    }


    test_losses = {}
    test_loss_means = {}
    test_loss_stds = {}
    test_names = []

    num_random_tests = 48
    repetitions = 3
    epochs = 100
    colors = plt.cm.jet(np.linspace(0,1,num_random_tests))

    test = 0
    while test < num_random_tests:

        try:
            model_kwargs = {x: random.choice(model_kwargs_base[x]) for x in model_kwargs_base.keys()}
            model_name = "".join([str(x) + "," for x in model_kwargs.values()])
            test_losses[model_name] = [[] for _ in range(repetitions)]
            test_loss_means[model_name] = []
            test_loss_stds[model_name] = []

            unused_sets = [s for s in sets if s != data_name]
            # sum losses
            t_loss_sum = []
            v_loss_sum = []
            first_v_loss = []
            for label in labels:
                for repetition in range(repetitions):
                    t_loss, v_loss, = train_model(data_name, label, unused_sets, epochs=epochs, dsize=dsize, model_kwargs=model_kwargs)
                    #plot_losses(data_name, label, t_loss, v_loss, [data_name] + unused_sets)
                    first_v_loss.append(v_loss[0])
                    # add losses elementwise
                    if len(t_loss_sum) == 0:
                        t_loss_sum = t_loss
                        v_loss_sum = v_loss
                    else:
                        t_loss_sum = [x + y for x, y in zip(t_loss_sum, t_loss)]
                        v_loss_sum = [x + y for x, y in zip(v_loss_sum, v_loss)]


                    test_losses[model_name][repetition] = v_loss[0]
            test_loss_means[model_name] = np.asarray(np.mean(test_losses[model_name], axis=0))
            test_loss_stds[model_name] = np.asarray(np.std(test_losses[model_name], axis=0))

            if not len(test_loss_stds[model_name]):
                continue
            test_names.append(model_name)

            plt.figure(figsize=(20, 10))
            last_items = []
            for k, model_name in enumerate(test_names):
                plt.plot(test_loss_means[model_name], label=model_name, c=colors[k])
                plt.fill_between(np.arange(len(test_loss_means[model_name])), test_loss_means[model_name] - np.asarray(test_loss_stds[model_name]),
                                 test_loss_means[model_name] + test_loss_stds[model_name], alpha=.1, color=colors[k])
                last_items.append((model_name, test_loss_means[model_name][-1]))
            last_items_sorted = sorted(range(len(last_items)), key=lambda i: 0-last_items[i][1])
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            handles, _labels = plt.gca().get_legend_handles_labels()
            handles = [handles[i] for i in last_items_sorted]
            _labels = [_labels[i] for i in last_items_sorted]
            plt.legend(handles, _labels, loc='center left', bbox_to_anchor=(1, 1))
            plt.ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(f'supervised/test-losses.png')

            # plot sum
            '''plot_losses(data_name, 'sum', t_loss_sum, v_loss_sum, [data_name])
    
            plt.figure(figsize=(20, 10))
            for label, loss in zip(labels, first_v_loss):
                plt.plot(loss, label=label)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.legend()
            plt.ylim(bottom=0)
            plt.savefig(f'supervised/{data_name}-first-losses.png')'''
            test += 1
        except BaseException as e:
            print(e)
            traceback.print_exc()
