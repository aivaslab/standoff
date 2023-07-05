
import itertools

import sys
import os

import pandas as pd
import heapq

from scipy.stats import sem, t

sys.path.append(os.getcwd())

from src.objects import *
from torch.utils.data import DataLoader
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

    # Calculate conditions for special parameters
    swaps_gt_0 = swaps > 0
    first_swap_is_both_false = not first_swap_is_both
    delay_2nd_bait_false = not delay_2nd_bait
    swaps_eq_2 = swaps == 2
    visible_baits_eq_1 = visible_baits == 1
    visible_swaps_eq_1 = visible_swaps == 1

    return pd.Series({
        "visible_baits": visible_baits,
        "swaps": swaps,
        "visible_swaps": visible_swaps,
        "first_swap_is_both": first_swap_is_both if swaps_gt_0 else 'N/A (swaps==0)',
        "second_swap_to_first_loc": second_swap_to_first_loc if swaps_eq_2 and delay_2nd_bait_false else 'N/A (swaps<2 or 2nd bait delayed)',
        "delay_2nd_bait": delay_2nd_bait if swaps_gt_0 and first_swap_is_both_false else 'N/A (swaps=0 or first swap both)',
        "first_bait_size": first_bait_size,
        "uninformed_bait": uninformed_bait if visible_baits_eq_1 else 'N/A (informed baits!=1)',
        "uninformed_swap": uninformed_swap if swaps_eq_2 and visible_swaps_eq_1 else 'N/A (swaps<2 or informed swaps!=1)',
        "first_swap": first_swap if swaps_gt_0 and not delay_2nd_bait and not first_swap_is_both else 'N/A (swaps=0 or 2nd bait delayed or first swap both)'
    })

def train_model(data_name, label, additional_val_sets, path='supervised/', dsize=2500, epochs=100, model_kwargs=None, lr=0.001):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    #pd.DataFrame(param_losses_list).to_csv(path + 'param_losses.csv')
    #df = pd.read_csv('supervised/param_losses.csv')
    # Apply the decoding function to each row

    df = pd.DataFrame(param_losses_list)

    df = df.join(df['param'].apply(decode_event_name))



    return train_losses, val_losses, df


def calculate_ci(group):
    confidence = 0.95
    group_arr = group.values
    n = group_arr.shape[0]
    m = np.mean(group_arr)
    std_err = sem(group_arr)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    return pd.DataFrame({'lower': [m - h], 'upper': [m + h], 'mean': [m]},
                         columns=['lower', 'upper', 'mean'])


def calculate_statistics(df, params):
    avg_loss = {}
    variances = {}
    ranges = {}
    range_dict = {}
    range_dict3 = {}

    last_epoch_df = df[df['epoch'] == df['epoch'].max()]
    param_pairs = itertools.combinations(params, 2)
    param_triples = itertools.combinations(params, 3)

    for param in params:
        ci_df = df.groupby([param, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
        avg_loss[param] = ci_df

    for param1, param2 in param_pairs:
        ci_df = df.groupby([param1, param2, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
        avg_loss[(param1, param2)] = ci_df

        means = last_epoch_df.groupby([param1, param2]).mean(numeric_only=True)
        #variances[(param1, param2)] = grouped.var().mean()
        ranges[(param1, param2)] = means['accuracy'].max() - means['accuracy'].min()

        for value1 in df[param1].unique():
            subset = last_epoch_df[last_epoch_df[param1] == value1]
            new_means = subset.groupby(param2)['accuracy'].mean()
            range_dict[(param1, value1, param2)] = new_means.max() - new_means.min()

    for param1, param2, param3 in param_triples:
        ci_df = df.groupby([param1, param2, param3, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
        avg_loss[(param1, param2)] = ci_df

        means = last_epoch_df.groupby([param1, param2, param3]).mean(numeric_only=True)
        #variances[(param1, param2)] = grouped.var().mean()
        #ranges[(param1, param2, param3)] = means.max() - means.min()

        for value1 in df[param1].unique():
            subset = last_epoch_df[last_epoch_df[param1] == value1]
            for value2 in df[param2].unique():
                subsubset = subset[subset[param2] == value2]
                new_means = subsubset.groupby(param3)['accuracy'].mean()
                range_dict3[(param1, value1, param2, value2, param3)] = new_means.max() - new_means.min()

    return avg_loss, variances, ranges, range_dict, range_dict3

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


def save_figures(df, avg_loss, ranges, range_dict, range_dict3, params, num=5):
    top_pairs = sorted(ranges.items(), key=lambda x: x[1], reverse=True)[:num]
    top_n_ranges = heapq.nlargest(num, range_dict, key=range_dict.get)
    top_n_ranges3 = heapq.nlargest(num, range_dict3, key=range_dict3.get)

    save_double_param_figures(top_pairs, df, avg_loss)
    save_single_param_figures(params, df, avg_loss)
    save_fixed_double_param_figures(top_n_ranges, df, avg_loss)
    save_fixed_triple_param_figures(top_n_ranges3, df, avg_loss)

def save_double_param_figures(top_pairs, df, avg_loss):
    os.makedirs('supervised/doubleparams', exist_ok=True)
    for (param1, param2), _ in top_pairs:
        plt.figure(figsize=(10, 6))
        for value1 in df[param1].unique():
            for value2 in df[param2].unique():
                sub_df = avg_loss[(param1, param2)][
                    (avg_loss[(param1, param2)][param1] == value1) & (
                                avg_loss[(param1, param2)][param2] == value2)]

                str1 = f'{param1} = {value1}' if not isinstance(value1, str) or value1[0:3] != "N/A" else value1
                str2 = f'{param2} = {value2}' if not isinstance(value2, str) or value2[0:3] != "N/A" else value2
                plt.plot(sub_df['epoch'], sub_df['mean'],
                         label=f'{str1}, {str2}')
                plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)

        plt.title(f'Average accuracy vs Epoch for {param1} and {param2}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(os.path.join(os.getcwd(), f'supervised/doubleparams/{param1}{param2}.png'))
        plt.close()

def save_single_param_figures(params, df, avg_loss):
    os.makedirs('supervised/singleparams', exist_ok=True)
    for param in params:
        plt.figure(figsize=(10, 6))
        for value in df[param].unique():
            sub_df = avg_loss[param][avg_loss[param][param] == value]

            plt.plot(sub_df['epoch'], sub_df['mean'], label=f'{param} = {value}' if not isinstance(value, str) or value[0:3] != "N/A" else value)
            plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
        plt.title(f'Average accuracy vs Epoch for {param}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        file_path = os.path.join(os.getcwd(), 'supervised', 'singleparams', f'{param}.png')
        plt.savefig(file_path)
        plt.close()

def save_fixed_double_param_figures(top_n_ranges, df, avg_loss):
    os.makedirs('supervised/fixeddoubleparams', exist_ok=True)
    for combo in top_n_ranges:
        param1, value1, param2 = combo
        subset = df[df[param1] == value1]
        plt.figure(figsize=(10, 6))
        for value2 in subset[param2].unique():
            sub_df = avg_loss[param2][avg_loss[param2][param2] == value2]
            plt.plot(sub_df['epoch'], sub_df['mean'],
                     label=f'{param2} = {value2}' if not isinstance(value2, str) or value2[0:3] != "N/A" else value2)
            plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
        plt.title(f'Average accuracy vs Epoch for {param2} given {param1} = {value1}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        name = f'{param1}{str(value1)[:3]}{param2}'.replace('/', '-')
        plt.savefig(os.path.join(os.getcwd(), f'supervised/fixeddoubleparams/{name}.png'))
        plt.close()

def save_fixed_triple_param_figures(top_n_ranges, df, avg_loss):
    os.makedirs('supervised/fixedtripleparams', exist_ok=True)
    for combo in top_n_ranges:
        param1, value1, param2, value2, param3 = combo
        subset = df[(df[param1] == value1) & (df[param2] == value2)]
        plt.figure(figsize=(10, 6))
        for value3 in subset[param3].unique():
            sub_df = avg_loss[param3][avg_loss[param3][param3] == value3]
            plt.plot(sub_df['epoch'], sub_df['mean'],
                     label=f'{param3} = {value3}' if not isinstance(value3, str) or value3[0:3] != "N/A" else value3)
            plt.fill_between(sub_df['epoch'], sub_df['lower'], sub_df['upper'], alpha=0.2)
        plt.title(f'Average accuracy vs Epoch for {param3} given {param1}={value1} and {param2}={value2}')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')
        plt.legend()
        plt.ylim(0, 1)
        name = f'{param1}{str(value1)[:3]}{param2}{str(value2)[:3]}{param3}'.replace('/', '-')
        plt.savefig(os.path.join(os.getcwd(), f'supervised/fixedtripleparams/{name}.png'))
        plt.close()

# train_model('random-2500', 'exist')
if __name__ == '__main__':
    sets = ['296']
    dsize = 6000
    labels = ['correctSelection']
    gen_data(sets, dsize, labels)
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


    test_names = []

    num_random_tests = 48
    repetitions = 1
    epochs = 10
    colors = plt.cm.jet(np.linspace(0,1,num_random_tests))
    lr = 0.003

    test = 0
    while test < num_random_tests:
        try:
            model_kwargs = {x: random.choice(model_kwargs_base[x]) for x in model_kwargs_base.keys()}
            model_name = "".join([str(x) + "," for x in model_kwargs.values()])

            unused_sets = [s for s in sets if s != data_name]
            # sum losses
            t_loss_sum, v_loss_sum, first_v_loss = [], [], []
            for label in labels:
                df_list = []
                for repetition in range(repetitions):
                    t_loss, v_loss, df = train_model(data_name, label, unused_sets, epochs=epochs, dsize=dsize, model_kwargs=model_kwargs, lr=lr)
                    df_list.append(df)
                    #plot_losses(data_name, label, t_loss, v_loss, [data_name] + unused_sets)
                    first_v_loss.append(v_loss[0])
                    if len(t_loss_sum) == 0:
                        t_loss_sum = t_loss
                        v_loss_sum = v_loss
                    else:
                        t_loss_sum = [x + y for x, y in zip(t_loss_sum, t_loss)]
                        v_loss_sum = [x + y for x, y in zip(v_loss_sum, v_loss)]

                combined_df = pd.concat(df_list, ignore_index=True)
                params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
                          'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
                          'uninformed_bait', 'uninformed_swap', 'first_swap']

                avg_loss, variances, ranges, range_dict, range_dict3 = calculate_statistics(combined_df, params)

                save_figures(combined_df, avg_loss, ranges, range_dict, range_dict3, params, num=5)

            test_names.append(model_name)
            test += 1
        except BaseException as e:
            print(e)
            traceback.print_exc()
