import ast
import glob
import hashlib
import itertools
import math
import pickle
import re
#from torch.autograd import profiler


import sys
import os
import time

import numpy as np
from functools import lru_cache

from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from ablation_configs import *

import pandas as pd
import heapq
from scipy.stats import sem, t


from src.models.modules import AblationArchitecture, SimulationEndToEnd
from src.models.modules_multiagent import MultiAgentArchitecture
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.calculate_statistics import calculate_statistics
# from src.retrain_model import train_model_retrain, retrain_model
from src.utils.activation_processing import process_activations
from src.utils.plotting import save_double_param_figures, save_single_param_figures, save_fixed_double_param_figures, \
    save_fixed_triple_param_figures, save_key_param_figures, save_delta_figures, plot_regime_lengths, \
    plot_awareness_results, plot_accuracy_vs_vision

sys.path.append(os.getcwd())

# from src.objects import *
from torch.utils.data import DataLoader, random_split
import tqdm
import torch.nn as nn
import torch
#from src.supervised_learning import TrainDatasetBig, EvalDatasetBig, custom_collate, cLstmModel, CNNModel, cRNNModel, fCNN, tCNN, sRNN, sMLP, SimpleMultiDataset
from src.supervised_learning import SimpleMultiDataset
import traceback
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)




class HungarianMSELoss(nn.Module):
    def __init__(self):
        super(HungarianMSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        batch_size, num_classes = predictions.shape
        device = predictions.device
        batch_losses = []
        for b in range(batch_size):
            pred = predictions[b].detach().cpu().numpy()
            tgt = targets[b].detach().cpu().numpy()
            cost_matrix = np.zeros((num_classes, num_classes))
            for i in range(num_classes):
                for j in range(num_classes):
                    cost_matrix[i, j] = (pred[i] - tgt[j])**2
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            reordered_targets = torch.zeros(num_classes, device=device)
            for i, j in zip(row_ind, col_ind):
                reordered_targets[i] = targets[b, j]
            batch_losses.append(torch.mean((predictions[b] - reordered_targets)**2))
        return torch.mean(torch.stack(batch_losses))


def decode_event_name(name):
    byte_list = name.tolist()

    byte_list = [b for b in byte_list if b != 0]

    params_str = ''.join([chr(c) for c in byte_list])
    main_part, numerical_suffix = params_str.split('-')

    visible_baits = int(main_part[main_part.index('b') + 1:main_part.index('w')])
    swaps = int(main_part[main_part.index('w') + 1:main_part.index('v')])
    visible_swaps = int(main_part[main_part.index('v') + 1])
    first_swap_is_both = 1 if 'f' in main_part else 0
    second_swap_to_first_loc = 1 if 's' in main_part else 0
    delay_2nd_bait = 1 if 'd' in main_part else 0

    '''binary_suffix = format(int(numerical_suffix), '04b')

    # Extract parameters from the binary suffix
    first_bait_size = int(binary_suffix[3])
    uninformed_bait = int(binary_suffix[2])
    uninformed_swap = int(binary_suffix[1])
    first_swap = int(binary_suffix[0])'''

    # Convert the numerical suffix to binary
    binary_suffix = int(numerical_suffix)

    # Extract parameters from the binary suffix using bitwise operations
    first_bait_size = binary_suffix & 1
    uninformed_bait = (binary_suffix >> 1) & 1
    uninformed_swap = (binary_suffix >> 2) & 1
    first_swap = (binary_suffix >> 3) & 1

    # print(numerical_suffix, first_bait_size)

    # Calculate conditions for special parameters
    swaps_gt_0 = swaps > 0
    first_swap_is_both_false = not first_swap_is_both
    delay_2nd_bait_false = not delay_2nd_bait
    swaps_eq_2 = swaps == 2
    visible_baits_eq_1 = visible_baits == 1
    visible_swaps_eq_1 = visible_swaps == 1

    '''return pd.Series({
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
    })'''
    return {
        "visible_baits": visible_baits,
        "swaps": swaps,
        "visible_swaps": visible_swaps,
        "first_swap_is_both": first_swap_is_both if swaps_gt_0 else 'na',
        "second_swap_to_first_loc": second_swap_to_first_loc if swaps_eq_2 and delay_2nd_bait_false else 'na',
        "delay_2nd_bait": delay_2nd_bait if swaps_gt_0 and first_swap_is_both_false else 'na',
        "first_bait_size": first_bait_size,
        "uninformed_bait": uninformed_bait if visible_baits_eq_1 else 'na',
        "uninformed_swap": uninformed_swap if swaps_eq_2 and visible_swaps_eq_1 else 'na',
        "first_swap": first_swap if swaps_gt_0 and not delay_2nd_bait and not first_swap_is_both else 'na'
    }




def create_scheduler(model, total_steps, global_lr=1e-3, pct_start=0.2, div_factor=2, final_div_factor=5):
    param_groups = []

    for name, module in model.named_children():
        if hasattr(module, 'parameters'):
            params = list(module.parameters())
            if params:
                param_groups.append({
                    'params': params,
                    'max_lr': module.learn_rate_multiplier * global_lr
                })

    optimizer = torch.optim.AdamW(param_groups, lr=global_lr, betas=(0.9, 0.999))

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[group['max_lr'] for group in param_groups],
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

    return optimizer, scheduler

def load_model(model_type, model_kwargs, device, multi=False):

    if model_type[0] != 'a':
        model_class = {
            'cnn': lambda: CNNModel(use_conv2=True, **model_kwargs),
            'fcnn': lambda: fCNN(**model_kwargs),
            'tcnn': lambda: tCNN(**model_kwargs),
            'srnn': lambda: sRNN(**model_kwargs),
            'smlp': lambda: sMLP(**model_kwargs),
            'crnn': lambda: cRNNModel(**model_kwargs),
            'clstm': lambda: cLstmModel(**model_kwargs),
        }[model_type]()
        return model_class.to(device)

    config = BASE_CONFIG.copy()
    random_probs = BASE_RANDOM.copy()
    model_config, model_random = MODEL_SPECS[model_type]
    config.update(model_config)
    random_probs.update(model_random)

    #print('random probs:', random_probs)
    print("load model", model_type)
    if multi:
        print('multi')
        return MultiAgentArchitecture(config, random_probs, model_kwargs['batch_size']).to(device)
    if "simv2" in model_type:
        return SimulationEndToEnd(config, random_probs, model_kwargs['batch_size']).to(device)
    return AblationArchitecture(config, random_probs, model_kwargs['batch_size']).to(device)
    #return End2EndArchitecture(config, random_probs, model_kwargs['batch_size']).to(device)


def load_last_model(model_save_path, repetition):
    files = [f for f in os.listdir(model_save_path) if f.startswith(f'{repetition}-checkpoint-') and f.endswith('.pt')]

    if not files:
        raise FileNotFoundError(f"No checkpoint files found for repetition {repetition} in {model_save_path}")

    numbers = []
    for f in files:
        parts = f.split('-')
        if len(parts) > 2:
            try:
                number = int(parts[-1].replace('.pt', ''))
                numbers.append(number)
            except ValueError:
                continue

    print('numbers found', numbers)
    highest_number = max(numbers)
    file_name = f'{repetition}-checkpoint-{highest_number}.pt'
    return file_name


def load_model_eval(model_save_path, repetition, use_prior=False, desired_epoch=None):
    if use_prior:
        file_name = f'{repetition}-checkpoint-prior.pt'
    elif desired_epoch is not None:
        file_name = f'{repetition}-checkpoint-{desired_epoch}.pt'
        if desired_epoch == 49: # we name things badly
            file_name = f'{repetition}-checkpoint-{19}.pt'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            files = [f for f in os.listdir(model_save_path) if f == file_name]
            if not files:
                file_name = load_last_model(model_save_path, repetition)
    else:
        files = [f for f in os.listdir(model_save_path) if f.startswith(f'{repetition}-checkpoint-') and f.endswith('.pt')]
        if not files:
            raise FileNotFoundError(f"No checkpoint files found for repetition {repetition} in {model_save_path}")
        file_name = files[-1]


    # Load the model
    print('loading model for eval:', model_save_path, file_name)
    model_kwargs, state_dict = torch.load(os.path.join(model_save_path, file_name))
    return model_kwargs, state_dict

def load_model_data_eval_retrain(test_sets, load_path, target_label, last_timestep, prior_metrics, model_save_path,
                                 repetition, model_type, oracle_labels, save_labels, act_label_names, test_percent,
                                 use_prior=False, desired_epoch=None):
    # November 24: Removed mmap mode 'r'
    mmap_mode = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    special_criterion = nn.CrossEntropyLoss(reduction='none')
    #special_criterion = nn.MSELoss(reduction='mean')
    oracle_criterion = nn.MSELoss(reduction='none')

    print('load_model_data_eval', test_sets)

    if 'hardcoded' not in model_type:
        model_kwargs, state_dict = load_model_eval(model_save_path, repetition, use_prior, desired_epoch=None)#f'{repetition}-model_epoch{epoch_number}.pt'))
        #batch_size = model_kwargs['batch_size']
        batch_size = 2048
        model_kwargs['batch_size'] = batch_size
        model = load_model(model_type, model_kwargs, device)
        model.load_state_dict(state_dict)
    else:
        model_kwargs = {}
        batch_size = 1024
        model_kwargs['batch_size'] = batch_size
        model = load_model(model_type, model_kwargs, device)

    if len(oracle_labels) and oracle_labels[0] is None:
        oracle_labels = []

    prior_metrics_data = {}
    test_regime_data = []

    test_loaders = []
    data, labels, params, oracles, act_labels = [], [], [], [], []
    print(last_timestep, "last timestep")
    for val_set_name in test_sets:
        regime_name = val_set_name[3:]
        dir = os.path.join(load_path, val_set_name)
        current_data = np.load(os.path.join(dir, 'obs.npz'), mmap_mode=mmap_mode)['arr_0']
        data.append(current_data)
        test_regime_array = np.array([regime_name] * len(current_data), dtype=object)
        test_regime_data.append(test_regime_array)

        labels_raw = np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0']  # todo: try labelcheck
        print('loaded eval labels', val_set_name, target_label, labels_raw.shape)
        if target_label == 'shouldGetBig':
            # it's 5 bools, so we take the last and turn it into 1-hot
            if last_timestep:
                x = np.eye(2)[labels_raw[:, -1].astype(int)]  # single timestep
            else:
                x = np.eye(2)[labels_raw.astype(int)].reshape(-1, 10)  # 5 timesteps
            # print(x.shape, x[0])
            labels.append(x)

        elif len(labels_raw.shape) > 2:
            if last_timestep:
                labels.append(labels_raw[..., -1, :])  # use only the last timestep (was last dimension but I changed it)...
            else:
                #print(labels_raw.shape)
                print(target_label)
                labels.append(labels_raw[...,:-1].reshape(-1, 25))
        else:
            #print(labels_raw.shape)
            labels.append(labels_raw.reshape(-1, 25))
            # labels.append(labels_raw)
        # print('first', np.sum(labels_raw, axis=0), labels_raw[15, -1])
        params.append(np.load(os.path.join(dir, 'params.npz'), mmap_mode=mmap_mode)['arr_0'])
        for metric in set(prior_metrics):
            metric_data = np.load(os.path.join(dir, 'label-' + metric + '.npz'), mmap_mode=mmap_mode)['arr_0']
            if metric in prior_metrics_data.keys():
                prior_metrics_data[metric].append(metric_data)
            else:
                prior_metrics_data[metric] = [metric_data]

        if oracle_labels:
            oracle_data = []
            for oracle_label in oracle_labels:
                this_oracle = np.load(os.path.join(dir, 'label-' + oracle_label + '.npz'))['arr_0']
                flattened_oracle = this_oracle.reshape(this_oracle.shape[0], -1)
                oracle_data.append(flattened_oracle)
            combined_oracle_data = np.concatenate(oracle_data, axis=-1)
            oracles.append(combined_oracle_data)

        if save_labels:  # save labels for comparison with activations
            act_label_data_dict = {}
            for act_label_name in act_label_names:
                this_label = np.load(os.path.join(dir, 'label-' + act_label_name + '.npz'))['arr_0']
                flattened_label = this_label.reshape(this_label.shape[0], -1)
                act_label_data_dict[act_label_name] = flattened_label
            act_labels.append(act_label_data_dict)

    prior_metrics_data['test_regime'] = test_regime_data
    for metric, arrays in prior_metrics_data.items():
        prior_metrics_data[metric] = np.concatenate(arrays, axis=0)

    print('shape', labels[0].shape, 'last timestep', last_timestep)

    # November 2024 simplification, will temporarily break train/test split

    #test_indices = filter_indices(data, labels, params, oracles, is_train=False, test_percent=test_percent)
    #val_dataset = EvalDatasetBig(data, labels, params, oracles, test_indices, metrics=prior_metrics_data, act_list=act_labels)
    #test_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate))

    val_dataset = SimpleMultiDataset(data, labels, params, oracles, metrics=prior_metrics_data, act_list=act_labels)
    test_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))

    return test_loaders, special_criterion, oracle_criterion, model, device, data, labels, params, oracles, act_labels, batch_size, prior_metrics_data, model_kwargs


def analyze_batch_awareness(model, inputs, vision_prob, num_samples=1):
    device = inputs.device
    batch_size = inputs.shape[0]

    full_perception = model.perception(inputs)
    full_beliefs = model.my_belief(full_perception['treats_visible'],
                                   torch.ones(batch_size, 5, device=device))

    gt_null = full_beliefs[..., 5] > 0.5  # [batch, 2]
    gt_positions = full_beliefs[..., :5].argmax(dim=2)  # [batch, 2]

    vision = torch.rand(batch_size, 5, device=device) < vision_prob
    #masked_opponent_vision = full_perception['opponent_vision'] * vision

    masked_input = inputs * vision.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    masked_perception = model.perception(masked_input)
    beliefs = model.op_belief(masked_perception['treats_visible'], vision.float())

    pred_null = beliefs[..., 5] > 0.5  # [batch, 2]
    pred_positions = beliefs[..., :5].argmax(dim=2)  # [batch, 2]

    correct = ~gt_null & ~pred_null & (pred_positions == gt_positions)
    incorrect = ~gt_null & ~pred_null & (pred_positions != gt_positions)
    null = pred_null

    state_masks = [
        correct[:, 0] & correct[:, 1],  # TT
        correct[:, 0] & incorrect[:, 1],  # TF
        correct[:, 0] & null[:, 1],  # TN
        incorrect[:, 0] & correct[:, 1],  # FT
        incorrect[:, 0] & incorrect[:, 1],  # FF
        incorrect[:, 0] & null[:, 1],  # FN
        null[:, 0] & correct[:, 1],  # NT
        null[:, 0] & incorrect[:, 1],  # NF
        null[:, 0] & null[:, 1],  # NN
    ]

    awareness = torch.zeros(batch_size, 9, device=device)
    for i, mask in enumerate(state_masks):
        awareness[:, i] = mask.float()

    state_names = ['TT', 'TF', 'TN', 'FT', 'FF', 'FN', 'NT', 'NF', 'NN']
    return {name: awareness[:, i] for i, name in enumerate(state_names)}


def should_include(data, label, param, oracle, bucket_ratio=0.8, is_train=True):
    composite_key = f"{data}_{label}_{param}_{oracle}"
    hasher = hashlib.sha256()
    hasher.update(composite_key.encode('utf-8'))
    hash_digest = hasher.hexdigest()
    bucket = int(hash_digest[-1], 16) < 16 * bucket_ratio
    return bucket if is_train else not bucket


def filter_indices(data_list, labels_list, params_list, oracles_list, is_train=True, test_percent=0.2):
    included_indices = []
    train_percent = 1 - test_percent
    global_index = 0

    for sublist_idx in range(len(data_list)):
        sublist_data = data_list[sublist_idx]
        sublist_labels = labels_list[sublist_idx]
        sublist_params = params_list[sublist_idx]
        sublist_oracles = oracles_list[sublist_idx] if oracles_list and len(oracles_list) > sublist_idx else [None] * len(sublist_data)

        for local_index in range(len(sublist_data)):
            data = sublist_data[local_index]
            label = sublist_labels[local_index]
            param = sublist_params[local_index]
            oracle = sublist_oracles[local_index]

            if should_include(data, label, param, oracle, train_percent, is_train):
                included_indices.append(global_index)

            global_index += 1

    return included_indices


def weight_distance(original_weights, new_weights):
    return sum(torch.norm(ow - nw).item() for ow, nw in zip(original_weights, new_weights))


@lru_cache(maxsize=None)
def get_t_value(n, confidence=0.95):
    return t.ppf((1 + confidence) / 2, n - 1)


def calculate_ci(group):
    group_arr = group.values
    n = group_arr.shape[0]
    m = group_arr.mean()
    std_err = sem(group_arr)
    h = std_err * get_t_value(n)

    # return {'lower': m - h, 'upper': m + h, 'mean': m}
    return pd.DataFrame({'lower': [m - h], 'upper': [m + h], 'mean': [m]}, columns=['lower', 'upper', 'mean'])


lenny = len("tensor(")


def convert_to_numeric(x):
    if isinstance(x, str):
        return int(x[lenny:-1])
    if torch.is_tensor(x) and x.numel() == 1:
        return x.item()
    try:
        return int(x)
    except (ValueError, TypeError):
        print(x)
        return np.nan


def custom_merge(df1, df2, cols_to_check, other_cols):
    merge_conditions = []
    for col in cols_to_check:
        condition = ((df1[col] == df2[col]) | (df1[col] == -1) | (df2[col] == -1))
        merge_conditions.append(condition)

    final_condition = merge_conditions[0]
    for condition in merge_conditions[1:]:
        final_condition &= condition

    for col in other_cols:
        final_condition &= (df1[col] == df2[col])

    return df1[final_condition]


def get_descendants(category, op, categories):
    src, tgt = op.split('-')

    if src not in category:
        return []

    descendants = []
    # below technique gets all categories with the change, not just one otherwise identical (usually 6)

    # for potential_descendant in categories:
    #    if tgt in potential_descendant and src not in potential_descendant:
    #        descendants.append(potential_descendant)
    direct_descendant = category.replace(src, tgt)
    return [direct_descendant]
    # return descendants


def extract_last_value_from_list_str(value):
    if not isinstance(value, str):
        return value

    if isinstance(value, str):
        last_row = value.split('\n')[-1].strip()

        numbers = re.findall(r'\d+\.\s*', last_row)

        if numbers:
            return [float(num.strip()) for num in numbers]
    return value

def get_last_timestep(value):
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed[-1]
        except (ValueError, SyntaxError):
            return value
    return value




def save_figures(path, df, avg_loss, ranges, range_dict, range_dict3, params, last_epoch_df, num=10,
                 key_param_stats=None, oracle_stats=None, key_param=None, delta_sum=None, delta_x=None,
                 key_param_stats_special=[]):
    top_pairs = sorted(ranges.items(), key=lambda x: x[1], reverse=True)[:num]
    top_n_ranges = heapq.nlargest(num, range_dict, key=range_dict.get)
    top_n_ranges3 = heapq.nlargest(num, range_dict3, key=range_dict3.get)

    if delta_sum:
        save_delta_figures(path, delta_sum, delta_x)

    if key_param_stats is not None:
        save_key_param_figures(path, key_param_stats, oracle_stats, key_param, key_param_stats_special=key_param_stats_special)

    save_double_param_figures(path, top_pairs, avg_loss, last_epoch_df)
    save_single_param_figures(path, params, avg_loss, last_epoch_df)
    save_fixed_double_param_figures(path, top_n_ranges, df, avg_loss, last_epoch_df)
    save_fixed_triple_param_figures(path, top_n_ranges3, df, avg_loss, last_epoch_df)


def string_to_tensor(s):
    # Use regex to extract the list of integers from the string
    numbers = list(map(int, re.findall(r'(\d+)', s)))
    # Convert the list of integers to a tensor
    return torch.tensor(numbers, dtype=torch.int32)


def decode_param(tensor):
    byte_list = tensor.tolist()
    byte_list = [b for b in byte_list if b != 0]
    char_list = [chr(c) for c in byte_list]
    return ''.join(char_list)


def write_metrics_to_file(filepath, df, ranges, params, stats, key_param=None, d_s=None, d_x=None):
    '''
    Produces a file showing several interesting metrics from the session(s)
    '''
    df2 = df[df['epoch'] == df['epoch'].max()]
    with open(filepath, 'w') as f:
        mean_accuracy = df2['accuracy'].mean()
        std_accuracy = df2['accuracy'].std()
        f.write(f"Average accuracy: {mean_accuracy} ({std_accuracy})\n")

        df2['param'] = df2['param'].apply(lambda s: decode_param(string_to_tensor(s)))
        param_stats = df2.groupby('param')['accuracy'].agg(['mean', 'std', 'count'])
        perfect_accuracy_tasks = param_stats[param_stats['mean'] == 1.0]
        f.write(f"Number of tasks with perfect accuracy: {len(perfect_accuracy_tasks)}\n\n")

        top_10 = param_stats.nlargest(10, 'mean')
        bottom_10 = param_stats.nsmallest(10, 'mean')

        f.write("Top 10 tasks based on average accuracy:\n")
        for param, row in top_10.iterrows():
            f.write(f"Task: {param} (n={row['count']}), Mean accuracy: {row['mean']}, Std deviation: {row['std']}\n")
        f.write("\nBottom 10 tasks based on average accuracy:\n")
        for param, row in bottom_10.iterrows():
            f.write(f"Task: {param} (n={row['count']}), Mean accuracy: {row['mean']}, Std deviation: {row['std']}\n")

        sorted_ranges = sorted(ranges.items(), key=lambda x: x[1], reverse=True)
        f.write("All parameters sorted by range of mean accuracy across values:\n")
        for this_param in sorted_ranges:
            f.write(f"Parameter: {this_param[0]}, Range: {this_param[1]}\n")

        if key_param:
            f.write(f"\nAccuracy for each unique value of {key_param}:\n")
            key_param_groups = df2.groupby(key_param)

            for key, group in key_param_groups:
                f.write(f"\n{key_param}: {key}\n")
                mean_accuracy = group['accuracy'].mean()
                std_accuracy = group['accuracy'].std()
                f.write(f"Acc {key_param}: {key}: {mean_accuracy} ({std_accuracy})\n")

                train_regime = key.split('-')[-1] # get the train regime
                # aggregate the mean but only where test-regime is not in the train regime set...
                test_regimes = group['test_regime']
                sub_regime_keys = [
                    "Fn", "Nf", "Tn", "Ff", "Tf", "Ft", "Tt"
                ]
                train_map = {
                    's3': [x + '0' for x in sub_regime_keys] + [x + '1' for x in sub_regime_keys] + ['Nn1a', 'Nt1a', 'Nn1b', 'Nt1b',],
                    's2': [x + '0' for x in sub_regime_keys] + ['Tt1', 'Nn0', 'Nt0'],
                    's1': [x + '0' for x in sub_regime_keys] + ['Nn0', 'Nt0'],
                    's21': [x + '0' for x in sub_regime_keys] + ['Tt1', 'Nn1a', 'Nt1a'],
                    'homogeneous': ['Tt0', 'Ff0', 'Nn0', 'Tt1', 'Ff1', 'Nn1']
                }

                group['is_novel_task'] = True#~test_regimes.isin(train_map[train_regime]) if 'hard' not in key else False
                mean_novel_accuracy = group.loc[group['is_novel_task'], 'accuracy'].mean()
                std_novel_accuracy = group.loc[group['is_novel_task'], 'accuracy'].std()
                f.write(f"Novel acc {key_param}: {key}: {mean_novel_accuracy} ({std_novel_accuracy})\n")

                group['isnt_novel_task'] = True#test_regimes.isin(train_map[train_regime]) if 'hard' not in key else True
                mean_xnovel_accuracy = group.loc[group['isnt_novel_task'], 'accuracy'].mean()
                std_xnovel_accuracy = group.loc[group['isnt_novel_task'], 'accuracy'].std()
                f.write(f"Old acc {key_param}: {key}: {mean_xnovel_accuracy} ({std_xnovel_accuracy})\n")


        param_value_stats = []
        for param in params:
            param_stats_mean = df2.groupby(param)['accuracy'].mean().reset_index().values
            param_stats_std = df2.groupby(param)['accuracy'].std().reset_index().values
            # Include the parameter's name in the values
            param_stats_with_name = [(param, value, mean, std) for (value, mean), (_, std) in
                                     zip(param_stats_mean, param_stats_std)]
            param_value_stats.extend(param_stats_with_name)

        param_value_stats.sort(key=lambda x: x[2], reverse=True)  # sort by mean accuracy

        f.write("All parameter values sorted by average accuracy:\n")
        for param, value, mean_accuracy, std_dev in param_value_stats:
            f.write(f"Parameter: {param}, Value: {value}, Mean accuracy: {mean_accuracy}, Std. deviation: {std_dev}\n")

        f.write("\nCorrelations between parameters and accuracy:\n")
        for param in stats['vars']:
            f.write(f"Correlation between {param} and acc: {stats['accuracy_correlations'][param]}\n")

        f.write("\nCorrelations between parameters:\n")
        # print(len(params), len(stats['param_correlations']))
        # print(params, stats['param_correlations'].columns)
        for i in range(len(stats['param_correlations'])):
            for j in range(i + 1, len(stats['param_correlations'])):
                f.write(f"Correlation between {stats['param_correlations'].columns[i]} and {stats['param_correlations'].columns[j]}: {stats['param_correlations'].iloc[i, j]}\n")

        if d_s is not None:
            f.write("\nDelta prediction results:\n")
            f.write(str(d_x))
            f.write("\nDelta prediction table:\n")
            f.write(str(d_s))

        '''key_param_corr = df2.corr()[key_param].sort_values(key='absolute', ascending=False)
        f.write("\nCorrelations between key parameter and other parameters:\n")
        for param, corr in key_param_corr.iteritems():
            f.write(f"Correlation between {key_param} and {param}: {corr}\n")'''


def find_df_paths(directory, file_pattern):
    """Find all CSV files in a directory based on a pattern."""
    return glob.glob(os.path.join(directory, file_pattern))


def run_supervised_session(save_path, repetitions=1, epochs=5, train_sets=None, eval_sets=None,
                           load_path='supervised', oracle_labels=[], skip_train=True, batch_size=64,
                           prior_metrics=[], key_param=None, key_param_value=None, save_every=1, skip_calc=True,
                           oracle_early=False, skip_eval=False, oracle_is_target=True, skip_figures=True,
                           act_label_names=[], skip_activations=True, batches=5000, label='correct-loc',
                           last_timestep=True, model_type=None, do_retrain_model=True, train_sets_dict=None,
                           curriculum_name=None):
    '''
    Runs a session of supervised learning. Different steps, such as whether we train+save models, evaluate models, or run statistics on evaluations, are optional.

    '''
    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap']

    test_names = []

    num_random_tests = 1
    top_n = 5

    test = 0
    model_kwargs = {"batch_size": 1024, "oracle_early": oracle_early, "hidden_size": 32, "num_layers": 3, "kernels": 16, "kernel_size1": 3, "kernel_size2": 5, "stride1": 1, "pool_kernel_size": 3, "pool_stride": 1, "padding1": 1, "padding2": 1, "use_pool": False, "use_conv2": False, "kernels2": 16, "lr": 0.0003, "oracle_len": 0, "output_len": 5, "channels": 3}

    model_name = "".join([str(x) + "," for x in model_kwargs.values()])

    dfs_paths = []
    last_epoch_df_paths = []
    loss_paths = []

    if do_retrain_model:
        retrain_path = save_path + '-retrain'
        os.makedirs(retrain_path, exist_ok=True)

    good_for_early_stop = 0
    print('repetitions', repetitions)
    for repetition in range(repetitions):
        if not skip_train:
            loss_file_path = os.path.join(save_path, f'losses-{repetition}.csv')
            should_train = True
            if os.path.exists(loss_file_path) and False:
                df = pd.read_csv(loss_file_path, index_col=None)
                print(df.columns, os.path.abspath(loss_file_path), df)
                final_acc = df['Accuracy'].iloc[-1]
                if final_acc >= 0.996:
                    should_train = False
                    good_for_early_stop += 1
            if should_train:
                final_acc = train_model(train_sets, label, load_path=load_path, model_type=model_type,
                            save_path=save_path, epochs=epochs, batches=batches, model_kwargs=model_kwargs,
                            oracle_labels=oracle_labels, repetition=repetition, save_every=save_every,
                            oracle_is_target=oracle_is_target, last_timestep=last_timestep, train_sets_dict=train_sets_dict,
                            curriculum_name=curriculum_name)
                if final_acc > 0.996:
                    good_for_early_stop += 1

        loss_paths.append(os.path.join(save_path, f'losses-{repetition}.csv'))
        if good_for_early_stop > 4 and False:
            break
        if do_retrain_model:
            epoch = epochs - 1
            print('retraining', epoch)
            retrain_model(eval_sets, label, load_path=load_path,
                          model_type=model_type,
                          model_load_path=save_path,
                          retrain_path=retrain_path,
                          oracle_labels=oracle_labels,
                          repetition=repetition,
                          epoch_number=epoch,
                          prior_metrics=prior_metrics,
                          act_label_names=act_label_names,
                          last_timestep=last_timestep,
                          retrain_batches=batches)
    if not skip_eval:
        print('doing eval')
        loss_file_pattern = os.path.join(save_path, 'losses-*.csv')
        all_loss_paths = glob.glob(loss_file_pattern)

        top_repetitions = []

        for loss_path in all_loss_paths:
            rep_str = os.path.basename(loss_path).replace('losses-', '').replace('.csv', '')
            rep_num = int(rep_str)
            df = pd.read_csv(loss_path)
            if not df.empty:
                final_loss = df['Novel_Loss'].iloc[-1]
                top_repetitions.append((rep_num, final_loss))

        top_repetitions.sort(key=lambda x: x[1])
        best_three = top_repetitions[:min(top_n, len(top_repetitions))]
        best_repetition_numbers = [rep for rep, loss in best_three]

        #if "-r-" in model_type or 'hard' in model_type:
        #    best_repetition_numbers = [0]

        print(f"best repetitions: {best_three}")
        for repetition in best_repetition_numbers:
            for epoch in [epochs - 1]:
                print('evaluating', epoch, model_name, save_path)
                #try:
                for this_path in [save_path]:#, retrain_path]: #retrain path removed
                    for eval_prior in [False]:#[False, True] if this_path == save_path else [False]:
                        print('e', last_timestep)
                        df_paths = evaluate_model(eval_sets, label, load_path=load_path,
                                              model_type=model_type,
                                              model_load_path=this_path,
                                              oracle_labels=[None],
                                              repetition=repetition,
                                              epoch_number=epoch,
                                              prior_metrics=prior_metrics,
                                              oracle_is_target=oracle_is_target,
                                              act_label_names=act_label_names,
                                              oracle_early=oracle_early,
                                              last_timestep=False,
                                              use_prior=eval_prior,
                                              num_activation_batches=0)
                        print('d')
                        if df_paths is not None:
                            dfs_paths.extend(df_paths)
                            if epoch == epochs - 1 or eval_prior:
                                last_epoch_df_paths.extend(df_paths)
                #except Exception as e:
                #    print(f"Error {e}")

        
        #loss_paths.append(os.path.join(retrain_path, f'losses-{repetition}.csv'))
        #for file in glob.glob(os.path.join(retrain_path, '*-rt-losses.csv')):
        #    loss_paths.append(file)

    dfs_paths = []
    for this_path in [save_path]:#, retrain_path]:
        if skip_train:
            #skipping prior
            dfs_paths.extend([path for path in find_df_paths(this_path, 'param_losses_*_*.csv') if 'prior' not in path])
            print('sup sess dfs paths', dfs_paths, save_path)
        if len(dfs_paths):
            matches = []
            prior_matches = []
            for path in dfs_paths:
                match = re.search(r'(\d+)', path.split('_')[-2])
                if 'prior' in path.split('_')[-2]:
                    prior_matches.append(path)
                if match:
                    matches.append((path, int(match.group())))
            max_epoch = max(epoch for path, epoch in matches)
            last_epoch_df_paths = [path for path, epoch in matches if epoch == max_epoch]
            if prior_matches:
                last_epoch_df_paths.extend(prior_matches)

        if not skip_calc and len(last_epoch_df_paths):
            print('loading dfs (with gzip)...', dfs_paths)

            filtered_paths = []
            for path in last_epoch_df_paths:
                parts = path.split('_')
                if len(parts) >= 2:
                    rep_part = parts[-1].replace('.csv', '').replace('.gz', '')
                    try:
                        rep_num = int(rep_part)
                        if rep_num in best_repetition_numbers[:top_n]:  # Use same best_repetition_numbers from eval
                            filtered_paths.append(path)
                    except ValueError:
                        continue

            df_list = [pd.read_csv(df_path) for df_path in filtered_paths]
            combined_df = pd.concat(df_list, ignore_index=True)

            last_df_list = [pd.read_csv(df_path) for df_path in filtered_paths]
            last_epoch_df = pd.concat(last_df_list, ignore_index=True)
            replace_dict = {'1': 1, '0': 0}
            combined_df.replace(replace_dict, inplace=True)
            last_epoch_df.replace(replace_dict, inplace=True)
            combined_df[key_param] = key_param_value
            last_epoch_df[key_param] = key_param_value
            # print('join replace took', time.time()-cur_time)
            # print('last_df cols', last_epoch_df.columns, last_epoch_df.t)
            combined_df['i-informedness'] = combined_df['i-informedness'].fillna('none')
            last_epoch_df['i-informedness'] = last_epoch_df['i-informedness'].fillna('none')


            avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, _, _, _, _ = calculate_statistics(
                combined_df, last_epoch_df, params + prior_metrics, skip_3x=True)

            write_metrics_to_file(os.path.join(this_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats, key_param=key_param)

            if not skip_figures:
                save_figures(os.path.join(this_path, 'figs'), combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                             params + prior_metrics, last_epoch_df, num=12)

            test_names.append(model_name)

    if not skip_activations:
        print('corring activations...')
        process_activations(save_path, [epochs - 1], [x for x in range(repetitions)], use_inputs=False) #use 0 in epochs for prior
        process_activations(save_path + '-retrain', [epochs - 1], [x for x in range(repetitions)], use_inputs=False)

    loss_file_pattern = os.path.join(save_path, 'losses-*.csv')
    all_loss_paths = glob.glob(loss_file_pattern)
    
    top_repetitions = []
    for loss_path in all_loss_paths:
        try:
            rep_str = os.path.basename(loss_path).replace('losses-', '').replace('.csv', '')
            rep_num = int(rep_str)
            df = pd.read_csv(loss_path)
            if not df.empty:
                final_loss = df['Novel_Loss'].iloc[-1]
                final_accuracy = df['Accuracy'].iloc[-1] if 'Accuracy' in df.columns else 0
                top_repetitions.append((rep_num, final_loss, loss_path))
        except Exception as e:
            print(f"Error reading {loss_path}: {e}")
            continue

    
    top_repetitions.sort(key=lambda x: (x[1]))
    best_n = top_repetitions[:min(top_n, len(top_repetitions))]
    best_repetition_numbers = [rep for rep, _, _, in best_n]
    
    print(f"Best {top_n} repetitions based on validation performance:")
    for i, (rep, loss, _) in enumerate(best_n):
        print(f"  #{i+1}: Rep {rep}, Loss: {loss:.4f}")
    
    filtered_dfs_paths = []
    filtered_last_epoch_df_paths = []
    filtered_loss_paths = []
    
    for path in dfs_paths:
        parts = path.split('_')
        if len(parts) >= 2:
            rep_part = parts[-1].replace('.csv', '').replace('.gz', '')
            rep_num = int(rep_part)
            if rep_num in best_repetition_numbers:
                filtered_dfs_paths.append(path)
    
    for path in last_epoch_df_paths:
        parts = path.split('_')
        if len(parts) >= 2:
            rep_part = parts[-1].replace('.csv', '').replace('.gz', '')
            rep_num = int(rep_part)
            if rep_num in best_repetition_numbers:
                filtered_last_epoch_df_paths.append(path)
    
    for path in loss_paths:
        rep_str = os.path.basename(path).replace('losses-', '').replace('.csv', '')
        rep_num = int(rep_str)
        if rep_num in best_repetition_numbers:
            filtered_loss_paths.append(path)

    if len(filtered_dfs_paths) < 1:
        print("couldn't find good validation losses")
        return dfs_paths, last_epoch_df_paths, loss_paths

    #exit()

    return filtered_dfs_paths, filtered_last_epoch_df_paths, filtered_loss_paths