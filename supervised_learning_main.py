import glob
import itertools
import pickle

import sys
import os
import time
import numpy as np
from functools import lru_cache

import pandas as pd
import heapq

from scipy.stats import sem, t

from src.utils.activation_processing import process_activations
from src.utils.plotting import save_double_param_figures, save_single_param_figures, save_fixed_double_param_figures, \
    save_fixed_triple_param_figures, save_key_param_figures, save_delta_figures

sys.path.append(os.getcwd())

# from src.objects import *
from torch.utils.data import DataLoader, random_split
import tqdm
import torch.nn as nn
import torch
from src.supervised_learning import RNNModel, TrainDatasetBig, EvalDatasetBig, FeedForwardModel
import traceback
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


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


class SaveActivations:
    def __init__(self):
        self.activations_out = None
        self.activations_hidden = None

    def __call__(self, module, module_in, module_out):
        self.activations_out = module_out[0]
        self.activations_hidden = module_out[1]


def evaluate_model(test_sets, target_label, load_path='supervised/', model_save_path='', oracle_labels=[], repetition=0,
                   epoch_number=0, prior_metrics=[], num_activation_batches=5, use_ff=False, oracle_is_target=False, act_label_names=[]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    special_criterion = nn.CrossEntropyLoss(reduction='none')
    oracle_criterion = nn.MSELoss(reduction='none')

    model_kwargs, state_dict = torch.load(os.path.join(model_save_path, f'{repetition}-model_epoch{epoch_number}.pt'))
    batch_size = model_kwargs['batch_size']
    if use_ff:
        model = FeedForwardModel(**model_kwargs).to(device)
    else:
        model = RNNModel(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if len(oracle_labels) and oracle_labels[0] is None:
        oracle_labels = []

    param_losses_list = []
    prior_metrics_data = {}

    test_loaders = []
    data, labels, params, oracles, act_labels = [], [], [], [], []
    for val_set_name in test_sets:
        dir = os.path.join(load_path, val_set_name)
        data.append(np.load(os.path.join(dir, 'obs.npz'), mmap_mode='r')['arr_0'])
        labels.append(np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0'])
        params.append(np.load(os.path.join(dir, 'params.npz'), mmap_mode='r')['arr_0'])
        for metric in set(prior_metrics):
            metric_data = np.load(os.path.join(dir, 'label-' + metric + '.npz'), mmap_mode='r')['arr_0']
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

        if True: # save labels for comparison with activations
            act_label_data_dict = {}
            for act_label_name in act_label_names:
                this_label = np.load(os.path.join(dir, 'label-' + act_label_name + '.npz'))['arr_0']
                flattened_label = this_label.reshape(this_label.shape[0], -1)
                act_label_data_dict[act_label_name] = flattened_label
            act_labels.append(act_label_data_dict)

    for metric, arrays in prior_metrics_data.items():
        prior_metrics_data[metric] = np.concatenate(arrays, axis=0)

    val_dataset = EvalDatasetBig(data, labels, params, oracles, prior_metrics_data, act_labels)
    test_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=True))

    hook = SaveActivations()
    activation_data = {
        'activations_out': [],
        'activations_hidden_short': [],
        'activations_hidden_long': [],
        'inputs': [],
        'labels': [],
        'oracles': [],
    }

    for idx, _val_loader in enumerate(test_loaders):
        with torch.no_grad():
            if use_ff:
                handle = model.fc.register_forward_hook(hook)
            else:
                handle = model.rnn.register_forward_hook(hook)

            tq = tqdm.trange(len(_val_loader))

            for i, (inputs, labels, params, oracles, metrics, act_labels_dict) in enumerate(_val_loader):
                inputs, labels, oracles = inputs.to(device), labels.to(device), oracles.to(device)


                if not oracle_is_target:
                    outputs = model(inputs, oracles)
                    losses = special_criterion(outputs, torch.argmax(labels, dim=1))
                    _, predicted = torch.max(outputs, 1)
                    oracle_accuracy = torch.zeros(labels.size(0), device=device)
                    oracle_outputs = torch.zeros((labels.size(0), 10), device=device)
                else:
                    outputs = model(inputs, None)
                    typical_outputs = outputs[:, :5]
                    _, predicted = torch.max(typical_outputs, 1)
                    oracle_outputs = outputs[:, 5:]

                    losses = special_criterion(typical_outputs, torch.argmax(labels, dim=1))
                    oracle_losses = oracle_criterion(oracle_outputs, oracles).sum(dim=1)
                    binary_oracle_outputs = (oracle_outputs > 0.5).float()
                    oracle_accuracy = ((binary_oracle_outputs == oracles).float().sum(dim=1) / 10).float()

                corrects = (predicted == torch.argmax(labels, dim=1))
                pred = predicted.cpu()
                small_food_selected = (pred == torch.argmax(metrics['loc'][:, :, 0], dim=1))
                big_food_selected = (pred == torch.argmax(metrics['loc'][:, :, 1], dim=1))
                neither_food_selected = ~(small_food_selected | big_food_selected)

                tq.update(1)

                if i < num_activation_batches:
                    activation_data['activations_out'].append(hook.activations_out.cpu().reshape(batch_size, -1))
                    activation_data['activations_hidden_short'].append(
                        hook.activations_hidden[0].cpu().reshape(batch_size, -1))
                    activation_data['activations_hidden_long'].append(
                        hook.activations_hidden[1].cpu().reshape(batch_size, -1))
                    activation_data['inputs'].append(inputs.cpu().numpy().reshape(batch_size, -1))
                    activation_data['labels'].append(labels.cpu().numpy().reshape(batch_size, -1))
                    activation_data['oracles'].append(oracles.cpu().numpy().reshape(batch_size, -1))

                    # Handle each act_label separately
                    for name, act_label in act_labels_dict.items():
                        key = f"act_label_{name}"
                        if key not in activation_data:
                            activation_data[key] = []
                        activation_data[key].append(act_label.cpu().numpy().reshape(batch_size, -1))
                        print(key, act_label.shape)

                if i == num_activation_batches - 1:
                        handle.remove()

                batch_param_losses = [
                    {
                        'param': param,
                        **decode_event_name(param),
                        'epoch': epoch_number,
                        'pred': _pred.item(),
                        'o_pred': o_pred.tolist(),
                        'loss': loss.item(),
                        'accuracy': correct.item(),
                        'small_food_selected': small.item(),
                        'big_food_selected': big.item(),
                        'neither_food_selected': neither.item(),
                        **{x: v[k].numpy() if hasattr(v[k], 'numpy') else v[k] for x, v in metrics.items()},
                        'o_acc': o_acc.item(),
                        ** ({'oracle_loss': oracle_loss.item()} if oracle_is_target else {})

                    }
                    for k, (param, loss, correct, small, big, neither, _pred, o_pred, o_acc, oracle_loss) in enumerate(
                        zip(params, losses, corrects, small_food_selected, big_food_selected, neither_food_selected,
                            pred, oracle_outputs, oracle_accuracy, (oracle_losses if oracle_is_target else [0]*len(pred))))]
                param_losses_list.extend(batch_param_losses)

    # save dfs periodically to free up ram:
    df_paths = []
    os.makedirs(model_save_path, exist_ok=True)

    df = pd.DataFrame(param_losses_list)
    print('saving csv...')
    df.to_csv(os.path.join(model_save_path, f'param_losses_{epoch_number}_{repetition}.csv'), index=False)
    df_paths.append(os.path.join(model_save_path, f'param_losses_{epoch_number}_{repetition}.csv'))
    print('saving activations...')
    with open(os.path.join(model_save_path, f'activations_{epoch_number}_{repetition}.pkl'), 'wb') as f:
        pickle.dump(activation_data, f)
    print('finished')
    return df_paths


def train_model(train_sets, target_label, load_path='supervised/', save_path='', epochs=100,
                model_kwargs=None,
                oracle_labels=[], repetition=0, use_ff=False,
                save_models=True, save_every=5, record_loss=False,
                oracle_is_target=False):
    use_cuda = torch.cuda.is_available()
    if len(oracle_labels) == 0 or oracle_labels[0] == None:
        oracle_labels = []
    data, labels, params, oracles = [], [], [], []
    batch_size = model_kwargs['batch_size']
    lr = model_kwargs['lr']

    for data_name in train_sets:
        dir = os.path.join(load_path, data_name)
        data.append(np.load(os.path.join(dir, 'obs.npz'), mmap_mode='r')['arr_0'])
        labels.append(np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0'])
        params.append(np.load(os.path.join(dir, 'params.npz'), mmap_mode='r')['arr_0'])
        if oracle_labels:
            oracle_data = []
            for oracle_label in oracle_labels:
                this_oracle = np.load(os.path.join(dir, 'label-' + oracle_label + '.npz'), mmap_mode='r')['arr_0']
                flattened_oracle = this_oracle.reshape(this_oracle.shape[0], -1)
                oracle_data.append(flattened_oracle)
            combined_oracle_data = np.concatenate(oracle_data, axis=-1)
            oracles.append(combined_oracle_data)

    train_dataset = TrainDatasetBig(data, labels, params, oracles)
    del data, labels, params, oracles
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0)  # can't use more on windows

    if record_loss:
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_kwargs['oracle_len'] = 0 if len(oracle_labels) == 0 else len(train_dataset.oracles_list[0][0])
    model_kwargs['output_len'] = 5  # np.prod(labels.shape[1:])
    model_kwargs['channels'] = 5  # np.prod(params.shape[2])
    model_kwargs['oracle_is_target'] = oracle_is_target

    device = torch.device('cuda' if use_cuda else 'cpu')
    model = RNNModel(**model_kwargs).to(device) if not use_ff else FeedForwardModel(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    oracle_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = epochs

    t = tqdm.trange(num_epochs * len(train_loader))
    avg_epoch_loss = 100

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (inputs, target_labels, _, oracles, _) in enumerate(train_loader):
            inputs, target_labels, oracles = inputs.to(device), target_labels.to(device), oracles.to(device)
            if not oracle_is_target:
                outputs = model(inputs, oracles)
                loss = criterion(outputs, torch.argmax(target_labels, dim=1))
            else:
                outputs = model(inputs, None)
                typical_outputs = outputs[:, :5]
                oracle_outputs = outputs[:, 5:]
                typical_loss = criterion(typical_outputs, torch.argmax(target_labels, dim=1))
                oracle_loss = oracle_criterion(oracle_outputs, oracles)
                loss = typical_loss + oracle_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.update(1)

            if record_loss:
                total_loss += loss.item()
        if save_models and (epoch % save_every == save_every - 1):
            os.makedirs(save_path, exist_ok=True)
            torch.save([model.kwargs, model.state_dict()],
                       os.path.join(save_path, f'{repetition}-model_epoch{epoch}.pt'))
        torch.cuda.empty_cache()

    if record_loss:
        with torch.no_grad():
            test_loss = 0.0
            for inputs, target_labels, _, oracle_inputs, _ in test_loader:
                inputs, target_labels, oracle_inputs = inputs.to(device), target_labels.to(
                    device), oracle_inputs.to(device)
                outputs = model(inputs, oracle_inputs)
                loss = criterion(outputs, torch.argmax(target_labels, dim=1))
                test_loss += loss.item()

            avg_epoch_loss = test_loss / len(test_loader)

        return avg_epoch_loss


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
    for potential_descendant in categories:
        if tgt in potential_descendant and src not in potential_descendant:
            descendants.append(potential_descendant)
    return descendants

def calculate_statistics(df, last_epoch_df, params, skip_3x=True, skip_2x1=False, key_param=None, skip_1x=True, record_delta_pi=True):
    '''
    Calculates various statistics from datasets of model outputs detailing model performance.
    '''
    avg_loss = {}
    variances = {}
    ranges_1 = {}
    ranges_2 = {}
    range_dict = {}
    range_dict3 = {}

    print('calculating statistics...')

    print('making categorical')
    for param in params:
        if df[param].dtype == 'object':
            df[param] = df[param].astype('category')
            last_epoch_df[param] = last_epoch_df[param].astype('category')
    params = [param for param in params if param not in ['delay', 'perm']]

    print('params:', params)

    param_pairs = itertools.combinations(params, 2)
    param_triples = itertools.combinations(params, 3)

    variable_columns = last_epoch_df.select_dtypes(include=[np.number]).nunique().index[
        last_epoch_df.select_dtypes(include=[np.number]).nunique() > 1].tolist()

    correlations = last_epoch_df[variable_columns + ['accuracy']].corr()
    target_correlations = correlations['accuracy'][variable_columns]
    stats = {
        'param_correlations': correlations,
        'accuracy_correlations': target_correlations,
        'vars': variable_columns
    }

    unique_vals = {param: df[param].unique() for param in params}

    if not skip_1x:
        print('calculating single params')

        for param in params:
            avg_loss[param] = df.groupby([param, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
            means = last_epoch_df.groupby([param]).mean(numeric_only=True)
            ranges_1[param] = means['accuracy'].max() - means['accuracy'].min()
        print('calculating double params')

        for param1, param2 in tqdm.tqdm(param_pairs):
            avg_loss[(param1, param2)] = df.groupby([param1, param2, 'epoch'])['accuracy'].apply(
                calculate_ci).reset_index()

            means = last_epoch_df.groupby([param1, param2]).mean()
            ranges_2[(param1, param2)] = means['accuracy'].max() - means['accuracy'].min()

            if not skip_2x1:
                for value1 in unique_vals[param1]:
                    subset = last_epoch_df[last_epoch_df[param1] == value1]
                    if len(subset[param2].unique()) > 1:
                        new_means = subset.groupby(param2)['accuracy'].mean()
                        range_dict[(param1, value1, param2)] = new_means.max() - new_means.min()

    if not skip_3x:
        for param1, param2, param3 in param_triples:
            ci_df = df.groupby([param1, param2, param3, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
            avg_loss[(param1, param2, param3)] = ci_df

            for value1 in unique_vals[param1]:
                for value2 in unique_vals[param2]:
                    subset = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == param1)]
                    if len(subset[param3].unique()) > 1:
                        new_means = subset.groupby(param3)['accuracy'].mean()
                        range_dict3[(param1, value1, param2, value2, param3)] = new_means.max() - new_means.min()

    df_summary = {}
    delta_operator_summary = {}
    print('key param stats')
    key_param_stats = {}
    oracle_key_param_stats = {}
    if key_param is not None:
        for param in params:
            if param != key_param:
                # Initializing a nested dictionary for each unique key_param value
                for acc_type, save_dict in zip(['accuracy', 'o_acc'], [key_param_stats, oracle_key_param_stats]):
                    for key_val in unique_vals[key_param]:
                        subset = last_epoch_df[last_epoch_df[key_param] == key_val]
                        grouped = subset.groupby(param)[acc_type]
                        means = grouped.mean()
                        stds = grouped.std()
                        counts = grouped.size()
                        z_value = 1.96  # For a 95% CI
                        standard_errors = z_value * np.sqrt(means * (1 - means) / counts)

                        Q1 = grouped.quantile(0.25)
                        Q3 = grouped.quantile(0.75)

                        if key_val not in save_dict:
                            save_dict[key_val] = {}
                        save_dict[key_val][param] = {
                            'mean': means.to_dict(),
                            'std': stds.to_dict(),
                            'q1': Q1.to_dict(),
                            'q3': Q3.to_dict(),
                            'ci': standard_errors.to_dict(),
                        }
                        # dict order is key_val > param > mean/std > param_val

        if record_delta_pi:
            # todo: if delay_2nd_bait or first_swap are na, just make them 0? shouldn't affect anything
            print('calculating delta preds')

            translation_dict = {
                'Fn': 'Fn',
                'Nf': 'Nf',
                'Ff': 'Ff',
                'Tn': 'Tn',
                'Tf': 'Tf',
                'Ft': 'Ft',
                'Tt': 'Tt',
                'Nt': 'Nt',
                'Nn': 'Nn'
            }

            operators = ['T-F', 't-f', 'F-N', 'f-n', 'T-N', 't-n']

            for key_val in unique_vals[key_param]: # for each train regime, etc

                required_columns = [f'pred_{i}' for i in range(5)]
                perm_keys = ['p-b-0', 'p-b-1', 'p-s-0', 'p-s-1', 'delay_2nd_bait', 'first_swap', 'first_bait_size', 'delay']
                # the p-x-y columns show which location is selected for each bait and swap, or -1 if none (only applicable to swaps)
                subset = last_epoch_df[last_epoch_df[key_param] == key_val]
                subset['pred'] = subset['pred'].apply(convert_to_numeric).astype(np.int8)
                subset['informedness'] = subset['informedness'].replace(translation_dict)
                conv = pd.concat([subset, pd.get_dummies(subset['pred'], prefix='pred')], axis=1)
                subset = conv[required_columns + ['informedness', 'correctSelection'] + perm_keys]

                for col in required_columns: # a model might not predict all 5 values
                    if col not in conv.columns:
                        conv[col] = 0

                # OPERATOR ONE
                delta_mean = {key: [] for key in operators}
                delta_std = {key: [] for key in operators}
                delta_mean_correct = {key: [] for key in operators}
                delta_std_correct = {key: [] for key in operators}
                for op in operators:

                    delta_preds = []
                    delta_preds_correct = []
                    for key in translation_dict.values():
                        # key is 1st position, descendents are 2nd
                        descendants = get_descendants(key, op, translation_dict)
                        if len(descendants) < 1:
                            continue
                        inf = subset[subset['informedness'] == key].groupby(perm_keys + ['informedness', 'correctSelection'], observed=True).mean().reset_index()
                        print('ddd', descendants)
                        noinf = subset[subset['informedness'].isin(descendants)].groupby(perm_keys + ['informedness', 'correctSelection'], observed=True).mean().reset_index()
                        print('key', key, len(inf), len(noinf))

                        merged_df = pd.merge(
                            inf,
                            noinf,
                            on=perm_keys,
                            suffixes=('_m', ''),
                            how='inner',
                        )

                        merged_df['changed_target'] = (
                                    merged_df['correctSelection_m'] != merged_df['correctSelection']).astype(int)

                        for i in range(5):
                            merged_df[f'pred_diff_{i}'] = abs(merged_df[f'pred_{i}_m'] - merged_df[f'pred_{i}'])
                        merged_df['total_pred_diff'] = merged_df[[f'pred_diff_{idx}' for idx in range(5)]].sum(
                            axis=1) / 2
                        merged_df['total_pred_diff_correct'] = 1 - abs(
                            merged_df['total_pred_diff'] - merged_df['changed_target'])

                        delta_preds.extend(merged_df['total_pred_diff'].tolist())
                        delta_preds_correct.extend(merged_df['total_pred_diff_correct'].tolist())

                    # these lines are wrong
                    delta_mean[op] = np.mean(delta_preds)
                    delta_std[op] = np.std(delta_preds)
                    delta_mean_correct[op] = np.mean(delta_preds_correct)
                    delta_std_correct[op] = np.std(delta_preds_correct)
                    # all these are for one operator

                delta_operator_summary[key_val] = pd.DataFrame({
                    'operator': list(delta_mean.keys()),
                    'dpred': [f"{delta_mean[key]} ({delta_std[key]})" for key in delta_mean.keys()],
                    'dpred_correct': [f"{delta_mean_correct[key]} ({delta_std_correct[key]})" for key in
                                      delta_mean_correct.keys()]
                })
                print('do', delta_operator_summary[key_val])


                # CATEGORY ONE

                #for col in perm_keys + ['informedness', 'correctSelection']:
                #    print(f"{col} has {subset[col].nunique()} unique values:", subset[col].unique())

                inf = subset[subset['informedness'] == 'Tt'].groupby(perm_keys + ['informedness', 'correctSelection'], observed=True).mean().reset_index()
                noinf = subset[subset['informedness'] != 'Tt'].groupby(perm_keys + ['informedness', 'correctSelection'], observed=True).mean().reset_index()

                merged_df = pd.merge(
                    inf,
                    noinf,
                    on=perm_keys,
                    suffixes=('_m', ''),
                    how='inner',
                )

                merged_df['changed_target'] = (merged_df['correctSelection_m'] != merged_df['correctSelection']).astype(int)

                for i in range(5):
                    merged_df[f'pred_diff_{i}'] = abs(merged_df[f'pred_{i}_m'] - merged_df[f'pred_{i}'])
                merged_df['total_pred_diff'] = merged_df[[f'pred_diff_{idx}' for idx in range(5)]].sum(axis=1) / 2
                merged_df['total_pred_diff_correct'] = 1 - abs(merged_df['total_pred_diff'] - merged_df['changed_target'])

                delta_preds = {}
                delta_preds_correct = {}

                for key in merged_df['informedness'].unique():
                    delta_preds[key] = merged_df.loc[merged_df['informedness'] == key, 'total_pred_diff'].tolist()
                    delta_preds_correct[key] = merged_df.loc[merged_df['informedness'] == key, 'total_pred_diff_correct'].tolist()

                delta_mean = {key: np.mean(val) for key, val in delta_preds.items()}
                delta_std = {key: np.std(val) for key, val in delta_preds.items()}

                delta_mean_correct = {key: np.mean(val) for key, val in delta_preds_correct.items()}
                delta_std_correct = {key: np.std(val) for key, val in delta_preds_correct.items()}

                df_summary[key_val] = pd.DataFrame({
                    'Informedness': list(delta_mean.keys()),
                    'dpred': [f"{delta_mean[key]} ({delta_std[key]})" for key in delta_mean.keys()],
                    'dpred_correct': [f"{delta_mean_correct[key]} ({delta_std_correct[key]})" for key in delta_mean_correct.keys()]
                })

                # todo: add informedness operators, rather than just category
                print(df_summary[key_val])

    print('finished')
    return avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats, oracle_key_param_stats, df_summary, delta_operator_summary


def save_figures(path, df, avg_loss, ranges, range_dict, range_dict3, params, last_epoch_df, num=10,
                 key_param_stats=None,  oracle_stats=None, key_param=None, delta_sum=None, delta_x=None):
    top_pairs = sorted(ranges.items(), key=lambda x: x[1], reverse=True)[:num]
    top_n_ranges = heapq.nlargest(num, range_dict, key=range_dict.get)
    top_n_ranges3 = heapq.nlargest(num, range_dict3, key=range_dict3.get)

    if delta_sum:
        save_delta_figures(path, delta_sum, delta_x)

    if key_param_stats is not None:
        save_key_param_figures(path, key_param_stats, oracle_stats, key_param)

    save_double_param_figures(path, top_pairs, avg_loss, last_epoch_df)
    save_single_param_figures(path, params, avg_loss, last_epoch_df)
    save_fixed_double_param_figures(path, top_n_ranges, df, avg_loss, last_epoch_df)
    save_fixed_triple_param_figures(path, top_n_ranges3, df, avg_loss, last_epoch_df)


def write_metrics_to_file(filepath, df, ranges, params, stats, key_param=None, d_s=None, d_x=None):
    '''
    Produces a file showing several interesting metrics from the session(s)
    '''
    df2 = df[df['epoch'] == df['epoch'].max()]
    with open(filepath, 'w') as f:
        mean_accuracy = df2['accuracy'].mean()
        std_accuracy = df2['accuracy'].std()
        f.write(f"Average accuracy: {mean_accuracy} ({std_accuracy})\n")

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
        #print(len(params), len(stats['param_correlations']))
        #print(params, stats['param_correlations'].columns)
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
    return glob.glob(f"{directory}/{file_pattern}")


def run_supervised_session(save_path, repetitions=1, epochs=5, train_sets=None, eval_sets=None,
                           load_path='supervised', oracle_labels=[], skip_train=True, batch_size=64,
                           prior_metrics=[], key_param=None, key_param_value=None, save_every=1, skip_calc=True,
                           use_ff=False, oracle_layer=0, skip_eval=False, oracle_is_target=False, skip_figures=True,
                           act_label_names=[]):

    '''
    Runs a session of supervised learning. Different steps, such as whether we train+save models, evaluate models, or run statistics on evaluations, are optional.

    '''
    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap']

    test_names = []

    num_random_tests = 1
    lr = 0.002

    test = 0
    while test < num_random_tests:
        try:
            model_kwargs = {"hidden_size": 32, "num_layers": 3, "kernels": 16, "kernel_size1": 3, "kernel_size2": 5, "stride1": 1, "pool_kernel_size": 3, "pool_stride": 1, "padding1": 1, "padding2": 1, "use_pool": False, "use_conv2": False, "kernels2": 16, "batch_size": 256, "lr": 0.001, "oracle_len": 0, "output_len": 5, "channels": 5}

            model_name = "".join([str(x) + "," for x in model_kwargs.values()])

            dfs_paths = []
            last_epoch_df_paths = []
            for repetition in range(repetitions):
                if not skip_train:
                    train_model(train_sets, 'correctSelection', load_path=load_path,
                                save_path=save_path, epochs=epochs, model_kwargs=model_kwargs,
                                oracle_labels=oracle_labels, repetition=repetition, save_every=save_every,
                                use_ff=use_ff, oracle_is_target=oracle_is_target)
                if not skip_eval:
                    for epoch in range(epochs):
                        if epoch % save_every == save_every - 1 or epoch == epochs - 1:
                            print('evaluating', epoch)
                            df_paths = evaluate_model(eval_sets, 'correctSelection', load_path=load_path,
                                                      model_save_path=save_path,
                                                      oracle_labels=oracle_labels, repetition=repetition,
                                                      epoch_number=epoch,
                                                      prior_metrics=prior_metrics,
                                                      use_ff=use_ff,
                                                      oracle_is_target=oracle_is_target,
                                                      act_label_names=act_label_names)
                            dfs_paths.extend(df_paths)
                            if epoch == epochs - 1:
                                last_epoch_df_paths.extend(df_paths)
            else:
                dfs_paths = find_df_paths(save_path, 'param_losses_*_*.csv')
                print(save_path)
                max_epoch = max(int(path.split('_')[-2]) for path in dfs_paths)
                last_epoch_df_paths = [path for path in dfs_paths if int(path.split('_')[-2]) == max_epoch]

            if not skip_calc:
                print('loading dfs...')

                df_list = [pd.read_csv(df_path) for df_path in dfs_paths]
                combined_df = pd.concat(df_list, ignore_index=True)

                last_df_list = [pd.read_csv(df_path) for df_path in last_epoch_df_paths]
                last_epoch_df = pd.concat(last_df_list, ignore_index=True)
                replace_dict = {'1': 1, '0': 0}
                combined_df.replace(replace_dict, inplace=True)
                last_epoch_df.replace(replace_dict, inplace=True)
                last_epoch_df[key_param] = key_param_value
                # print('join replace took', time.time()-cur_time)
                # print('last_df cols', last_epoch_df.columns, last_epoch_df.t)
                combined_df['informedness'] = combined_df['informedness'].fillna('none')
                last_epoch_df['informedness'] = last_epoch_df['informedness'].fillna('none')

                avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, _, _, _, _ = calculate_statistics(
                    combined_df, last_epoch_df, params + prior_metrics, skip_3x=True)

                write_metrics_to_file(os.path.join(save_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats,
                                      key_param=key_param)

                if not skip_figures:
                    save_figures(os.path.join(save_path, 'figs'), combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                             params + prior_metrics, last_epoch_df, num=12)

                test_names.append(model_name)
            test += 1
        except BaseException as e:
            print(e)
            traceback.print_exc()

    print('corring activations...')
    if not skip_calc:
        process_activations(save_path, [epochs-1], [0])
    return dfs_paths, last_epoch_df_paths


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return value
