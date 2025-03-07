import ast
import glob
import hashlib
import itertools
import math
import pickle
import re


import sys
import os
import time

import numpy as np
from functools import lru_cache

from matplotlib import pyplot as plt

from ablation_configs import *

import pandas as pd
import heapq
from scipy.stats import sem, t
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR

from src.models.modules import AblationArchitecture
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
from src.supervised_learning import TrainDatasetBig, EvalDatasetBig, custom_collate, cLstmModel, CNNModel, cRNNModel, fCNN, tCNN, sRNN, sMLP, SimpleMultiDataset
import traceback
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


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


class SaveActivations:
    def __init__(self):
        self.activations = {}

    def save_activation(self, name, save_inputs):
        def hook(module, input, output):
            if save_inputs:
                if isinstance(input, tuple):
                    for idx, item in enumerate(input):
                        if isinstance(item, torch.Tensor):
                            self.activations[f"{name}_input_{idx}"] = item.detach()
                elif isinstance(input, torch.Tensor):
                    self.activations[f"{name}_input"] = input.detach()

            if isinstance(module, nn.LSTM):
                output_seq, (h_n, c_n) = output
                num_layers = h_n.size(0)
                #print(output_seq.shape)
                seq_len = output_seq.size(1)

                for t in range(seq_len):
                    for layer in range(num_layers):
                        self.activations[f"{name}_hidden_state_l{layer}_t{t}"] = h_n[layer, :, :].detach()
                        self.activations[f"{name}_cell_state_l{layer}_t{t}"] = c_n[layer, :, :].detach()

            else:
                if isinstance(output, tuple):
                    for idx, item in enumerate(output):
                        if isinstance(item, torch.Tensor):
                            self.activations[f"{name}_output_{idx}"] = item.detach()
                elif isinstance(output, torch.Tensor):
                    self.activations[f"{name}_output"] = output.detach()

            #print(self.activations.keys(), [(x, len(self.activations[x])) for x in self.activations.keys()])

        return hook


def register_hooks(model, hook):
    save_inputs = True
    for name, layer in model.named_modules():
        layer.register_forward_hook(hook.save_activation(name, save_inputs=save_inputs))
        save_inputs = False


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

class SigmoidTempScheduler:
    def __init__(self, model, start_temp=1.0, end_temp=100.0, total_steps=10000, vision_prob_start=0.9, vision_prob_end=1, rate=4.0):
        self.model = model
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps
        self.current_step = 0
        self.vision_prob = 1.0
        self.vision_prob_end = vision_prob_end
        self.vision_prob_start = vision_prob_start
        self.rate = rate

    def step(self):
        current_temp, current_prob = self.get_temp()

        # Update all module temps
        self.model.treat_perception.sigmoid_temp = current_temp
        self.model.vision_perception.sigmoid_temp = current_temp
        self.model.presence_perception.sigmoid_temp = current_temp
        self.model.my_belief.sigmoid_temp = current_temp
        self.model.op_belief.sigmoid_temp = current_temp
        self.model.my_decision.sigmoid_temp = current_temp
        self.model.op_decision.sigmoid_temp = current_temp
        self.model.vision_prob = current_prob
        self.current_step = min(self.current_step + 1, self.total_steps)
        #print('current temp/prob:', current_temp, current_prob)

    def get_temp(self):
        progress = min(1.0, self.rate * self.current_step / self.total_steps)
        cosine_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return self.start_temp + (self.end_temp - self.start_temp) * cosine_progress, self.vision_prob_start + (self.vision_prob_end - self.vision_prob_start) * cosine_progress


def load_model(model_type, model_kwargs, device):
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
    return AblationArchitecture(config, random_probs, model_kwargs['batch_size']).to(device)


def load_last_model(model_save_path, repetition):
    files = [f for f in os.listdir(model_save_path) if f.startswith(f'{repetition}-checkpoint-') and f.endswith('.pt')]

    if not files:
        raise FileNotFoundError(f"No checkpoint files found for repetition {repetition} in {model_save_path}")

    numbers = []
    for f in files:
        parts = f.split('-')
        if len(parts) > 2:
            try:
                # Extract the number part
                number = int(parts[-1].replace('.pt', ''))
                numbers.append(number)
            except ValueError:
                # Skip if the conversion fails
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
        file_name = load_last_model(model_save_path, repetition)


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
    oracle_criterion = nn.MSELoss(reduction='none')

    if 'hardcoded' not in model_type and '-r-' not in model_type:
        model_kwargs, state_dict = load_model_eval(model_save_path, repetition, use_prior, desired_epoch=desired_epoch)#f'{repetition}-model_epoch{epoch_number}.pt'))
        batch_size = model_kwargs['batch_size']
        batch_size = 1024
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
                labels.append(labels_raw.reshape(-1, 25))
        else:
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


def retrain_model(test_sets, target_label, load_path='supervised/', model_load_path='', oracle_labels=[], repetition=0, save_labels=True,
                  epoch_number=0, prior_metrics=[], last_timestep=True, act_label_names=None, model_type=None, model_kwargs=None,
                  seed=0, test_percent=0.2, retrain_batches=5000, retrain_path=''):
    test_loaders, special_criterion, oracle_criterion, model, device, \
    data, labels, params, oracles, act_labels, batch_size, prior_metrics_data, \
    model_kwargs = load_model_data_eval_retrain(test_sets, load_path,
                                                target_label, last_timestep,
                                                prior_metrics, model_load_path,
                                                repetition,
                                                model_type, oracle_labels,
                                                save_labels, act_label_names,
                                                test_percent)
    print("retraining model on all")
    train_indices = filter_indices(data, labels, params, oracles, is_train=True, test_percent=test_percent)
    test_indices = filter_indices(data, labels, params, oracles, is_train=False, test_percent=test_percent)
    train_dataset = TrainDatasetBig(data, labels, params, oracles, train_indices, metrics=prior_metrics_data)
    val_dataset = TrainDatasetBig(data, labels, params, oracles, test_indices, metrics=prior_metrics_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for optimizer in ['adam']:  # ['sgd', 'adam', 'momentum09', 'nesterov09', 'rmsprop']:
        train_model_retrain(model, train_loader, val_loader, retrain_path, model_kwargs=model_kwargs, opt=optimizer, max_tries=5, repetition=repetition, max_batches=retrain_batches)


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

    # Determine states per treat
    correct = ~gt_null & ~pred_null & (pred_positions == gt_positions)
    incorrect = ~gt_null & ~pred_null & (pred_positions != gt_positions)
    null = pred_null

    # Compute combined states
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

def evaluate_model(test_sets, target_label, load_path='supervised/', model_load_path='', oracle_labels=[], repetition=0,
                   epoch_number=0, prior_metrics=[], num_activation_batches=-1, oracle_is_target=False, act_label_names=[], save_labels=False,
                   oracle_early=False, last_timestep=True, model_type=None, seed=0, test_percent=0.2, use_prior=False, train_sets=None):
    test_loaders, special_criterion, oracle_criterion, model, device, \
    data, labels, params, oracles, act_labels, batch_size, prior_metrics_data, \
    model_kwargs = load_model_data_eval_retrain(test_sets, load_path,
                                                target_label, last_timestep,
                                                prior_metrics, model_load_path,
                                                repetition,
                                                model_type, oracle_labels,
                                                save_labels, act_label_names,
                                                test_percent, use_prior=use_prior,
                                                desired_epoch=epoch_number)
    param_losses_list = []

    print('eval sets', test_sets)

    hook = SaveActivations()
    activation_data = {
        'inputs': [],
        'pred': [],
        'labels': [],
        #'oracles': [],
    }
    register_hooks(model, hook)
    model.eval()
    model.batch_size = 1024

    #print('num_activation_batches', num_activation_batches)
    overall_correct = 0
    overall_total = 0
    np.set_printoptions(threshold=sys.maxsize)

    num_visions = 10

    do_vision_stuff = False
    if True:
        print('evaluating vision changes')
        vision_probs = np.arange(0, 1.05, 0.1)
        uncertainties = [0, 0.1, 0.3, 0.5, 1]
        accuracy_results1 = {u: {prob: [] for prob in vision_probs} for u in uncertainties}
        #awareness_results = {prob: [] for prob in vision_probs}
        #accuracy_results1 = {prob: [] for prob in vision_probs}

        #accuracy_results2 = {prob: [] for prob in vision_probs}



    # i have commented out oracle related things
    # this includes oracle_is_target check
    for idx, _val_loader in enumerate(test_loaders):

        with torch.inference_mode():

            tq = tqdm.trange(len(_val_loader))

            for i, (inputs, labels, params, _, metrics, act_labels_dict) in enumerate(_val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                #inputs, labels, oracles = inputs.to(device), labels.to(device), oracles.to(device)
                #print(labels.shape)
                max_labels = torch.argmax(labels, dim=1)
                #if torch.isnan(max_labels).any():
                #    print("labels has nan")
                vision_prob = 1.0
                #masked_vision = (torch.rand(inputs.size(0), 5, device=inputs.device) <= vision_prob).float()
                #masked_input = inputs * masked_vision.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                #model.my_belief.uncertainty = 0.3
                outputs = model(inputs, None)['my_decision']
                if do_vision_stuff:
                    for uncertainty in uncertainties:
                        for vision_prob in vision_probs:
                            model.vision_prob = vision_prob
                            model.my_belief.uncertainty = uncertainty
                            model.num_visions = num_visions

                            outputs = model(inputs, None)['my_decision']
                            predicted = outputs.argmax(1)
                            accuracy = (predicted == max_labels).float().mean().item()

                            accuracy_results1[uncertainty][vision_prob].append(accuracy)
                            #batch_awareness = analyze_batch_awareness(model, inputs, vision_prob)

                            '''
                            masked_vision = (torch.rand(inputs.size(0), 5, device=inputs.device) <= vision_prob).float()
                            masked_input = inputs * masked_vision.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                            model.my_belief.uncertainty = 0.3
                            outputs = model(masked_input, masked_vision)
                            predicted = outputs.argmax(1)
                            accuracy = (predicted == max_labels).float().mean().item()
                            accuracy_results1[vision_prob].append(accuracy)
    
                            # using these to see difference
                            masked_vision = (torch.rand(inputs.size(0), 5, device=inputs.device) <= vision_prob).float()
                            masked_input = inputs * masked_vision.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                            model.my_belief.uncertainty = 0.0
                            outputs = model(masked_input, masked_vision)
                            predicted = outputs.argmax(1)
                            accuracy = (predicted == max_labels).float().mean().item()
                            accuracy_results2[vision_prob].append(accuracy)
    
                            for k in range(len(inputs)):
                                awareness_results[vision_prob].append(
                                    {key: v[k].item() for key, v in batch_awareness.items()}
                                )'''
                losses = special_criterion(outputs, max_labels)
                predicted = outputs.argmax(1)

                #print(params, predicted[:10], max_labels[:10], labels[0])

                #oracle_accuracy = torch.zeros(labels.size(0), device=device)
                #oracle_outputs = torch.zeros((labels.size(0), 10), device=device)

                # this was alternative if oracle was target
                '''
                outputs = model(inputs, None)
                typical_outputs = outputs[:, :5]
                predicted = typical_outputs.argmax(1)
                oracle_outputs = outputs[:, 5:]

                losses = special_criterion(typical_outputs, torch.argmax(labels, dim=1))
                oracle_losses = oracle_criterion(oracle_outputs, oracles).sum(dim=1)
                binary_oracle_outputs = (oracle_outputs > 0.5).float()
                oracle_accuracy = ((binary_oracle_outputs == oracles).float().sum(dim=1) / 10).float()
                '''

                corrects = predicted.eq(max_labels)
                total = corrects.numel()
                num_correct = corrects.sum().item()
                pred = predicted.cpu()
                overall_correct += num_correct
                overall_total += total
                #small_food_selected = (pred == torch.argmax(metrics['small-loc'][:, -1, :, 0], dim=1))
                #big_food_selected = (pred == torch.argmax(metrics['big-loc'][:, -1, :, 1], dim=1))
                #neither_food_selected = ~(small_food_selected | big_food_selected)

                tq.update(1)

                if i < num_activation_batches or num_activation_batches < 0:
                    real_batch_size = len(labels)
                    for layer_name, activation in hook.activations.items():
                        if layer_name not in activation_data:
                            activation_data[layer_name] = []
                        activation_data[layer_name].append(activation.cpu().reshape(real_batch_size, -1).to(torch.float16))

                    activation_data['inputs'].append(inputs.cpu().numpy().reshape(real_batch_size, -1))
                    activation_data['pred'].append(pred.numpy().reshape(real_batch_size, -1))
                    activation_data['labels'].append(labels.cpu().numpy().reshape(real_batch_size, -1))
                    #activation_data['oracles'].append(oracles.cpu().numpy().reshape(real_batch_size, -1))

                    for name, act_label in act_labels_dict.items():
                        activation_data.setdefault(f"act_{name}", []).append(act_label.cpu().reshape(real_batch_size, -1).numpy())


                batch_param_losses = []
                '''for k, elements in enumerate(zip(params, losses, corrects, small_food_selected, big_food_selected, neither_food_selected, pred, oracle_outputs, oracle_accuracy, oracle_losses if oracle_is_target else [0] * len(pred))):
                    param, loss, correct, small, big, neither, _pred, o_pred, o_acc, oracle_loss = elements
                    data_dict = {
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
                        'o_acc': o_acc.item()
                    }

                    if oracle_is_target:
                        data_dict['oracle_loss'] = oracle_loss.item()

                    for x, v in metrics.items():
                        data_dict[x] = v[k].numpy() if hasattr(v[k], 'numpy') else v[k]

                    batch_param_losses.append(data_dict)'''
                epoch_number_dict = {'epoch': epoch_number}

                batch_param_losses = [
                    {
                        'param': param,
                        **decode_event_name(param),
                        **epoch_number_dict,
                        'pred': pred.item(),
                        'label': max_label.item(),
                        'loss': loss.item(),
                        'accuracy': correct.item(),
                        #'is_train_regime': metrics['test_regime'][k] in train_sets, #replaced with groups in calculate_statistics
                        #'small_food_selected': small.item(),
                        #'big_food_selected': big.item(),
                        #'neither_food_selected': neither.item(),
                        **{x: v[k].numpy() if hasattr(v[k], 'numpy') else v[k] for x, v in metrics.items()}
                    }
                    #for k, (param, loss, correct, small, big, neither, pred) in enumerate(zip(params, losses, corrects, small_food_selected, big_food_selected, neither_food_selected, pred))
                    for k, (param, loss, correct, pred, max_label) in enumerate(zip(params, losses, corrects, pred, max_labels))
                ]

                param_losses_list.extend(batch_param_losses)

    print('correct', overall_correct, overall_total)

    fig = plot_accuracy_vs_vision(accuracy_results1, vision_probs, uncertainties)
    save_path = os.path.join(load_path, f'accuracy_plot_epoch{epoch_number}-{num_visions}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if False:
        final_awareness_results = []
        for prob in vision_probs:
            avg_result = {
                'visionProb': prob * 100,
                **{key: np.mean([r[key] for r in awareness_results[prob]])
                   for key in ['TT', 'TF', 'TN', 'FT', 'FF', 'FN', 'NT', 'NF', 'NN']},
                'uncertain-accuracy': np.mean(accuracy_results1[prob]),
                'certain-accuracy': np.mean(accuracy_results2[prob]),
            }
            final_awareness_results.append(avg_result)

        fig = plot_awareness_results(final_awareness_results)

        save_path = os.path.join(load_path, f'awareness_plot_epoch{epoch_number}-o.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        fig = plot_awareness_results(final_awareness_results, merge_states=True)
        save_path = os.path.join(load_path, f'awareness_plot_epoch{epoch_number}-m-o.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


    #print('returning without saving a csv!!!!!!')
    #return None
    # save dfs periodically to free up ram:
    df_paths = []
    os.makedirs(model_load_path, exist_ok=True)

    df = pd.DataFrame(param_losses_list)
    print('converting')
    df = df.convert_dtypes()
    print('done')
    # weak attempt at shortening the data
    df['loss'] = df['loss'].round(4)
    #df[['small_food_selected', 'big_food_selected', 'neither_food_selected']] =  df[['small_food_selected', 'big_food_selected', 'neither_food_selected']].astype(int)

    #int_columns = ['visible_baits', 'swaps', 'visible_swaps', 'first_bait_size', 'small_food_selected', 'big_food_selected', 'neither_food_selected', 'epoch']
    int_columns = ['visible_baits', 'swaps', 'visible_swaps', 'first_bait_size', 'epoch', 'pred']

    def convert_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    df['pred'] = df['pred'].apply(convert_tensor)

    # why is opponents an object

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    i_inf_str = df['i-informedness'].apply(lambda x: ''.join(map(str, x[-1])))
    unique_regimes = i_inf_str.unique()

    op_str = df['opponents'].astype(str)
    unique_ops = op_str.unique()

    '''print("Distributions by Regime:", df.keys())
    for regime in unique_regimes:
        for op in unique_ops:
            regime_data = df[(i_inf_str == regime) & (op_str == op)]

            pred_dist = np.bincount(regime_data['pred'].astype(int), minlength=5) / len(regime_data)
            accurate = regime_data['accuracy'].astype(float).mean()
            # Convert one-hot to indices for label distribution
            label_dist = np.bincount(regime_data['label'].astype(int), minlength=5) / len(regime_data)

            print(f"Regime {regime}, {op}:")
            print(f"Prediction distribution: {pred_dist}")
            print(f"Label distribution: {label_dist}")
            print(f"Accuracy: {accurate}")'''

    #float_columns = ['o_acc']
    #for col in float_columns:
    #    df[col] = df[col].astype('float16')

    curtime = time.time()
    print(df.info(memory_usage='deep'))

    print('saving csv...')
    # print('cols', df.columns)
    if use_prior:
        epoch_number = 'prior'


    df.to_csv(os.path.join(model_load_path, f'param_losses_{epoch_number}_{repetition}.csv'), index=False, compression='gzip')
    print('compressed write time (gzip):', time.time() - curtime)

    df_paths.append(os.path.join(model_load_path, f'param_losses_{epoch_number}_{repetition}.csv'))
    print('saving activations...')
    with open(os.path.join(model_load_path, f'activations_{epoch_number}_{repetition}.pkl'), 'wb') as f:
        pickle.dump(activation_data, f)
    return df_paths


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


def train_model_retrain(model, train_loader, test_loader, save_path,
                        model_kwargs=None,
                        oracle_labels=[],
                        repetition=0,
                        save_models=True,
                        record_loss=True,
                        oracle_is_target=False,
                        max_batches=10000,
                        tolerance=5e-4,
                        max_tries=3,
                        opt='sgd'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = model_kwargs['lr']
    old_weights = [param.data.clone() for param in model.parameters()]

    criterion = nn.CrossEntropyLoss()
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == 'momentum09':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    elif opt == 'nesterov09':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    t = tqdm.trange(max_batches)
    iter_loader = iter(train_loader)
    epoch_length = len(train_loader)
    epoch_losses_df = pd.DataFrame(columns=['Batch', 'Loss'])
    retrain_stats = pd.DataFrame(columns=['Batch', 'Validation Loss', 'Weight Distance'])
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    tries = 0
    total_loss = 0.0
    start_loss = 1000000
    breaking = False
    model.to(device)
    model.train()
    for batch in range(max_batches):
        try:
            inputs, target_labels, _, _, _ = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            inputs, target_labels, _, _, _ = next(iter_loader)
        inputs, target_labels = inputs.to(device), target_labels.to(device)

        outputs = model(inputs, None)
        loss = criterion(outputs, torch.argmax(target_labels, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t.update(1)

        if record_loss and ((batch % epoch_length == 0) or (batch == max_batches - 1)):
            total_loss += loss.item()

            scheduler.step()

            model.eval()
            with torch.inference_mode():
                test_loss = 0.0
                for inputs, target_labels, _, oracle_inputs, _ in test_loader:
                    inputs, target_labels, oracle_inputs = inputs.to(device), target_labels.to(device), oracle_inputs.to(device)
                    outputs = model(inputs, oracle_inputs)
                    loss = criterion(outputs, torch.argmax(target_labels, dim=1))
                    test_loss += loss.item()
                epoch_losses_df = epoch_losses_df.append({'Batch': batch, 'Loss': test_loss / len(test_loader)}, ignore_index=True)

            if batch % epoch_length == 0:
                avg_loss = test_loss / len(test_loader)

                if avg_loss >= start_loss - tolerance:
                    tries += 1
                    # start_loss = avg_loss - tolerance
                    print(f"Tries incremented: {tries}, Start Loss updated: {start_loss}, avg_loss was: {avg_loss}")
                    if tries >= max_tries:
                        breaking = True
                        print("break")
                else:
                    start_loss = avg_loss - tolerance
                    print('reset tries', start_loss)
                    tries = 0

                if save_models:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-{batch // epoch_length}.pt'))

            model.train()

        if breaking:
            new_weights = [param.data.clone() for param in model.parameters()]
            weight_dist = weight_distance(old_weights, new_weights)
            retrain_stats = retrain_stats.append({'Batch': (batch // epoch_length) - tries, 'Validation Loss': avg_loss, 'Weight Distance': weight_dist}, ignore_index=True)
            break

    retrain_stats.to_csv(os.path.join(save_path, f'{opt}-stats-{repetition}.csv'), index=False)
    if record_loss:
        epoch_losses_df.to_csv(os.path.join(save_path, f'{opt}-losses-{repetition}.csv'), index=False)


def train_model(train_sets, target_label, load_path='supervised/', save_path='', epochs=100,
                model_kwargs=None, model_type=None,
                oracle_labels=[], repetition=0,
                save_models=True, save_every=5, record_loss=True,
                oracle_is_target=False, batches=5000, last_timestep=True,
                seed=0, test_percent=0.2):
    use_cuda = torch.cuda.is_available()
    print(torch.cuda.is_available())
    if len(oracle_labels) == 0 or oracle_labels[0] == None:
        oracle_labels = []
    data, labels, params, module_data_combined = [], [], [], []
    module_labels = {
        #'treat_perception': ['loc-large', 'loc-small'],  
        #'vision_perception': 'vision',
        'presence_perception': 'opponents',
        'my_combiner': ['loc-large', 'loc-small'],  
        'op_combiner': ['b-loc-large', 'b-loc-small'],  
        'my_decision': 'correct-loc',
        'op_decision': 'target-loc'
    }
    module_label_data = {module: [] for module in module_labels.keys()} 
    batch_size = model_kwargs['batch_size']
    model_kwargs['batch_size'] = 1024
    lr = model_kwargs['lr']
    batch_size = model_kwargs['batch_size']

    print('lr:', lr, 'bs:', batch_size)
    

    for data_name in train_sets:
        dir = os.path.join(load_path, data_name)
        data.append(np.load(os.path.join(dir, 'obs.npz'), mmap_mode='r')['arr_0'])
        
        # Load the main target label
        labels_raw = np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0']
        if target_label == 'shouldGetBig':
            if last_timestep:
                x = np.eye(2)[labels_raw[:, -1].astype(int)]  # single timestep
            else:
                x = np.eye(2)[labels_raw.astype(int)].reshape(-1, 10)  # 5 timesteps
            labels.append(x)
        elif len(labels_raw.shape) > 2:
            if last_timestep:
                labels.append(labels_raw[..., -1, :])  # use only the last timestep
            else:
                labels.append(labels_raw.reshape(-1, 25))
        else:
            #print(labels_raw.shape, labels_raw[0])
            labels.append(labels_raw.reshape(-1, 25))
            
        params.append(np.load(os.path.join(dir, 'params.npz'), mmap_mode='r')['arr_0'])
        
        data_arrays = []
        for module, label_list in module_labels.items():
            if not isinstance(label_list, list):
                label_list = [label_list]
                
            for label_name in label_list:
                try:
                    label_data = np.load(os.path.join(dir, f'label-{label_name}.npz'), mmap_mode='r')['arr_0']
                    if last_timestep and len(label_data.shape) > 2:
                        label_data = label_data[..., -1, :5]
                    elif len(label_data.shape) > 2:
                        label_data = label_data.reshape(label_data.shape[0], -1)
                    if label_data.ndim == 1:
                        label_data = label_data.reshape(label_data.shape[0], 1)
                        
                    data_arrays.append(label_data)
                    #print('y', label_name, label_data.shape)
                except Exception as e:
                    print(f"Error loading {label_name}: {e}")
        
        if data_arrays:
            try:
                combined = np.concatenate(data_arrays, axis=1)
                module_data_combined.append(combined)
                #print('xxx', module, combined.shape)
            except Exception as e:
                print(f"Error concatenating arrays: {e}")
                module_data_combined.append(np.zeros((len(data[-1]), 0)))
        else:
            module_data_combined.append(np.zeros((len(data[-1]), 0)))

    included_indices = filter_indices(data, labels, params, module_data_combined, is_train=True, test_percent=test_percent)
    train_dataset = TrainDatasetBig(data, labels, params, module_data_combined, included_indices)
    del data, labels, params, module_data_combined

    if record_loss:
        #print('recording loss')
        train_size = int(0.9 * len(train_dataset))
        print('epochs:', train_size // batch_size)
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42+repetition))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  # can't use more on windows

    #print('oracle length:', model_kwargs['oracle_len'])
    model_kwargs['output_len'] = 5  # np.prod(labels.shape[1:])
    model_kwargs['channels'] = 5  # np.prod(params.shape[2])
    model_kwargs['oracle_is_target'] = oracle_is_target

    device = torch.device('cuda' if use_cuda else 'cpu')
    total_steps = 10

    model = load_model(model_type, model_kwargs, device)

    #criterion = torchvision.ops.sigmoid_focal_loss
    criterion = nn.CrossEntropyLoss()
    oracle_criterion = nn.MSELoss(reduction='mean')
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    #model.vision_prob = 0.8

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.95, 0.999))
    sigmoid_scheduler = SigmoidTempScheduler(model, start_temp=90.0, end_temp=90.0, total_steps=total_steps, 
        vision_prob_start=model.vision_prob, vision_prob_end=model.vision_prob_base, rate=5.0)
    #scheduler = ExponentialLR(optimizer, gamma=0.92)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-2,
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=1,
        final_div_factor=5,
    )

    if False:  # if loading previous model, only for progressions
        last_digit = int(save_path[-1])
        print(last_digit)
        if last_digit > 0:
            save_path2 = save_path[:-1] + str(last_digit - 1)
            filename = os.path.join(save_path2, f'{repetition}-model_epoch{epochs - 1}.pt')
            loaded_model_info = torch.load(filename, map_location=device)
            loaded_model_kwargs, loaded_model_state_dict = loaded_model_info
            model.load_state_dict(loaded_model_state_dict)

    epoch_length = batches // total_steps

    t = tqdm.trange(batches)
    iter_loader = iter(train_loader)
    epoch_losses_df = pd.DataFrame(columns=['Batch', 'Loss', 'Accuracy'] + [f"{module}_mse" for module in module_labels.keys()])

    if save_models:
        os.makedirs(save_path, exist_ok=True)
        torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-prior.pt'))

    for batch in range(batches):
        total_loss = 0.0
        try:
            inputs, target_labels, _, oracles, _ = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            inputs, target_labels, _, oracles, _ = next(iter_loader)
        inputs, target_labels, oracles = inputs.to(device), target_labels.float().to(device), oracles.to(device)
        target_labels = target_labels[:,:-1]
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        outputs = model(inputs, None)
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
        #print('xxxxxxx', outputs)
        #loss = criterion(outputs['my_decision'], target_labels, reduction="mean") # used for Focal
        loss = criterion(outputs['my_decision'], torch.argmax(target_labels, dim=1)) # used for bcd
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        #t.update(1)

        if (batch+1) % epoch_length == 0:
            if batch > 1:
                scheduler.step()
                sigmoid_scheduler.step()
                t.update(epoch_length)

        if record_loss and (((batch) % epoch_length == 0) or (batch == batches - 1)):
            total_loss += loss.item()
            model.eval()
            test_losses = []
            all_preds = []
            all_targets = []

            with torch.inference_mode():
                module_mse_values = {module: [] for module in module_labels.keys()}
                module_ranges = {}
                first_batch_processed = False
                for inputs, target_labels, _, oracle_inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    target_labels = target_labels.float().to(device)[:,:-1]
                    oracle_inputs = oracle_inputs.to(device)

                    outputs = model(inputs, None)
                    loss = criterion(outputs['my_decision'], torch.argmax(target_labels, dim=1))
                    #loss = criterion(outputs['my_decision'], target_labels, reduction="mean") # used for Focal
                    test_losses.append(loss.item())

                    if not first_batch_processed:
                        flat_oracle = oracles.reshape(oracles.size(0), -1)
                        total_oracle_dims = flat_oracle.size(1)
                        
                        pos = 0
                        for module_name in module_labels.keys():
                            module_output = outputs[module_name]
                            flattened_output_size = module_output.reshape(module_output.size(0), -1)[:, :5].size(1)
                            module_ranges[module_name] = (pos, pos + flattened_output_size)
                            pos += flattened_output_size
                        first_batch_processed = True

                    for module_name in module_labels.keys():
                        start_idx, end_idx = module_ranges[module_name]
                            
                        end_idx = min(end_idx, flat_oracle.size(1))
                        
                        module_target = flat_oracle[:, start_idx:end_idx]
                        
                        module_output = outputs[module_name].reshape(outputs[module_name].size(0), -1)[:, :5]

                        # I am manually cutting all at 5 because that's how beliefs are
                        
                        if module_output.size(0) != module_target.size(0):
                            min_dim = min(module_output.size(0), module_target.size(0))
                            module_output = module_output[:min_dim, :]
                            module_target = module_target[:min_dim, :]

                        #print(module_name, module_output.shape, module_target.shape)
                            
                        module_mse = oracle_criterion(module_output, module_target).item()
                        module_mse_values[module_name].append(module_mse)

                    all_preds.append(outputs['my_decision'].argmax(dim=1))
                    all_targets.append(torch.argmax(target_labels, dim=1))

            test_loss = sum(test_losses) / len(test_losses)
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            accuracy = (all_preds == all_targets).float().mean().item()

            new_row_data = {
                'Batch': batch,
                'Loss': test_loss,
                'Accuracy': accuracy
            }
            for module, mse_val in module_mse_values.items():
                new_row_data[f"{module}_mse"] = np.mean(mse_val)
            new_row = pd.DataFrame([new_row_data])
            
            epoch_losses_df = pd.concat([epoch_losses_df, new_row], ignore_index=True)
            model.train()

            print(f"Accuracy: {accuracy:.4f}, Loss: {test_loss:.4f}, Vision:", model.vision_prob, 'Sigmoid:', model.treat_perception.sigmoid_temp)
            #print("Module MSE values:")
            #for module, mse_val in module_mse_values.items():
            #    print(f"  {module}: {np.mean(mse_val):.4f}")

            if save_models and batch == batches - 1:
                os.makedirs(save_path, exist_ok=True)
                torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-{batch // epoch_length}.pt'))


    if record_loss:
        epoch_losses_df.to_csv(os.path.join(save_path, f'losses-{repetition}.csv'), index=False)


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


def calculate_statistics(df, last_epoch_df, params, skip_3x=True, skip_2x1=False, key_param=None, skip_1x=True, record_delta_pi=False, used_regimes=None, savepath=None, last_timestep=True):
    '''
    Calculates various statistics from datasets of model outputs detailing model performance.
    '''
    avg_loss = {}
    variances = {}
    ranges_1 = {}
    ranges_2 = {}
    range_dict = {}
    range_dict3 = {}

    check_labels = ['p-s-0', 'target', 'delay', 'b-loc', 'p-b-0', 'p-b-1', 'p-s-1', 'shouldAvoidSmall', 'shouldGetBig', 'vision', 'loc']

    sub_regime_keys = [
        "Nn", "Fn", "Nf", "Tn", "Nt", "Ff", "Tf", "Ft", "Tt"
    ]

    

    direct = [x + '1' for x in sub_regime_keys]
    progression_groups = {}
    progression_groups['s1'] = [x + '0' for x in sub_regime_keys]
    progression_groups['s2'] = ['Tt1']
    progression_groups['s21'] = ['Nn1a', 'Nt1a'] 
    progression_groups['s3'] = [x for x in direct if x != 'Tt1' and x not in ['Nn1a', 'Nt1a']] + ['Nn1b', 'Nt1b']

    group_mapping = {}
    for group_name, regimes_list in progression_groups.items():
        for regime in regimes_list:
            group_mapping[regime] = group_name

    print("All test_regime values:", last_epoch_df['test_regime'].unique())
    print("Mapping keys:", list(group_mapping.keys()))
    

    last_epoch_df['test_group'] = last_epoch_df['test_regime'].apply(
        lambda x: group_mapping.get(x, '?')
    )

    params.append('test_group')


    print('calculating statistics...')
    for col in last_epoch_df.columns:
        if col in check_labels:
            last_epoch_df[col] = last_epoch_df[col].apply(extract_last_value_from_list_str)

    print('making categorical')
    for param in params:
        if last_epoch_df[param].dtype == 'object':
            try:
                last_epoch_df[param] = last_epoch_df[param].astype('category')
            except TypeError:
                try:
                    last_epoch_df[param] = last_epoch_df[param].apply(tuple)
                    last_epoch_df[param] = last_epoch_df[param].astype('category')
                except TypeError:
                    print('double error on', param)
                    pass
                pass
    params = [param for param in params if param not in ['delay', 'perm']]

    # print('params:', params)

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
    if used_regimes:
        print('calculating regime size')
        regime_lengths = {}
        grouped = last_epoch_df.groupby(['regime'])['accuracy'].mean()
        print(grouped)
        for regime_item in used_regimes:
            regime_length = 0
            for data_name in regime_item:
                pathy = os.path.join('supervised/', data_name, 'params.npz')
                regime_items = np.load(pathy, mmap_mode='r')['arr_0']
                regime_length += len(regime_items)
            regime_lengths[regime_item[0][3:]] = regime_length
            print(regime_item, regime_length)
        if False:
            plot_regime_lengths(regime_lengths, grouped, savepath + 'scatter.png')

    unique_vals = {param: last_epoch_df[param].unique() for param in params}
    print('found unique vals', unique_vals)
    for par, val in unique_vals.items():
        print(par, type(val[0]))
    '''if last_timestep:
        print('reducing timesteps')
        for param in unique_vals:

            last_epoch_df[param] = last_epoch_df[param].apply(get_last_timestep)'''
            #This is redundant with above extract part and also doesn't work properly.
    unique_vals = {param: last_epoch_df[param].unique() for param in params}
    print('new unique vals', unique_vals)

    if not skip_1x:
        print('calculating single params')

        for param in params:
            avg_loss[param] = df.groupby([param, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
            means = last_epoch_df.groupby([param]).mean(numeric_only=True)
            ranges_1[param] = means['accuracy'].max() - means['accuracy'].min()
        print('calculating double params')

        for param1, param2 in tqdm.tqdm(param_pairs):
            avg_loss[(param1, param2)] = df.groupby([param1, param2, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()

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
            print(param)
            if param != key_param:
                # Initializing a nested dictionary for each unique key_param value
                for acc_type, save_dict in zip(['accuracy'], [key_param_stats]): #zip(['accuracy', 'o_acc'], [key_param_stats, oracle_key_param_stats]):
                    #print('SSSSSSSSSS', unique_vals) #unique vals of regime has only 6 of them
                    for key_val in unique_vals[key_param]:
                        subset = last_epoch_df[last_epoch_df[key_param] == key_val]

                        grouped = subset.groupby(['repetition', param])[acc_type]
                        repetition_means = grouped.mean()
                        overall_means = repetition_means.groupby(level=param).mean()
                        means_std = repetition_means.groupby(level=param).std()

                        Q1 = grouped.quantile(0.25).groupby(level=param).mean()
                        Q3 = grouped.quantile(0.75).groupby(level=param).mean()
                        counts = grouped.size()

                        z_value = 1.96  # For a 95% CI
                        standard_errors = (z_value * np.sqrt(repetition_means * (1 - repetition_means) / counts)).groupby(level=param).mean()

                        #print(key_val, param, 'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')

                        if key_val not in save_dict:
                            save_dict[key_val] = {}
                        save_dict[key_val][param] = {
                            'mean': overall_means.to_dict(),
                            'std': means_std.to_dict(),
                            'q1': Q1.to_dict(),
                            'q3': Q3.to_dict(),
                            'ci': standard_errors.to_dict(),
                        }
                        # dict order is key_val > param > mean/std > param_val

                        #print('test group', key_val)
                        #print("Unique test_regime values:", subset['test_regime'].unique())

                        for group_name in ['s1', 's2', 's21', 's3']:
                            subset2 = subset[subset['test_group'] == group_name]
                            #print(group_name, len(subset2))
                            if len(subset) > 0:
                                grouped = subset2.groupby(['repetition', param])[acc_type]
                                repetition_means = grouped.mean()

                                if key_val not in save_dict:
                                    save_dict[key_val] = {}
                                save_dict[key_val][group_name] = {
                                    'mean': repetition_means.groupby(level=param).mean().to_dict(),
                                    'std': repetition_means.groupby(level=param).std().to_dict(),
                                    'q1': grouped.quantile(0.25).groupby(level=param).mean().to_dict(),
                                    'q3': grouped.quantile(0.75).groupby(level=param).mean().to_dict(),
                                    'ci': (z_value * np.sqrt(repetition_means * (1 - repetition_means) / grouped.size())).groupby(level=param).mean().to_dict(),
                                }

        if record_delta_pi:
            delta_pi_stats(unique_vals, key_param, last_epoch_df, delta_operator_summary, df_summary)

    #print(key_param_stats['a-mix-r-perception-100-loc-s2'].keys())
    #exit()

    return avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats, oracle_key_param_stats, df_summary, delta_operator_summary


def delta_pi_stats(unique_vals, key_param, last_epoch_df, delta_operator_summary, df_summary):
    # todo: if delay_2nd_bait or first_swap are na, just make them 0? shouldn't affect anything
    print('calculating delta preds')

    set1 = ['T', 'F', 'N']
    set2 = ['t', 'f', 'n']
    set3 = ['0', '1']
    combinations = list(itertools.product(set1, set2, set3))
    combinations_str = [''.join(combo) for combo in combinations]

    operators = ['T-F', 't-f', 'F-N', 'f-n', 'T-N', 't-n', '1-0']
    required_columns = [f'pred_{i}' for i in range(5)]
    perm_keys = ['p-b-0', 'p-b-1', 'p-s-0', 'p-s-1', 'delay_2nd_bait', 'first_swap', 'first_bait_size', 'delay']

    for key_val in unique_vals[key_param]:  # for each train regime, etc
        unique_repetitions = last_epoch_df['repetition'].unique()

        delta_mean_rep = {key: [] for key in operators}
        delta_std_rep = {key: [] for key in operators}
        delta_mean_correct_rep = {key: [] for key in operators}
        delta_std_correct_rep = {key: [] for key in operators}
        delta_mean_accurate_rep = {key: [] for key in operators}
        delta_std_accurate_rep = {key: [] for key in operators}

        delta_mean_p_t_rep = {key: [] for key in operators}
        delta_mean_p_f_rep = {key: [] for key in operators}
        delta_mean_m_t_rep = {key: [] for key in operators}
        delta_mean_m_f_rep = {key: [] for key in operators}

        delta_mean_p_t_t_rep = {key: [] for key in operators}
        delta_mean_p_t_f_rep = {key: [] for key in operators}
        delta_mean_p_f_t_rep = {key: [] for key in operators}
        delta_mean_p_f_f_rep = {key: [] for key in operators}

        delta_mean_m_t_t_rep = {key: [] for key in operators}
        delta_mean_m_t_f_rep = {key: [] for key in operators}
        delta_mean_m_f_t_rep = {key: [] for key in operators}
        delta_mean_m_f_f_rep = {key: [] for key in operators}

        delta_mean = [{key: [] for key in operators} for _ in unique_repetitions]
        delta_mean_correct = [{key: [] for key in operators} for _ in unique_repetitions]
        delta_mean_accurate = [{key: [] for key in operators} for _ in unique_repetitions]

        dpred_mean_p_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_p_f = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_f = [{key: [] for key in operators} for _ in unique_repetitions]

        dpred_mean_p_t_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_p_t_f = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_p_f_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_p_f_f = [{key: [] for key in operators} for _ in unique_repetitions]

        dpred_mean_m_t_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_t_f = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_f_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_f_f = [{key: [] for key in operators} for _ in unique_repetitions]

        for rep in unique_repetitions:
            subset_rep = last_epoch_df[last_epoch_df['repetition'] == rep]

            # print(subset_rep.columns, key_param, key_val)
            subset = subset_rep[subset_rep[key_param] == key_val]
            subset['pred'] = subset['pred'].apply(convert_to_numeric).astype(np.int8)
            conv = pd.concat([subset, pd.get_dummies(subset['pred'], prefix='pred')], axis=1)

            for col in required_columns:  # a model might not predict all 5 values
                if col not in conv.columns:
                    conv[col] = 0
            subset = conv[required_columns + ['i-informedness', 'correct-loc', 'opponents', 'accuracy'] + perm_keys]

            # OPERATOR PART
            for op in operators:

                delta_preds = []
                delta_preds_correct = []
                delta_preds_accurate = []

                dpred_p_t = []
                dpred_p_f = []
                dpred_m_t = []
                dpred_m_f = []

                dpred_p_t_t = []
                dpred_p_t_f = []
                dpred_p_f_t = []
                dpred_p_f_f = []

                dpred_m_t_t = []
                dpred_m_t_f = []
                dpred_m_f_t = []
                dpred_m_f_f = []

                for key in combinations_str:
                    # key is 1st position, descendents are 2nd
                    descendants = get_descendants(key, op, combinations_str)

                    mapping = {'T': 2, 'F': 1, 'N': 0, 't': 2, 'f': 1, 'n': 0, '0': 0, '1': 1}

                    key = [mapping[char] for char in key]
                    key_informedness = '[' + ' '.join(map(str, key[:2])) + ']'
                    key_opponents = np.float64(key[2])
                    if '0' not in op and key_opponents == 0:
                        # for most operators, we only use cases with opponents present
                        continue

                    descendants = [[mapping[char] for char in descendant] for descendant in descendants]
                    descendant_informedness = ['[' + ' '.join(map(str, descendant[:2])) + ']' for descendant in
                                               descendants]
                    descendant_opponents = [np.float64(descendant[2]) for descendant in descendants]

                    if len(descendants) < 1:
                        continue

                    inf = subset[
                        (subset['i-informedness'] == key_informedness) & (subset['opponents'] == key_opponents)].groupby(
                        perm_keys + ['i-informedness', 'opponents', 'correct-loc'],
                        observed=True).mean().reset_index()
                    noinf = subset[(subset['i-informedness'].isin(descendant_informedness)) &
                                   (subset['opponents'].isin(descendant_opponents))].groupby(
                        perm_keys + ['i-informedness', 'opponents', 'correct-loc'],
                        observed=True).mean().reset_index()
                    # print('lens1', len(inf), len(noinf), len(subset))

                    merged_df = pd.merge(
                        inf,
                        noinf,
                        on=perm_keys,
                        suffixes=('_m', ''),
                        how='inner',
                    )

                    merged_df['changed_target'] = (
                            merged_df['correct-loc_m'] != merged_df['correct-loc']).astype(int)

                    for i in range(5):
                        merged_df[f'pred_diff_{i}'] = abs(merged_df[f'pred_{i}_m'] - merged_df[f'pred_{i}'])
                    merged_df['total_pred_diff'] = merged_df[[f'pred_diff_{idx}' for idx in range(5)]].sum(axis=1) / 2
                    delta_preds.extend(merged_df['total_pred_diff'].tolist())
                    merged_df['total_pred_diff_correct'] = 1 - abs(
                        merged_df['total_pred_diff'] - merged_df['changed_target'])
                    delta_preds_correct.extend(merged_df['total_pred_diff_correct'].tolist())
                    merged_df['total_pred_diff_accurate'] = merged_df['total_pred_diff_correct'] * merged_df[
                        'accuracy'] * merged_df['accuracy_m']
                    delta_preds_accurate.extend(merged_df['total_pred_diff_accurate'].tolist())

                    merged_df['changed_target'] = merged_df['changed_target'].astype(bool)

                    merged_df['dpred_p_t'] = (merged_df['changed_target'] == 1) * (merged_df['total_pred_diff'])
                    merged_df['dpred_p_f'] = (merged_df['changed_target'] == 1) * (1 - merged_df['total_pred_diff'])
                    merged_df['dpred_m_t'] = (merged_df['changed_target'] == 0) * (merged_df['total_pred_diff'])
                    merged_df['dpred_m_f'] = (merged_df['changed_target'] == 0) * (1 - merged_df['total_pred_diff'])

                    merged_df['total_pred_diff_p_T_T'] = (merged_df['changed_target']) * (merged_df['accuracy_m']) * (
                        merged_df['accuracy'])
                    merged_df['total_pred_diff_p_T_F'] = (merged_df['changed_target']) * (merged_df['accuracy_m']) * (
                            1 - merged_df['accuracy'])
                    merged_df['total_pred_diff_p_F_T'] = (merged_df['changed_target']) * (
                            1 - merged_df['accuracy_m']) * (merged_df['accuracy'])
                    merged_df['total_pred_diff_p_F_F'] = (merged_df['changed_target']) * (
                            1 - merged_df['accuracy_m']) * (1 - merged_df['accuracy'])

                    merged_df['total_pred_diff_m_T_T'] = (1 - merged_df['changed_target']) * (
                        merged_df['accuracy_m']) * (merged_df['accuracy'])
                    merged_df['total_pred_diff_m_T_F'] = (1 - merged_df['changed_target']) * (
                        merged_df['accuracy_m']) * (1 - merged_df['accuracy'])
                    merged_df['total_pred_diff_m_F_T'] = (1 - merged_df['changed_target']) * (
                            1 - merged_df['accuracy_m']) * (merged_df['accuracy'])
                    merged_df['total_pred_diff_m_F_F'] = (1 - merged_df['changed_target']) * (
                            1 - merged_df['accuracy_m']) * (1 - merged_df['accuracy'])

                    # print(op, key, descendants, np.mean(merged_df['changed_target']))
                    dpred_p_t.extend(merged_df['dpred_p_t'].tolist())
                    dpred_p_f.extend(merged_df['dpred_p_f'].tolist())
                    dpred_m_t.extend(merged_df['dpred_m_t'].tolist())
                    dpred_m_f.extend(merged_df['dpred_m_f'].tolist())

                    dpred_p_t_t.extend(merged_df['total_pred_diff_p_T_T'].tolist())
                    dpred_p_t_f.extend(merged_df['total_pred_diff_p_T_F'].tolist())
                    dpred_p_f_t.extend(merged_df['total_pred_diff_p_F_T'].tolist())
                    dpred_p_f_f.extend(merged_df['total_pred_diff_p_F_F'].tolist())
                    dpred_m_t_t.extend(merged_df['total_pred_diff_m_T_T'].tolist())
                    dpred_m_t_f.extend(merged_df['total_pred_diff_m_T_F'].tolist())
                    dpred_m_f_t.extend(merged_df['total_pred_diff_m_F_T'].tolist())
                    dpred_m_f_f.extend(merged_df['total_pred_diff_m_F_F'].tolist())

                # first level aggregate: for each operator, within one repetition
                r = int(rep)
                delta_mean[r][op] = np.mean(delta_preds)
                delta_mean_correct[r][op] = np.mean(delta_preds_correct)
                delta_mean_accurate[r][op] = np.mean(delta_preds_accurate)

                dpred_mean_p_t[r][op] = np.mean(dpred_p_t)
                dpred_mean_p_f[r][op] = np.mean(dpred_p_f)
                dpred_mean_m_t[r][op] = np.mean(dpred_m_t)
                dpred_mean_m_f[r][op] = np.mean(dpred_m_f)

                dpred_mean_p_t_t[r][op] = np.mean(dpred_p_t_t)
                dpred_mean_p_t_f[r][op] = np.mean(dpred_p_t_f)
                dpred_mean_p_f_t[r][op] = np.mean(dpred_p_f_t)
                dpred_mean_p_f_f[r][op] = np.mean(dpred_p_f_f)

                dpred_mean_m_t_t[r][op] = np.mean(dpred_m_t_t)
                dpred_mean_m_t_f[r][op] = np.mean(dpred_m_t_f)
                dpred_mean_m_f_t[r][op] = np.mean(dpred_m_f_t)
                dpred_mean_m_f_f[r][op] = np.mean(dpred_m_f_f)

        # second level aggregate: over one repetition

        for op in operators:
            op_values_mean = [rep_dict[op] for rep_dict in delta_mean]
            op_values_mean_correct = [rep_dict[op] for rep_dict in delta_mean_correct]
            op_values_mean_accurate = [rep_dict[op] for rep_dict in delta_mean_accurate]

            op_values_mean_pt = [rep_dict[op] for rep_dict in dpred_mean_p_t]
            op_values_mean_pf = [rep_dict[op] for rep_dict in dpred_mean_p_f]
            op_values_mean_mt = [rep_dict[op] for rep_dict in dpred_mean_m_t]
            op_values_mean_mf = [rep_dict[op] for rep_dict in dpred_mean_m_f]

            op_values_mean_ptt = [rep_dict[op] for rep_dict in dpred_mean_p_t_t]
            op_values_mean_ptf = [rep_dict[op] for rep_dict in dpred_mean_p_t_f]
            op_values_mean_pft = [rep_dict[op] for rep_dict in dpred_mean_p_f_t]
            op_values_mean_pff = [rep_dict[op] for rep_dict in dpred_mean_p_f_f]
            op_values_mean_mtt = [rep_dict[op] for rep_dict in dpred_mean_m_t_t]
            op_values_mean_mtf = [rep_dict[op] for rep_dict in dpred_mean_m_t_f]
            op_values_mean_mft = [rep_dict[op] for rep_dict in dpred_mean_m_f_t]
            op_values_mean_mff = [rep_dict[op] for rep_dict in dpred_mean_m_f_f]

            delta_mean_rep[op] = np.mean(op_values_mean)
            delta_std_rep[op] = np.std(op_values_mean)

            delta_mean_correct_rep[op] = np.mean(op_values_mean_correct)
            delta_std_correct_rep[op] = np.std(op_values_mean_correct)

            delta_mean_accurate_rep[op] = np.mean(op_values_mean_accurate)
            delta_std_accurate_rep[op] = np.std(op_values_mean_accurate)

            delta_mean_p_t_rep[op] = np.mean(op_values_mean_pt)
            delta_mean_p_f_rep[op] = np.mean(op_values_mean_pf)
            delta_mean_m_t_rep[op] = np.mean(op_values_mean_mt)
            delta_mean_m_f_rep[op] = np.mean(op_values_mean_mf)

            delta_mean_p_t_t_rep[op] = np.mean(op_values_mean_ptt)
            delta_mean_p_t_f_rep[op] = np.mean(op_values_mean_ptf)
            delta_mean_p_f_t_rep[op] = np.mean(op_values_mean_pft)
            delta_mean_p_f_f_rep[op] = np.mean(op_values_mean_pff)
            delta_mean_m_t_t_rep[op] = np.mean(op_values_mean_mtt)
            delta_mean_m_t_f_rep[op] = np.mean(op_values_mean_mtf)
            delta_mean_m_f_t_rep[op] = np.mean(op_values_mean_mft)
            delta_mean_m_f_f_rep[op] = np.mean(op_values_mean_mff)

        # third level aggregate: all key_vals
        delta_operator_summary[key_val] = pd.DataFrame({
            'operator': list(delta_mean_rep.keys()),
            'dpred': [f"{delta_mean_rep[key]:.2f} ({delta_std_rep[key]:.2f})" for key in delta_mean_rep.keys()],
            'dpred_correct': [f"{delta_mean_correct_rep[key]:.2f} ({delta_std_correct_rep[key]:.2f})" for key in
                              delta_mean_correct_rep.keys()],
            'dpred_accurate': [f"{delta_mean_accurate_rep[key]:.2f} ({delta_std_accurate_rep[key]:.2f})" for key in
                               delta_mean_accurate_rep.keys()],
            'ptt': [delta_mean_p_t_t_rep[key] for key in delta_mean_p_t_t_rep.keys()],
            'ptf': [delta_mean_p_t_f_rep[key] for key in delta_mean_p_t_f_rep.keys()],
            'pft': [delta_mean_p_f_t_rep[key] for key in delta_mean_p_f_t_rep.keys()],
            'pff': [delta_mean_p_f_f_rep[key] for key in delta_mean_p_f_f_rep.keys()],
            'mtt': [delta_mean_m_t_t_rep[key] for key in delta_mean_m_t_t_rep.keys()],
            'mtf': [delta_mean_m_t_f_rep[key] for key in delta_mean_m_t_f_rep.keys()],
            'mft': [delta_mean_m_f_t_rep[key] for key in delta_mean_m_f_t_rep.keys()],
            'mff': [delta_mean_m_f_f_rep[key] for key in delta_mean_m_f_f_rep.keys()],
            'pt': [delta_mean_p_t_rep[key] for key in delta_mean_p_t_rep.keys()],
            'pf': [delta_mean_p_f_rep[key] for key in delta_mean_p_f_rep.keys()],
            'mt': [delta_mean_m_t_rep[key] for key in delta_mean_m_t_rep.keys()],
            'mf': [delta_mean_m_f_rep[key] for key in delta_mean_m_f_rep.keys()],
        })

        # print('do', key_val, delta_operator_summary[key_val])

        # CATEGORY ONE

        # for col in perm_keys + ['informedness', 'correct-loc']:
        #    print(f"{col} has {subset[col].nunique()} unique values:", subset[col].unique())

        inf = subset[subset['i-informedness'] == 'Tt'].groupby(perm_keys + ['i-informedness', 'correct-loc'],
                                                               observed=True).mean().reset_index()
        noinf = subset[subset['i-informedness'] != 'Tt'].groupby(perm_keys + ['i-informedness', 'correct-loc'],
                                                                 observed=True).mean().reset_index()

        merged_df = pd.merge(
            inf,
            noinf,
            on=perm_keys,
            suffixes=('_m', ''),
            how='inner',
        )

        merged_df['changed_target'] = (merged_df['correct-loc_m'] != merged_df['correct-loc']).astype(int)

        for i in range(5):
            merged_df[f'pred_diff_{i}'] = abs(merged_df[f'pred_{i}_m'] - merged_df[f'pred_{i}'])
        merged_df['total_pred_diff'] = merged_df[[f'pred_diff_{idx}' for idx in range(5)]].sum(axis=1) / 2
        merged_df['total_pred_diff_correct'] = 1 - abs(merged_df['total_pred_diff'] - merged_df['changed_target'])

        delta_preds = {}
        delta_preds_correct = {}

        for key in merged_df['i-informedness'].unique():
            delta_preds[key] = merged_df.loc[merged_df['i-informedness'] == key, 'total_pred_diff'].tolist()
            delta_preds_correct[key] = merged_df.loc[
                merged_df['i-informedness'] == key, 'total_pred_diff_correct'].tolist()

        delta_mean = {key: np.mean(val) for key, val in delta_preds.items()}
        delta_std = {key: np.std(val) for key, val in delta_preds.items()}

        delta_mean_correct = {key: np.mean(val) for key, val in delta_preds_correct.items()}
        delta_std_correct = {key: np.std(val) for key, val in delta_preds_correct.items()}

        delta_mean_accurate = {key: np.mean(val) for key, val in delta_preds_correct.items()}
        delta_std_accurate = {key: np.std(val) for key, val in delta_preds_correct.items()}

        df_summary[key_val] = pd.DataFrame({
            'Informedness': list(delta_mean.keys()),
            'dpred': [f"{delta_mean[key]} ({delta_std[key]})" for key in delta_mean.keys()],
            'dpred_correct': [f"{delta_mean_correct[key]} ({delta_std_correct[key]})" for key in
                              delta_mean_correct.keys()]
        })

        # print(df_summary[key_val])


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

                group['is_novel_task'] = ~test_regimes.isin(train_map[train_regime]) if 'hard' not in key else False
                mean_novel_accuracy = group.loc[group['is_novel_task'], 'accuracy'].mean()
                std_novel_accuracy = group.loc[group['is_novel_task'], 'accuracy'].std()
                f.write(f"Novel acc {key_param}: {key}: {mean_novel_accuracy} ({std_novel_accuracy})\n")

                group['isnt_novel_task'] = test_regimes.isin(train_map[train_regime]) if 'hard' not in key else True
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
                           last_timestep=True, model_type=None, do_retrain_model=True):
    '''
    Runs a session of supervised learning. Different steps, such as whether we train+save models, evaluate models, or run statistics on evaluations, are optional.

    '''
    params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
              'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
              'uninformed_bait', 'uninformed_swap', 'first_swap']

    test_names = []

    num_random_tests = 1

    test = 0
    while test < num_random_tests:
        try:
            model_kwargs = {"batch_size": 1024, "oracle_early": oracle_early, "hidden_size": 32, "num_layers": 3, "kernels": 16, "kernel_size1": 3, "kernel_size2": 5, "stride1": 1, "pool_kernel_size": 3, "pool_stride": 1, "padding1": 1, "padding2": 1, "use_pool": False, "use_conv2": False, "kernels2": 16, "batch_size": 256, "lr": 0.0003, "oracle_len": 0, "output_len": 5, "channels": 5}

            model_name = "".join([str(x) + "," for x in model_kwargs.values()])

            dfs_paths = []
            last_epoch_df_paths = []
            loss_paths = []

            if do_retrain_model:
                retrain_path = save_path + '-retrain'
                os.makedirs(retrain_path, exist_ok=True)
            for repetition in range(repetitions):
                if not skip_train:
                    train_model(train_sets, label, load_path=load_path, model_type=model_type,
                                save_path=save_path, epochs=epochs, batches=batches, model_kwargs=model_kwargs,
                                oracle_labels=oracle_labels, repetition=repetition, save_every=save_every,
                                oracle_is_target=oracle_is_target, last_timestep=last_timestep)
                loss_paths.append(os.path.join(save_path, f'losses-{repetition}.csv'))
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
                loss_file_pattern = os.path.join(save_path, 'losses-*.csv')
                all_loss_paths = glob.glob(loss_file_pattern)

                top_repetitions = []

                for loss_path in all_loss_paths:
                    rep_str = os.path.basename(loss_path).replace('losses-', '').replace('.csv', '')
                    rep_num = int(rep_str)
                    df = pd.read_csv(loss_path)
                    if not df.empty:
                        final_loss = df['Loss'].iloc[-1]
                        top_repetitions.append((rep_num, final_loss))

                top_repetitions.sort(key=lambda x: x[1])
                best_three = top_repetitions[:min(3, len(top_repetitions))]
                best_repetition_numbers = [rep for rep, loss in best_three]

                if "-r-" in model_type or 'hard' in model_type:
                    best_repetition_numbers = [0]

                print(f"best repetitions: {best_three}")
                for repetition in best_repetition_numbers:
                    for epoch in [epochs - 1]:
                        print('evaluating', epoch, model_name)
                        for this_path in [save_path]:#, retrain_path]: #retrain path removed
                            for eval_prior in [False]:#[False, True] if this_path == save_path else [False]:
                                df_paths = evaluate_model(eval_sets, label, load_path=load_path,
                                                      model_type=model_type,
                                                      model_load_path=this_path,
                                                      oracle_labels=oracle_labels,
                                                      repetition=repetition,
                                                      epoch_number=epoch,
                                                      prior_metrics=prior_metrics,
                                                      oracle_is_target=oracle_is_target,
                                                      act_label_names=act_label_names,
                                                      oracle_early=oracle_early,
                                                      last_timestep=last_timestep,
                                                      use_prior=eval_prior,
                                                      num_activation_batches=0)
                                if df_paths is not None:
                                    dfs_paths.extend(df_paths)
                                    if epoch == epochs - 1 or eval_prior:
                                        last_epoch_df_paths.extend(df_paths)

                
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

                    df_list = [pd.read_csv(df_path, compression='gzip') for df_path in dfs_paths]
                    combined_df = pd.concat(df_list, ignore_index=True)

                    last_df_list = [pd.read_csv(df_path, compression='gzip') for df_path in last_epoch_df_paths]
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
            test += 1
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            print(e)
            traceback.print_exc()

    if not skip_activations:
        print('corring activations...')
        process_activations(save_path, [epochs - 1], [x for x in range(repetitions)], use_inputs=False) #use 0 in epochs for prior
        process_activations(save_path + '-retrain', [epochs - 1], [x for x in range(repetitions)], use_inputs=False)

    return dfs_paths, last_epoch_df_paths, loss_paths


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return value
