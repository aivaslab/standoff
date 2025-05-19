import torch
import numpy as np
import os
import tqdm
from src.supervised_learning import TrainDatasetBig, SimpleMultiDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
import pandas as pd
import math
import sys
from matplotlib import pyplot as plt
import time
import pickle

from src.utils.plotting import save_double_param_figures, save_single_param_figures, save_fixed_double_param_figures, \
    save_fixed_triple_param_figures, save_key_param_figures, save_delta_figures, plot_regime_lengths, \
    plot_awareness_results, plot_accuracy_vs_vision




def register_hooks(model, hook):
    save_inputs = True
    for name, layer in model.named_modules():
        layer.register_forward_hook(hook.save_activation(name, save_inputs=save_inputs))
        save_inputs = False


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

def evaluate_model(test_sets, target_label, load_path='supervised/', model_load_path='', oracle_labels=[], repetition=0,
                   epoch_number=0, prior_metrics=[], num_activation_batches=-1, oracle_is_target=False, act_label_names=[], save_labels=False,
                   oracle_early=False, last_timestep=True, model_type=None, seed=0, test_percent=0.2, use_prior=False, train_sets=None):
    

    from supervised_learning_main import load_model_data_eval_retrain, load_model, decode_event_name

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

    model = load_model(model_type, model_kwargs, device, False)
    hardcoded_model = load_model('a-hardcoded', model_kwargs, device)

    #print('num_activation_batches', num_activation_batches)
    overall_correct = 0
    overall_total = 0
    np.set_printoptions(threshold=sys.maxsize)

    num_visions = 10

    do_vision_stuff = False
    if True:
        print('evaluating vision changes')
        vision_probs = np.arange(0, 1.05, 0.1)
        uncertainties = [0.3]
        accuracy_results1 = {u: {prob: [] for prob in vision_probs} for u in uncertainties}
        #awareness_results = {prob: [] for prob in vision_probs}
        #accuracy_results1 = {prob: [] for prob in vision_probs}

        #accuracy_results2 = {prob: [] for prob in vision_probs}

    model.num_visions = 1
    model.vision_prob = 1

    # i have commented out oracle related things
    # this includes oracle_is_target check
    for idx, _val_loader in enumerate(test_loaders):

        with torch.inference_mode():

            tq = tqdm.trange(len(_val_loader))

            for i, (inputs, labels, params, _, metrics, act_labels_dict) in enumerate(_val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                #inputs, labels, oracles = inputs.to(device), labels.to(device), oracles.to(device)
                #print(labels.shape)
                labels = labels[:,:-1]
                max_labels = torch.argmax(labels, dim=1)
                #if torch.isnan(max_labels).any():
                #    print("labels has nan")
                vision_prob = 1.0
                #masked_vision = (torch.rand(inputs.size(0), 5, device=inputs.device) <= vision_prob).float()
                #masked_input = inputs * masked_vision.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                model.my_belief.uncertainty = 0.3
                outputs = model(inputs, None)
                

                #print("egg")

                keys_to_compare = [
                    'treat_perception', 
                    'vision_perception', 
                    'vision_perception_my', 
                    'presence_perception', 
                    'my_decision', 
                    'my_belief',
                    'op_belief',
                    'op_decision'
                ]
                if False:
                    #hc_outputs = hardcoded_model(inputs, None)
                    differences = {}
                    all_same = True
                    threshold = 0.35
                    decimals = 0

                    batch_size = outputs[keys_to_compare[0]].shape[0]

                    different_items = set()
                    for key in keys_to_compare:
                        #print(outputs[key].shape, hc_outputs[key].shape, key)
                        difference = outputs[key] - hc_outputs[key]
                        
                        flattened_diff = difference.view(difference.shape[0], -1)
                        
                        batch_has_diff = torch.any(torch.abs(flattened_diff) > threshold, dim=1)
                        diff_batch_indices = torch.where(batch_has_diff)[0].tolist()
                        
                        different_items.update(diff_batch_indices)

                    #print(f"\nFound differences in {len(different_items)} of {batch_size} batch items")

                    for batch_idx in sorted(different_items):
                        #print(f"\nBatch item {batch_idx} differences:")
                        
                        for key in keys_to_compare:
                            difference = outputs[key][batch_idx] - hc_outputs[key][batch_idx]
                            max_diff = torch.max(torch.abs(difference)).item()
                            
                            
                            #print(f"  '{key}':")
                            
                            model_val = torch.round(outputs[key][batch_idx] * (10**decimals)) / (10**decimals)
                            hc_val = torch.round(hc_outputs[key][batch_idx] * (10**decimals)) / (10**decimals)

                            model_val = [f"{x:.0f}" for x in outputs[key][batch_idx].cpu().flatten().tolist()]
                            hc_val = [f"{x:.0f}" for x in hc_outputs[key][batch_idx].cpu().flatten().tolist()]
                            
                            #print(f"    MultiAgent: {model_val}")
                            #print(f"    TwoAgent: {hc_val}")

                outputs = outputs['my_decision']
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