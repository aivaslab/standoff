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

def visualize_transition_network(csv_path, save_path=None):
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    
    unique_states = list(set(df['from_state'].tolist() + df['to_state'].tolist()))
    cmap = plt.cm.Set3
    state_colors = {state: cmap(i/len(unique_states)) for i, state in enumerate(unique_states)}
    
    transitions = []
    node_states = {}
    
    for _, row in df.iterrows():
        from_state = row['from_state']
        to_state = row['to_state']
        from_node = "None" if from_state == 5 else str(from_state)
        to_node = "None" if to_state == 5 else str(to_state)
        
        node_states[from_node] = from_state
        node_states[to_node] = to_state
        
        treat_num = str(row['treat_state'])
        vision = row['vision']
        
        if vision == 0:
            base_text = f"{treat_num}({row['count']})"
            label_text = ''.join(char + '̶' for char in base_text)
            color = 'gray'
        else:
            label_text = f"{treat_num}({row['count']})"
            color = state_colors[to_state]

        transitions.append({
            'from': from_node, 'to': to_node, 
            'label': label_text, 'color': color, 'weight': row['count']
        })
        
        if not G.has_edge(from_node, to_node):
            G.add_edge(from_node, to_node)
    
    plt.figure(figsize=(15, 10))
    pos = nx.circular_layout(G)
    
    node_colors = [state_colors[node_states[node]] for node in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray', arrows=True, arrowsize=25, min_target_margin=30)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    
    edge_transitions = {}
    for t in transitions:
        key = (t['from'], t['to'])
        if key not in edge_transitions:
            edge_transitions[key] = []
        edge_transitions[key].append(t)
    
    processed_pairs = set()
    
    for (u, v), trans_list in edge_transitions.items():
        if u == v:
            x, y = pos[u]
            combined_label = "\n".join([t['label'] for t in trans_list])
            plt.text(x, y+0.3, combined_label, fontsize=8, ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.8))
        elif (v, u) in processed_pairs:
            continue
        else:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            if (v, u) in edge_transitions:
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                offset_x, offset_y = -dy/length * 0.04, dx/length * 0.04
                
                for i, t in enumerate(trans_list):
                    plt.text(mid_x + offset_x, mid_y + offset_y + i*0.05, t['label'], fontsize=8, ha='center', va='center', 
                            rotation=angle, bbox=dict(boxstyle="round,pad=0.2", facecolor=t['color'], alpha=0.8))
                
                for i, t in enumerate(edge_transitions[(v, u)]):
                    plt.text(mid_x - offset_x, mid_y - offset_y + i*0.05, t['label'], fontsize=8, ha='center', va='center', 
                            rotation=angle+180, bbox=dict(boxstyle="round,pad=0.2", facecolor=t['color'], alpha=0.8))
                processed_pairs.add((u, v))
            else:
                for i, t in enumerate(trans_list):
                    plt.text(mid_x, mid_y + i*0.05, t['label'], fontsize=8, ha='center', va='center', 
                            rotation=angle, bbox=dict(boxstyle="round,pad=0.2", facecolor=t['color'], alpha=0.8))
    
    plt.title("Belief State Transition Network")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')


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

def vision_check(uncertainties, vision_probs, model, inputs, max_labels):
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
def compare_with_hardcoded(hardcoded_model, inputs, outputs, keys_to_compare):
    hc_outputs = hardcoded_model(inputs, None)
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

def evaluate_model(test_sets, target_label, load_path='supervised/', model_load_path='', oracle_labels=[], repetition=0,
                   epoch_number=0, prior_metrics=[], num_activation_batches=-1, oracle_is_target=False, act_label_names=[], save_labels=False,
                   oracle_early=False, last_timestep=True, model_type=None, seed=0, test_percent=0.2, use_prior=False, train_sets=None):

    from supervised_learning_main import load_model_data_eval_retrain, load_model, decode_event_name
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

    #model = load_model(model_type, model_kwargs, device)
    #model.eval()
    hardcoded_model = load_model('a-hardcoded', model_kwargs, device)

    #print('num_activation_batches', num_activation_batches)
    overall_correct = 0
    overall_total = 0
    np.set_printoptions(threshold=sys.maxsize)

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
    num_visions = 1
    model.vision_prob = 1

    #visualize_transition_network(os.path.join(model_load_path, 'transitions.csv'), os.path.join(model_load_path, 'transition_network.png'))

    print('kwargs', model_kwargs)
    #exit()

    # i have commented out oracle related things
    # this includes oracle_is_target check
    for idx, _val_loader in enumerate(test_loaders):

        with torch.inference_mode():

            tq = tqdm.trange(len(_val_loader))

            for i, (inputs, labels, params, _, metrics, act_labels_dict) in enumerate(_val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                #inputs, labels, oracles = inputs.to(device), labels.to(device), oracles.to(device)
                #print(labels.shape)
                #print(labels[0])
                labels = labels[:,-5:]

                max_labels = torch.argmax(labels, dim=1)
                outputs = model(inputs, None)
                #print(inputs.shape, str(model))

                keys_to_compare = [
                    #'treat_perception', 
                    #'vision_perception', 
                    #'vision_perception_my', 
                    #'presence_perception', 
                    'my_decision', 
                    #'my_belief',
                    #'op_belief',
                    #'op_decision'
                ]
                if False:
                    compare_with_hardcoded(hardcoded_model, inputs, outputs, keys_to_compare)
                    

                outputs = outputs['my_decision']
                #print(outputs.shape, max_labels.shape)
                if do_vision_stuff:
                    vision_check(uncertainties, vision_probs, model, inputs, max_labels)
                    
                losses = special_criterion(outputs, max_labels)
                predicted = outputs.argmax(1)
                print(predicted, max_labels)

                corrects = predicted.eq(max_labels)
                total = corrects.numel()
                num_correct = corrects.sum().item()
                pred = predicted.cpu()
                overall_correct += num_correct
                overall_total += total

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
                    for k, (param, loss, correct, pred, max_label) in enumerate(zip(params, losses, corrects, pred, max_labels))
                ]

                param_losses_list.extend(batch_param_losses)

    print('correct', overall_correct, overall_total)

    model.op_belief_per_timestep.save_transition_table(os.path.join(model_load_path, f'transitions-{repetition}.csv'))
    visualize_transition_network(os.path.join(model_load_path, f'transitions-{repetition}.csv'), os.path.join(model_load_path, f'transition_network-{repetition}.png'))


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