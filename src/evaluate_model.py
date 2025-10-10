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
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, networkx as nx, ast, math, os
    def parse_onehot(s):
        if isinstance(s,str):
            t=ast.literal_eval(s); 
            return t.index(1) if 1 in t else 5
        return int(s)
    def uniq(seq):
        seen=set(); out=[]
        for x in seq:
            if x not in seen: out.append(x); seen.add(x)
        return out
    df=pd.read_csv(csv_path).copy()
    for c in ['timestep','from_state','treat_state','vision','to_state','count']: assert c in df.columns
    has_size='treat_size' in df.columns
    df['from_idx']=df['from_state'].apply(parse_onehot)
    df['to_idx']=df['to_state'].apply(parse_onehot)
    df['treat_idx']=df['treat_state'].apply(parse_onehot)
    df['vision']=df['vision'].astype(int)
    df['count']=df['count'].astype(float)
    if has_size: df['treat_size']=df['treat_size'].astype(int)
    states=list(range(6))
    label={i:('None' if i==5 else str(i)) for i in states}
    palette={0:'#1f77b4',1:'#ff7f0e',2:'#2ca02c',3:'#d62728',4:'#9467bd',5:'#7f7f7f'}
    pos_circle={label[i]:(math.cos(2*math.pi*i/6),math.sin(2*math.pi*i/6)) for i in states}
    max_t=int(df['timestep'].max())
    timesteps=list(range(max_t+1))
    def pct_table(frame, include_timestep=True, extra_keys=()):
        group_cols=(['timestep'] if include_timestep else [])+list(extra_keys)+['vision','from_idx','treat_idx','to_idx']
        group_cols=uniq(group_cols)
        agg=frame.groupby(group_cols,as_index=False)['count'].sum()
        denom_cols=(['timestep'] if include_timestep else [])+list(extra_keys)+['vision','from_idx','treat_idx']
        denom_cols=uniq(denom_cols)
        den=agg.groupby(denom_cols,as_index=False)['count'].sum().rename(columns={'count':'denom'})
        out=agg.merge(den,on=denom_cols,how='left')
        out['pct']=np.where(out['denom']>0,100.0*out['count']/out['denom'],0.0)
        return out
    def draw_network(sub,title,suffix):
        plt.figure(figsize=(9,7))
        G=nx.DiGraph()
        for i in states: G.add_node(label[i])
        pos={label[i]:pos_circle[label[i]] for i in states}
        node_cols=[palette[i] for i in states]
        nx.draw_networkx_nodes(G,pos,node_color=node_cols,node_size=1400,alpha=0.95)
        nx.draw_networkx_labels(G,pos,font_size=11,font_weight='bold')
        edge_map={}
        for _,r in sub.iterrows():
            u=label[int(r['from_idx'])]; v=label[int(r['to_idx'])]
            edge_map.setdefault((u,v),[]).append(r)
            if not G.has_edge(u,v): G.add_edge(u,v)
        nx.draw_networkx_edges(G,pos,alpha=0.7,edge_color='#000000',arrows=True,arrowsize=22,min_target_margin=12)
        used=set()
        for (u,v),rows in edge_map.items():
            x1,y1=pos[u]; x2,y2=pos[v]
            if u==v:
                cx,cy=0.0,0.0
                rx,ry=x1-cx,y1-cy
                rl=max((rx*rx+ry*ry)**0.5,1e-6)
                ux,uy=rx/rl,ry/rl
                base_x,base_y=x1+ux*0.28,y1+uy*0.28
                for i,r in enumerate(rows):
                    fc=(palette[int(r['treat_idx'])] if int(r['vision'])==1 else '#d0d0d0')
                    txt=f"{int(r['treat_idx'])}:{float(r['pct']):.0f}%"
                    plt.text(base_x,base_y+0.045*i,txt,fontsize=8,ha='center',va='center',rotation=0,bbox=dict(boxstyle='round,pad=0.18',facecolor=fc,edgecolor='#222222',alpha=0.95))
                continue
            mx,my=(x1+x2)/2,(y1+y2)/2
            ang=np.degrees(np.arctan2(y2-y1,x2-x1))
            if ang>90: ang-=180
            if ang<-90: ang+=180
            dx,dy=x2-x1,y2-y1
            L=max(math.hypot(dx,dy),1e-6)
            ox,oy=-dy/L*0.06,dx/L*0.06
            opp=edge_map.get((v,u),[])
            for i,r in enumerate(rows):
                fc=(palette[int(r['treat_idx'])] if int(r['vision'])==1 else '#d0d0d0')
                txt=f"{int(r['treat_idx'])}:{float(r['pct']):.0f}%"
                plt.text(mx+ox,my+oy+0.045*i,txt,fontsize=8,ha='center',va='center',rotation=ang,bbox=dict(boxstyle='round,pad=0.18',facecolor=fc,edgecolor='#222222',alpha=0.95))
            if len(opp)>0 and (v,u) not in used:
                for i,r in enumerate(opp):
                    fc=(palette[int(r['treat_idx'])] if int(r['vision'])==1 else '#d0d0d0')
                    txt=f"{int(r['treat_idx'])}:{float(r['pct']):.0f}%"
                    plt.text(mx-ox,my-oy+0.045*i,txt,fontsize=8,ha='center',va='center',rotation=ang+180,bbox=dict(boxstyle='round,pad=0.18',facecolor=fc,edgecolor='#222222',alpha=0.95))
                used.add((u,v))
        plt.title(title,fontsize=12)
        plt.axis('off')
        plt.margins(0.006,0.006)
        plt.subplots_adjust(0,0,1,1)
        if save_path:
            base,ext=os.path.splitext(save_path); fn=f"{base}{suffix}{ext}"; plt.savefig(fn,dpi=180,bbox_inches='tight',pad_inches=0.005); plt.close(); return fn
        else:
            plt.show(); return None
    outs=[]
    per_t=pct_table(df,include_timestep=True,extra_keys=())
    for t in timesteps:
        sub=per_t[per_t['timestep']==t]
        outs.append(draw_network(sub,f"Timestep {t}",f"_t{t}"))
    if has_size:
        per_t_size=pct_table(df,include_timestep=True,extra_keys=('treat_size',))
        for sz in sorted(df['treat_size'].unique()):
            for t in timesteps:
                sub=per_t_size[(per_t_size['treat_size']==sz)&(per_t_size['timestep']==t)]
                outs.append(draw_network(sub,f"Timestep {t} | size={int(sz)}",f"_t{t}_size_{int(sz)}"))
    agg_all=pct_table(df,include_timestep=False,extra_keys=())
    outs.append(draw_network(agg_all,"All timesteps","_agg"))
    agg_tr=pct_table(df,include_timestep=False,extra_keys=('treat_idx',))
    if has_size:
        agg_sz=pct_table(df,include_timestep=False,extra_keys=('treat_size',))
        for sz in sorted(df['treat_size'].unique()):
            outs.append(draw_network(agg_sz[agg_sz['treat_size']==sz],f"All timesteps | size={int(sz)}",f"_agg_sz{int(sz)}"))
    return [o for o in outs if o]



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

    try:
        model.end2end_model.save_transition_table(os.path.join(model_load_path, f'transitions-{repetition}.csv'))
    except:
        model.belief_op.save_transition_table(os.path.join(model_load_path, f'transitions-{repetition}.csv'))
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