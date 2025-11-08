import torch
import numpy as np
import os
import tqdm
from src.supervised_learning import TrainDatasetBig
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
import pandas as pd
import torch.nn.functional as F
import math
import time

from torch.multiprocessing import set_start_method
set_start_method('spawn', force=True)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.enable_flash_sdp(True)


from src.curriculum_configs import *

def last_step_targets(flat_oracle, module_ranges, module_name):
    s, e = module_ranges[module_name]
    seg = flat_oracle[:, s:e]                       # [B, 10] or [B, 50]

    if e - s == 10:                                 # op_belief (last step only)
        x = seg.reshape(-1, 2, 5)                   # [B, 2, 5]
    elif e - s == 50:                               # op_belief_t (5 steps)
        x = seg.reshape(-1, 5, 2, 5)[:, -1]         # [B, 2, 5]  # take last timestep
    else:
        raise ValueError(f"Unexpected width {e-s} for {module_name}")

    idx = x.argmax(dim=2)                           # [B, 2] lanes 0..4
    none_mask = (x.sum(dim=2) == 0)                 # [B, 2]
    idx[none_mask] = 5                              # class 5 = “none”
    return idx                                      # [B, 2]

def evaluate_model_stage(model, test_loader, novel_loader, novel_task_loader, criterion, device, stage_name=""):
    model.eval()
    test_losses, all_preds, all_targets = [], [], []
    sim_r_losses, sim_i_losses = [], []
    novel_sim_r_losses, novel_sim_i_losses = [], []
    novel_task_sim_r_losses, novel_task_sim_i_losses = [], []
    prev_time = time.time()

    with torch.inference_mode():
        for inputs, target_labels, _, _, _ in test_loader:
            inputs = inputs.to(device)
            target_labels = target_labels.to(device)
            outputs = model(inputs, None)

            loss = criterion(outputs['my_decision'], target_labels)
            test_losses.append(loss.item())
            if 'sim_loss' in outputs.keys():
                sim_r_losses.append(outputs['sim_loss']["r"].item())
                sim_i_losses.append(outputs['sim_loss']["i"].item())
            all_preds.append(outputs['my_decision'].argmax(dim=1))
            all_targets.append(target_labels)

        novel_accuracy = 0.0
        if novel_loader is not None:
            novel_preds, novel_targets = [], []
            for inputs, target_labels, _, _, _ in novel_loader:
                inputs = inputs.to(device)
                target_labels = target_labels.to(device)
                outputs = model(inputs, None)

                novel_preds.append(outputs['my_decision'].argmax(dim=1))
                novel_targets.append(target_labels)
                if 'sim_loss' in outputs.keys():
                    novel_sim_r_losses.append(outputs['sim_loss']["r"].item())
                    novel_sim_i_losses.append(outputs['sim_loss']["i"].item())

            if novel_preds:
                novel_preds = torch.cat(novel_preds)
                novel_targets = torch.cat(novel_targets)
                novel_accuracy = (novel_preds == novel_targets).float().mean().item()

        novel_task_accuracy = 0.0
        if novel_task_loader is not None:
            novel_task_preds, novel_task_targets = [], []
            for inputs, target_labels, _, _, _ in novel_task_loader:
                inputs = inputs.to(device)
                target_labels = target_labels.to(device)
                outputs = model(inputs, None)

                novel_task_preds.append(outputs['my_decision'].argmax(dim=1))
                novel_task_targets.append(target_labels)
                if 'sim_loss' in outputs.keys():
                    novel_task_sim_r_losses.append(outputs['sim_loss']["r"].item())
                    novel_task_sim_i_losses.append(outputs['sim_loss']["i"].item())

            if novel_task_preds:
                novel_task_preds = torch.cat(novel_task_preds)
                novel_task_targets = torch.cat(novel_task_targets)
                novel_task_accuracy = (novel_task_preds == novel_task_targets).float().mean().item()

    test_loss = sum(test_losses) / len(test_losses)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    accuracy = (all_preds == all_targets).float().mean().item()

    model.train()
    return_dict = {
        'accuracy': accuracy,
        'novel_accuracy': novel_accuracy,
        'novel_task_accuracy': novel_task_accuracy,
        'loss': test_loss,
        'stage': stage_name,
    }
    if len(sim_r_losses):
        sim_r_loss = sum(sim_r_losses) / len(sim_r_losses)
        sim_i_loss = sum(sim_i_losses) / len(sim_i_losses)
        novel_sim_r_loss = sum(novel_sim_r_losses) / len(novel_sim_r_losses) if novel_sim_r_losses else 0.0
        novel_sim_i_loss = sum(novel_sim_i_losses) / len(novel_sim_i_losses) if novel_sim_i_losses else 0.0
        novel_task_sim_r_loss = sum(novel_task_sim_r_losses) / len(novel_task_sim_r_losses) if novel_task_sim_r_losses else 0.0
        novel_task_sim_i_loss = sum(novel_task_sim_i_losses) / len(novel_task_sim_i_losses) if novel_task_sim_i_losses else 0.0

        return_dict.update({

            'sim_r_loss': sim_r_loss,
            'sim_i_loss': sim_i_loss,
            'novel_sim_r_loss': novel_sim_r_loss,
            'novel_sim_i_loss': novel_sim_i_loss,
            'novel_task_sim_r_loss': novel_task_sim_r_loss,
            'novel_task_sim_i_loss': novel_task_sim_i_loss}
        )
    print(time.time() - prev_time)
    return return_dict

    


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

        self.model.treat_perception_my.sigmoid_temp = current_temp
        self.model.treat_perception_op.sigmoid_temp = current_temp
        self.model.vision_perception_my.sigmoid_temp = current_temp
        self.model.vision_perception_op.sigmoid_temp = current_temp
        self.model.presence_perception_my.sigmoid_temp = current_temp
        self.model.presence_perception_op.sigmoid_temp = current_temp
        self.model.my_belief.sigmoid_temp = current_temp
        self.model.op_belief.sigmoid_temp = current_temp
        self.model.my_decision.sigmoid_temp = current_temp
        self.model.op_decision.sigmoid_temp = current_temp
        self.model.vision_prob = current_prob
        self.current_step = min(self.current_step + 1, self.total_steps)

    def get_temp(self):
        progress = min(1.0, self.rate * self.current_step / self.total_steps)
        cosine_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return self.start_temp + (self.end_temp - self.start_temp) * cosine_progress, self.vision_prob_start + (self.vision_prob_end - self.vision_prob_start) * cosine_progress

def filter_indices_by_params(data_list, labels_list, params_list, module_data_list, withheld_params, is_train=True):
    all_indices = []
    
    for i, (data, labels, params, module_data) in enumerate(zip(data_list, labels_list, params_list, module_data_list)):
        params_tuples = [tuple(param) for param in params]
        
        if is_train:
            valid_mask = np.array([param_tuple not in withheld_params for param_tuple in params_tuples])
        else:
            valid_mask = np.array([param_tuple in withheld_params for param_tuple in params_tuples])
        
        valid_indices = np.where(valid_mask)[0]
        
        offset = sum(len(d) for d in data_list[:i])
        all_indices.extend(offset + valid_indices)
    
    return all_indices

def train_model(train_sets, target_label, load_path='supervised/', save_path='', epochs=100,
                model_kwargs=None, model_type=None,
                oracle_labels=[], repetition=0,
                save_models=True, save_every=5, record_loss=True,
                oracle_is_target=False, batches=5000, last_timestep=True,
                seed=0, test_percent=0.2, train_sets_dict=None, curriculum_name=None):
    use_cuda = torch.cuda.is_available()
    print('got curriculum name:', curriculum_name)
    curriculum_config = CurriculumConfig(curriculum_name)

    torch.manual_seed(42+repetition+5)
    torch.cuda.manual_seed(42+repetition+5)
    torch.cuda.manual_seed_all(42+repetition+5)
    np.random.seed(42+repetition+5)

    model_kwargs['output_len'] = 5
    model_kwargs['channels'] = 3
    model_kwargs['oracle_is_target'] = oracle_is_target

    device = torch.device('cuda' if use_cuda else 'cpu')
    total_steps = 24
    epoch_steps = 24
    total_batches = sum(stage['batches'] for stage in curriculum_config.curriculum_stages)
    from supervised_learning_main import filter_indices, load_model


    if oracle_labels:
        module_labels = {}
        for label in oracle_labels:
            if 'op_belief' in label:
                if last_timestep:
                    module_labels['op_belief'] = ['b-loc-large', 'b-loc-small']
                else:
                    module_labels['op_belief_t'] = ['b-loc-large', 'b-loc-small']
            elif 'op_decision' in label:
                if last_timestep:
                    module_labels['op_decision'] = ['target-loc']
                else:
                    module_labels['op_decision_t'] = ['target-loc']
            elif 'my_belief' in label:
                if last_timestep:
                    module_labels['my_belief'] = ['loc-large', 'loc-small']
                else:
                    module_labels['my_belief_t'] = ['loc-large', 'loc-small']
    else:
        module_labels = {
            'my_decision': ['correct-loc'],
        }

    if True:
        oracle_labels = []
    if True:
        module_labels = {
            'my_decision': ['correct-loc'],
        }
    module_label_data = {module: [] for module in module_labels.keys()} 
    batch_size = model_kwargs['batch_size']
    model_kwargs['batch_size'] = 1024
    lr = model_kwargs['lr']
    batch_size = model_kwargs['batch_size']
    model = load_model(model_type, model_kwargs, device)

    print(model.kwargs, '######################################')

    if oracle_labels:
        model.store_per_timestep_beliefs = True
    else:
        model.store_per_timestep_beliefs = False

    stage_results = []

    real_batch = 0

    criterion = nn.CrossEntropyLoss()
    oracle_criterion = nn.CrossEntropyLoss() 
    
    model.vision_prob = 1.0

    epoch_losses_df = pd.DataFrame(columns=['Batch', 'Loss', 'Accuracy', 'Novel_Accuracy', 'Novel_Task_Accuracy'])

    t = tqdm.trange(total_batches)
    from experiments import init_regimes
    _, _, _, fregimes, _, _, _, _, _, _ = init_regimes()
    all_s3_regimes = fregimes['s3']

    for stage_config in curriculum_config.curriculum_stages:
        if 'trans' in model.kwargs['module_configs']['arch']:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=0.02)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        print('started curricular stage', stage_config['stage_name'])
        
        # stage_train_sets = apply_curriculum_stage(model, stage_config, train_sets_dict)
        stage_train_sets = []
        for regime in stage_config['data_regimes']:
            if regime in train_sets_dict:
                stage_train_sets.extend(train_sets_dict[regime])

        batches = stage_config['batches']
        print('training on these sets: ', stage_train_sets)

        #trainable = [name for name, module in model.get_module_dict().items() if module is not None and any(p.requires_grad for p in module.parameters())]
        #print('trainable modules:', trainable)
        print('module configs', model.kwargs['module_configs'])

        
        novel_regimes = [r for r in all_s3_regimes if r not in stage_train_sets]

        novel_eval_data, novel_eval_labels = [], []
        if novel_regimes:
            for regime_name in novel_regimes:
                dir = os.path.join(load_path, regime_name)
                if os.path.exists(dir):
                    regime_data = np.load(os.path.join(dir, 'obs.npz'), mmap_mode='r')['arr_0']
                    regime_labels_raw = np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0']
                    
                    if target_label == 'shouldGetBig':
                        if last_timestep:
                            regime_labels = np.eye(2)[regime_labels_raw[:, -1].astype(int)]
                        else:
                            regime_labels = np.eye(2)[regime_labels_raw.astype(int)].reshape(-1, 10)
                    elif len(regime_labels_raw.shape) > 2:
                        if last_timestep or True:
                            regime_labels = regime_labels_raw[..., -1, :]
                        else:
                            regime_labels = regime_labels_raw.reshape(-1, 25)
                    else:
                        regime_labels = regime_labels_raw.reshape(-1, 25)
                    
                    novel_eval_data.append(regime_data)
                    novel_eval_labels.append(regime_labels)

        if novel_eval_data:
            novel_params = [np.zeros((len(d), 0)) for d in novel_eval_data]
            novel_oracles = [np.zeros((len(d), 0)) for d in novel_eval_data]
            
            novel_indices = list(range(sum(len(d) for d in novel_eval_data)))
            novel_dataset = TrainDatasetBig(novel_eval_data, novel_eval_labels, novel_params, novel_oracles, novel_indices)
            novel_loader = DataLoader(novel_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        else:
            novel_loader = None

        if len(oracle_labels) == 0 or oracle_labels[0] == None:
            oracle_labels = []
        data, labels, params, module_data_combined = [], [], [], []
        all_params_sets = []

        for data_name in stage_train_sets:
            dir = os.path.join(load_path, data_name)
            data.append(np.load(os.path.join(dir, 'obs.npz'), mmap_mode='r')['arr_0'])
            
            labels_raw = np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0']
            if target_label == 'shouldGetBig':
                if last_timestep:
                    x = np.eye(2)[labels_raw[:, -1].astype(int)]
                else:
                    x = np.eye(2)[labels_raw.astype(int)].reshape(-1, 10)
                labels.append(x)
            elif len(labels_raw.shape) > 2:
                if last_timestep or True:
                    labels.append(labels_raw[..., -1, :])
                else:
                    labels.append(labels_raw.reshape(-1, 25))
            else:
                labels.append(labels_raw.reshape(-1, 25))
                
            current_params = np.load(os.path.join(dir, 'params.npz'), mmap_mode='r')['arr_0']
            params.append(current_params)
            all_params_sets.append(set(tuple(param) for param in current_params))
            
            data_arrays = []
            for module, label_list in module_labels.items():
                if not isinstance(label_list, list):
                    label_list = [label_list]
                    
                for label_name in label_list:
                    try:
                        label_data = np.load(os.path.join(dir, f'label-{label_name}.npz'), mmap_mode='r')['arr_0']
                        if last_timestep and len(label_data.shape) > 2:
                            label_data = label_data[..., -1, :]
                            if label_name == 'correct-loc':
                                label_data = label_data[:, :5] 
                            label_data = label_data.reshape(label_data.shape[0], -1)
                        elif len(label_data.shape) > 2:
                            label_data = label_data.reshape(label_data.shape[0], -1)
                        if label_data.ndim == 1:
                            label_data = label_data.reshape(label_data.shape[0], 1)

                            
                        data_arrays.append(label_data)
                    except Exception as e:
                        print(f"Error loading {label_name}: {e}")
            
            if data_arrays:
                try:
                    combined = np.concatenate(data_arrays, axis=1)
                    module_data_combined.append(combined)
                except Exception as e:
                    print(f"Error concatenating arrays: {e}")
                    module_data_combined.append(np.zeros((len(data[-1]), 0)))
            else:
                module_data_combined.append(np.zeros((len(data[-1]), 0)))

        all_unique_params = set()
        for param_set in all_params_sets:
            all_unique_params.update(param_set)
        
        all_unique_params = list(all_unique_params)
        n_withheld = max(1, int(0.1 * len(all_unique_params)))
        withheld_params = set(np.random.choice(len(all_unique_params), size=n_withheld, replace=False))
        withheld_params = {all_unique_params[i] for i in withheld_params}
        
        print(f"Total unique params: {len(all_unique_params)}, Withheld: {len(withheld_params)}")

        train_indices = filter_indices_by_params(data, labels, params, module_data_combined, withheld_params, is_train=True)
        novel_task_indices = filter_indices_by_params(data, labels, params, module_data_combined, withheld_params, is_train=False)
        
        train_dataset = TrainDatasetBig(data, labels, params, module_data_combined, train_indices)
        
        if novel_task_indices:
            novel_task_dataset = TrainDatasetBig(data, labels, params, module_data_combined, novel_task_indices)
            novel_task_loader = DataLoader(novel_task_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        else:
            novel_task_loader = None

        if record_loss:
            train_size = int(0.9 * len(train_dataset))
            print('epochs:', train_size // batch_size)
            test_size = len(train_dataset) - train_size
            train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42+repetition))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            baseline_results = evaluate_model_stage(model, test_loader, novel_loader, novel_task_loader, criterion, device, "baseline")
            new_row_data = {
                'Batch': real_batch,
                'Loss': baseline_results['loss'],
                'Accuracy': baseline_results['accuracy'],
                'Novel_Accuracy': baseline_results['novel_accuracy'],
                'Novel_Task_Accuracy': baseline_results['novel_task_accuracy'],
                'sim_r_loss': baseline_results['sim_r_loss'] if 'sim_r_loss' in baseline_results.keys() else 0,
                'sim_i_loss': baseline_results['sim_i_loss'] if 'sim_r_loss' in baseline_results.keys() else 0,
                'novel_sim_r_loss': baseline_results['novel_sim_r_loss'] if 'sim_r_loss' in baseline_results.keys() else 0,
                'novel_sim_i_loss': baseline_results['novel_sim_i_loss'] if 'sim_r_loss' in baseline_results.keys() else 0,
                'novel_task_sim_r_loss': baseline_results['novel_task_sim_r_loss'] if 'sim_r_loss' in baseline_results.keys() else 0,
                'novel_task_sim_i_loss': baseline_results['novel_task_sim_i_loss'] if 'sim_r_loss' in baseline_results.keys() else 0,
            }
            new_row = pd.DataFrame([new_row_data])
            epoch_losses_df = pd.concat([epoch_losses_df, new_row], ignore_index=True)
            print(f"Acc: {baseline_results['accuracy']:.4f}, Nacc: {baseline_results['novel_accuracy']:.4f}, NTacc: {baseline_results['novel_task_accuracy']:.4f}, L: {baseline_results['loss']:.4f}, P:", save_path, repetition)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        if False:
            last_digit = int(save_path[-1])
            print(last_digit)
            if last_digit > 0:
                save_path2 = save_path[:-1] + str(last_digit - 1)
                filename = os.path.join(save_path2, f'{repetition}-model_epoch{epochs - 1}.pt')
                loaded_model_info = torch.load(filename, map_location=device)
                loaded_model_kwargs, loaded_model_state_dict = loaded_model_info
                model.load_state_dict(loaded_model_state_dict)

        epoch_length = total_batches // total_steps
        epoch_length_val = total_batches // epoch_steps

        iter_loader = iter(train_loader)

        if save_models and False:
            os.makedirs(save_path, exist_ok=True)
            torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-prior.pt'))

        module_sizes = {'op_belief': 10, 'my_decision': 5, 'op_decision': 5, 'op_decision_t': 25, 'op_belief_t': 50, 'my_belief_t': 50}
        module_names = module_labels.keys()

        first_batch_processed = False
        module_ranges = {}

        if 'trans' in model.kwargs['module_configs']['arch']:
            print('transformer')
            scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=1e-4, total_steps=batches, anneal_strategy='cos', div_factor=5.0 )
        else:
            print('non transformer!')
            scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=5e-3, total_steps=batches, anneal_strategy='cos', div_factor=5.0 )

        for batch in range(batches):
            real_batch += 1
            total_loss = torch.zeros((), device=device)
            try:
                inputs, target_labels, _, oracles, _ = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                inputs, target_labels, _, oracles, _ = next(iter_loader)
            inputs, target_labels, oracles = (t.to(device, non_blocking=True) for t in (inputs, target_labels, oracles))


            outputs = model(inputs, None)
            my_decision_loss = criterion(outputs['my_decision'], target_labels)

            if not first_batch_processed:
                flat_oracle = oracles.reshape(oracles.size(0), -1)
                total_oracle_dims = flat_oracle.size(1)
                pos = 0
                for module_name in module_labels.keys():
                    module_output = outputs[module_name]
                    flattened_output_size = module_sizes[module_name]
                    module_ranges[module_name] = (pos, pos + flattened_output_size)
                    pos += flattened_output_size

                first_batch_processed = True

            if batch % 500 == 0 and batch > 0 and model.record_og_beliefs:
                with torch.no_grad():
                    original_use_neural = model.op_belief.use_neural
                    
                    model.op_belief.use_neural = False
                    model.og_op_belief.use_neural = False
                    hardcoded_outputs = model(inputs, None)
                    hardcoded_beliefs = hardcoded_outputs['op_belief']
                    oghardcoded_beliefs = hardcoded_outputs['og_op_belief']
                    hardcoded_decision = hardcoded_outputs['my_decision']
                    
                    model.op_belief.use_neural = True
                    neural_outputs = model(inputs, None)
                    neural_beliefs = neural_outputs['op_belief']
                    neural_decision = neural_outputs['my_decision']

                    neural_beliefs_last = neural_outputs['op_belief_t'][:, :, -1, :]     
                    hardcoded_beliefs_last = hardcoded_outputs['op_belief_t'][:, :, -1, :]
                    oghardcoded_beliefs_last = oghardcoded_beliefs
                    
                    model.op_belief.use_neural = original_use_neural


                hc_series = hardcoded_outputs['op_belief_t']
                hc_idx = hc_series.argmax(dim=-1)

                B = oracles.shape[0]
                s, e = module_ranges['op_belief_t']
                T = (e - s) // (2*5)
                oracle_chunk = oracles.reshape(B, -1)[:, s:e]
                oracle_5 = oracle_chunk.view(B, 2, T, 5)
                oracle_sum = oracle_5.sum(dim=-1)
                oracle_idx = oracle_5.argmax(dim=-1)
                oracle_idx[oracle_sum == 0] = 5
                og_idx = oghardcoded_beliefs.argmax(dim=-1)

                type_names = ["Large", "Small"]
                for i in range(min(5, B)):
                    print(f"\n=== Sample {i} ===")
                    for ttype in range(2):
                        print(f"{type_names[ttype]} Oracle: {oracle_idx[i,ttype].tolist()}")
                        print(f"{type_names[ttype]} HC:     {hc_idx[i,ttype].tolist()}")
                        print(f"{type_names[ttype]} OG:     {og_idx[i,ttype].item()}")

            oracle_losses = {}
            if oracle_labels:
                flat_oracle = oracles.view(oracles.size(0), -1)

                for name in module_labels.keys():
                    if name not in outputs or name not in module_ranges:
                        print('error')

                    s, e = module_ranges[name]
                    oracle_slice = flat_oracle[:, s:e] 
                    logits = outputs[name]

                    if "belief_t" in name:
                        B = oracle_slice.size(0)
                        T = (e - s) // (2 * 5)
                        tgt = oracle_slice.view(B, 2, T, 5)
                        idx = tgt.argmax(-1)
                        none_mask = tgt.sum(-1) == 0
                        idx[none_mask] = 5
                        idx = idx.permute(0, 2, 1).reshape(-1)
                        loss = oracle_criterion(logits.permute(0, 2, 1, 3).reshape(-1, 6),idx.reshape(-1))
                    elif name == "op_belief" or name == "my_belief":
                        B = oracle_slice.size(0)
                        tgt = oracle_slice.view(B, 2, 5)
                        idx = tgt.argmax(-1)
                        none_mask = tgt.sum(-1) == 0
                        idx[none_mask] = 5
                        loss = oracle_criterion(logits.reshape(-1, 6),idx.reshape(-1) )
                    elif name == "op_decision":
                        loss = oracle_criterion(logits,oracle_slice.argmax(-1))
                    elif name == "op_decision_t":
                        B = oracle_slice.size(0)
                        T = (e - s) // 5
                        tgt = oracle_slice.view(B, T, 5)
                        idx = tgt.argmax(-1)
                        none_mask = tgt.sum(-1) == 0 
                        idx[none_mask] = 5  
                        #print(logits.reshape(-1, 6).argmax(-1)[0], idx.reshape(-1)[0]) 
                        loss = oracle_criterion(logits.reshape(-1, 6), idx.reshape(-1))
                    oracle_losses[name] = loss

            oracle_loss = sum(oracle_losses.values())

            ###
            
            if False: # set to false for mini
                logits = outputs['op_belief_t']
                my_logits = outputs['my_belief_t']
                B, S, T, C = logits.shape
                device = logits.device

                vis_op = model.vision_perception_op(inputs, is_p1=0)          # [B,T], 0/1
                unseen_mask_full = (vis_op.round() <0.5).view(B, 1, T).expand(-1, S, -1)  # [B,2,T]

                logp_curr = logits.log_softmax(dim=-1)                        # [B,2,T,6]

                prior_logits = torch.full_like(logits[:, :, :1, :], -10)    # [B,2,1,6]
                prior_logits[..., 5] = 10.0

                prev_logits = torch.cat([prior_logits, logits[:, :, :-1, :].detach()], dim=2)  # [B,2,T,6]
                logp_prev  = prev_logits.log_softmax(dim=-1)                                    # [B,2,T,6]

                kl_t = F.kl_div(logp_curr, logp_prev, log_target=True, reduction="none").sum(dim=-1)  # [B,2,T]

                # average KL over unseen steps only (including t=0 when unseen)
                den = unseen_mask_full.float().sum().clamp_min(1.0)
                loss_unseen_stability = (kl_t * unseen_mask_full.float()).sum() / den
            if False: # set to false for mini
                logits = outputs['op_belief_t']
                my_logits = outputs['my_belief_t']
                B, S, T, C = logits.shape
                device = logits.device

                with torch.no_grad():
                    B = oracles.size(0)
                    flat_oracle = oracles.view(B, -1)
                    s_op, e_op = module_ranges['op_belief_t']
                    T = (e_op - s_op) // (2 * 5)  # 2 treats × 5 positions
                    op_gt = flat_oracle[:, s_op:e_op].view(B, 2, T, 5)
                    op_idx = op_gt.argmax(-1)                  # [B,2,T] in {0..4}
                    op_none = op_gt.sum(-1) == 0               # [B,2,T]
                    op_idx[op_none] = 5                        # 5 = "none"

                    vis = model.vision_perception_op(inputs, is_p1=0).round().long()  # [B,T]

                    prev = torch.cat([torch.full_like(op_idx[:, :, :1], 5), op_idx[:, :, :-1]], dim=2)  # [B,2,T]

                    t0_unseen = (vis[:, 0] < 0.5).view(B,1,1).expand(-1,2,1)                              # [B,2,1]
                    t0_seen = (vis[:, 0] > 0.5).view(B,1,1).expand(-1,2,1)                              # [B,2,1]
                    chg_t0 = ((op_idx[:, :, :1] != 5) & t0_unseen).float().mean().item()
                    chg_t0_seen = ((op_idx[:, :, :1] != 5) & t0_seen).float().mean().item()

                    seen_mask = (vis[:, 1:] > 0.5).view(B,1,T-1).expand(-1,2,-1)                          # [B,2,T-1]
                    unseen_mask = ~seen_mask

                    chg_unseen = ((op_idx[:, :, 1:] != prev[:, :, 1:]) & unseen_mask).float().mean().item()
                    chg_seen = ((op_idx[:, :, 1:] != prev[:, :, 1:]) & seen_mask).float().mean().item()
                    pred_idx = logits.argmax(-1)                      # [B,2,T]
                    pred_prev = torch.cat([torch.full_like(pred_idx[:, :, :1], 5), pred_idx[:, :, :-1]], dim=2)

                    seen_mask   = (vis[:, 1:] > 0.5).view(B,1,T-1).expand(-1,2,-1)
                    unseen_mask = ~seen_mask

                    pred_chg_unseen = ((pred_idx[:, :, 1:] != pred_prev[:, :, 1:]) & unseen_mask).float().mean().item()
                    pred_chg_seen   = ((pred_idx[:, :, 1:] != pred_prev[:, :, 1:]) & seen_mask).float().mean().item()

                logp = logits.log_softmax(dim=-1)                     # [B,2,T,6]


                if batch % 500 == 0:
                    print("GT:", chg_unseen, chg_seen, "PRED:", pred_chg_unseen, pred_chg_seen)

                with torch.no_grad():
                    vis_op   = model.vision_perception_op(inputs, is_p1=0).to(logits.dtype)  # [B,T] in {0,1}
                    gt_pos   = model.treat_perception_op(inputs)                             # [B,T,2,6]
                    tgt_idx  = gt_pos.argmax(dim=-1)                                         # [B,T,2], 0..5

                seen_mask   = (vis_op[:, 1:].round() == 1)                 # [B, T-1] bool
                unseen_mask = ~seen_mask                           # [B, T-1] bool
                seen_mask   = seen_mask.unsqueeze(1).expand(-1, S, -1)     # [B,2,T-1]
                unseen_mask = unseen_mask.unsqueeze(1).expand(-1, S, -1)   # [B,2,T-1]

                tgt_t = tgt_idx.permute(0, 2, 1)[:, :, 1:]         # [B,2,T-1]

                logp_curr = logits.log_softmax(-1)
                logp_prev = prev_logits.log_softmax(-1)

                seen_mask = (vis_op[:,1:].round()>0.51).view(B,1,T-1).expand(-1,S,-1)

                p_t   = logp_curr[:,:,1:,:].exp()
                p_tm1 = logp_prev[:,:,1:,:].exp()
                q     = 0.5*(p_t+p_tm1)
                js = 0.5*(
                    F.kl_div(logp_curr[:,:,1:,:], q.log(), log_target=True, reduction="none").sum(-1) +
                    F.kl_div(logp_prev[:,:,1:,:], q.log(), log_target=True, reduction="none").sum(-1)
                )
                margin = 0.10
                loss_seen_ce = torch.relu(margin - js)
                loss_seen_ce = (loss_seen_ce*seen_mask.float()).sum() / seen_mask.float().sum().clamp_min(1)

                # for persisting past zeroth step
                extended_seen_mask = torch.cat([(vis_op[:, :1] > 0.5).unsqueeze(1).expand(-1, 2, -1), seen_mask], dim=2)
                extended_unseen_mask = ~extended_seen_mask
                initial_logits = torch.zeros_like(logits[:, :, :1, :])
                initial_logits[:, :, :, 5] = 10.0
                extended_prev = torch.cat([initial_logits, logits[:, :, :-1, :]], dim=2)
                extended_curr = logits
                logp_curr = extended_curr.log_softmax(dim=-1)
                logp_prev = extended_prev.log_softmax(dim=-1)
                p_prev = logp_prev.exp()
                kl = (p_prev * (logp_prev - logp_curr)).sum(dim=-1)
                if extended_unseen_mask.any():
                    loss_persist = (kl * extended_unseen_mask.float()).sum() / extended_unseen_mask.float().sum().clamp_min(1.0)
                else:
                    loss_persist = torch.zeros((), device=device)

                if unseen_mask.any() and torch.rand(1).item() < 1:
                    n_unseen = unseen_mask.sum().item()
                    loss_misinformed = F.cross_entropy(logits[:, :, 1:, :][unseen_mask], torch.randint(0, 5, (n_unseen,), device=device))
                else:
                    loss_misinformed = torch.zeros((), device=device)

                delta_logits = (logits[:, :, 1:, :] - logits[:, :, :-1, :]).pow(2).mean(dim=-1)  # [B,2,T-1]
                if unseen_mask.any():
                    loss_delta = (delta_logits * unseen_mask.float()).sum() / unseen_mask.float().sum().clamp_min(1.0)
                else:
                    loss_delta = torch.zeros((), device=device)

                if True:
                    flat_oracle = oracles.view(oracles.size(0), -1)

                    s_my, e_my = module_ranges['my_belief_t']
                    my_oracle_slice = flat_oracle[:, s_my:e_my]
                    B = my_oracle_slice.size(0)
                    T = (e_my - s_my) // (2 * 5)
                    my_tgt = my_oracle_slice.view(B, 2, T, 5)
                    my_gt_targets = my_tgt.argmax(-1)  # [B, 2, T]
                    my_none_mask = my_tgt.sum(-1) == 0
                    my_gt_targets[my_none_mask] = 5

                    s_op, e_op = module_ranges['op_belief_t']
                    op_oracle_slice = flat_oracle[:, s_op:e_op]
                    op_tgt = op_oracle_slice.view(B, 2, T, 5)
                    op_gt_targets = op_tgt.argmax(-1)  # [B, 2, T]
                    op_none_mask = op_tgt.sum(-1) == 0
                    op_gt_targets[op_none_mask] = 5

                    my_pred = my_logits.argmax(dim=-1)  # [B, 2, T]
                    op_pred = logits.argmax(dim=-1)     # [B, 2, T]
                    should_differ = (my_gt_targets != op_gt_targets)  # [B, 2, T] bool
                    preds_same = (my_pred == op_pred)   # [B, 2, T] bool
                    wrong_agreement = should_differ & preds_same  # [B, 2, T] bool
                    loss_wrong_agreement = wrong_agreement.float().mean()

            #total_loss = my_decision_loss + 0.1*oracle_loss + 0.1*loss_misinformed + loss_seen_ce + loss_persist + loss_wrong_agreement
            total_loss = my_decision_loss + oracle_loss if model.use_oracle else my_decision_loss #+ 0.5*loss_unseen_stability + 0.01*loss_seen_ce + 0.01*loss_delta + 0.01*loss_persist

            #print(outputs["sim_loss"])
            if "r" in model.sim_style and not model.use_gt_sim and not model.skip_sim_loss:
                sim_r_loss = outputs["sim_loss"]["r"]
                total_loss += sim_r_loss
            else:
                sim_r_loss = 0.0
            if "i" in model.sim_style and not model.use_gt_sim and not model.skip_sim_loss:
                sim_i_loss = outputs["sim_loss"]["i"]
                total_loss += sim_i_loss
            else:
                sim_i_loss = 0.0

            #total_loss += 0.001 * sum((p1 - p2).pow(2).mean() for p1, p2 in zip(model.e2e_op_belief.parameters(), model.e2e_my_belief.parameters()))

            optimizer.zero_grad(set_to_none=True)
            #scaler.scale(total_loss).backward()
            #scaler.unscale_(optimizer) 
            #scaler.step(optimizer)
            #scaler.update()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if (batch+1) % epoch_length == 0:
                if batch > 1:
                    t.update(epoch_length)

            if record_loss and (((real_batch) % (2*epoch_length_val) == 0) or (real_batch == total_batches - 1)):

                stage_results = evaluate_model_stage(model, test_loader, novel_loader, novel_task_loader, criterion, device, stage_config['stage_name'])
    
                new_row_data = {
                    'Batch': real_batch,
                    'Loss': stage_results['loss'],
                    'OLoss_total': float(oracle_loss.item()) if oracle_labels else 0,
                    'Accuracy': stage_results['accuracy'],
                    'Novel_Accuracy': stage_results['novel_accuracy'],
                    'Novel_Task_Accuracy': stage_results['novel_task_accuracy'],
                    "sim_r_loss": stage_results['sim_r_loss'] if 'sim_r_loss' in stage_results.keys() else 0,
                    "sim_i_loss": stage_results['sim_i_loss'] if 'sim_r_loss' in stage_results.keys() else 0,
                    'novel_sim_r_loss': stage_results['novel_sim_r_loss'] if 'sim_r_loss' in stage_results.keys() else 0,
                    'novel_sim_i_loss': stage_results['novel_sim_i_loss'] if 'sim_r_loss' in stage_results.keys() else 0,
                    'novel_task_sim_r_loss': stage_results['novel_task_sim_r_loss'] if 'sim_r_loss' in stage_results.keys() else 0,
                    'novel_task_sim_i_loss': stage_results['novel_task_sim_i_loss'] if 'sim_r_loss' in stage_results.keys() else 0,

                }
                for k, v in oracle_losses.items():
                    new_row_data[f'OLoss_{k}'] = float(v.item())

                new_row = pd.DataFrame([new_row_data])
                epoch_losses_df = pd.concat([epoch_losses_df, new_row], ignore_index=True)
                model.train()

                final_accuracy = stage_results['accuracy']

                if oracle_labels:
                    oloss_strs = [f"{k}:{v.item():.3f}" for k, v in oracle_losses.items()]
                    oloss_line = " | ".join(oloss_strs) + f" | sum:{oracle_loss.item():.4f}"
                else:
                    oloss_line = "none"

                print(
                    f"Acc: {stage_results['accuracy']:.3f}, "
                    f"Nacc: {stage_results['novel_accuracy']:.3f}, "
                    f"NTacc: {stage_results['novel_task_accuracy']:.3f}, "
                    f"L: {stage_results['loss']:.3f}, "
                    f"O: {oloss_line}, "
                    f"SR: {stage_results['sim_r_loss'] if 'sim_r_loss' in stage_results.keys() else 0:.3f} {stage_results['novel_sim_r_loss'] if 'sim_r_loss' in stage_results.keys() else 0:.3f}, "
                    f"SI: {stage_results['sim_i_loss'] if 'sim_r_loss' in stage_results.keys() else 0:.3f} {stage_results['novel_sim_i_loss'] if 'sim_r_loss' in stage_results.keys() else 0:.3f}, "
                    #f"Persist: {loss_persist:.3f} delta: {loss_delta:.3f} ce: {loss_seen_ce:.3f} "
                    #f"LW: {loss_wrong_agreement:.3f} MI: {loss_misinformed:.3f}"
                    #f"LU: {loss_unseen_stability:.3f}"
                    f"Path: {save_path} Rep: {repetition}"
                )
                if save_models and real_batch == total_batches - 1:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-{stage_config["stage_name"]}.pt'))

    if record_loss:
        epoch_losses_df.to_csv(os.path.join(save_path, f'losses-{repetition}.csv'), index=False)
    return final_accuracy