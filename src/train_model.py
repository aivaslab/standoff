import torch
import numpy as np
import os
import tqdm
from src.supervised_learning import TrainDatasetBig
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
import pandas as pd
import math

from src.curriculum_configs import *

def evaluate_model_stage(model, test_loader, novel_loader, novel_task_loader, criterion, device, stage_name=""):
    model.eval()
    test_losses, all_preds, all_targets = [], [], []
    
    with torch.inference_mode():
        for inputs, target_labels, _, _, _ in test_loader:
            inputs = inputs.to(device)
            target_labels = target_labels.float().to(device)[:,:-1]
            
            outputs = model(inputs, None)
            loss = criterion(outputs['my_decision'], torch.argmax(target_labels, dim=1))
            test_losses.append(loss.item())
            all_preds.append(outputs['my_decision'].argmax(dim=1))
            all_targets.append(torch.argmax(target_labels, dim=1))
        
        novel_accuracy = 0.0
        if novel_loader is not None:
            novel_preds, novel_targets = [], []
            for inputs, target_labels, _, _, _ in novel_loader:
                inputs = inputs.to(device)
                target_labels = target_labels.float().to(device)
                
                outputs = model(inputs, None)
                novel_preds.append(outputs['my_decision'].argmax(dim=1))
                novel_targets.append(torch.argmax(target_labels, dim=1))
            
            if novel_preds:
                novel_preds = torch.cat(novel_preds)
                novel_targets = torch.cat(novel_targets)
                novel_accuracy = (novel_preds == novel_targets).float().mean().item()

        novel_task_accuracy = 0.0
        if novel_task_loader is not None:
            novel_task_preds, novel_task_targets = [], []
            for inputs, target_labels, _, _, _ in novel_task_loader:
                inputs = inputs.to(device)
                target_labels = target_labels.float().to(device)[:,:-1]
                
                outputs = model(inputs, None)
                novel_task_preds.append(outputs['my_decision'].argmax(dim=1))
                novel_task_targets.append(torch.argmax(target_labels, dim=1))
            
            if novel_task_preds:
                novel_task_preds = torch.cat(novel_task_preds)
                novel_task_targets = torch.cat(novel_task_targets)
                novel_task_accuracy = (novel_task_preds == novel_task_targets).float().mean().item()
    
    test_loss = sum(test_losses) / len(test_losses)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    accuracy = (all_preds == all_targets).float().mean().item()
    
    model.train()
    return {'accuracy': accuracy, 'novel_accuracy': novel_accuracy, 'novel_task_accuracy': novel_task_accuracy, 'loss': test_loss, 'stage': stage_name}

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

    torch.manual_seed(42+repetition)
    torch.cuda.manual_seed(42+repetition)
    torch.cuda.manual_seed_all(42+repetition)
    np.random.seed(42+repetition)

    model_kwargs['output_len'] = 5
    model_kwargs['channels'] = 5
    model_kwargs['oracle_is_target'] = oracle_is_target

    device = torch.device('cuda' if use_cuda else 'cpu')
    total_steps = 8
    epoch_steps = 8
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
    else:
        module_labels = {
            'my_decision': ['correct-loc'],
        }
    module_label_data = {module: [] for module in module_labels.keys()} 
    batch_size = model_kwargs['batch_size']
    model_kwargs['batch_size'] = 1024
    lr = model_kwargs['lr']
    batch_size = model_kwargs['batch_size']
    model = load_model(model_type, model_kwargs, device)

    stage_results = []

    real_batch = 0

    criterion = nn.CrossEntropyLoss()
    oracle_criterion = nn.MSELoss()
    
    model.vision_prob = 1.0

    epoch_losses_df = pd.DataFrame(columns=['Batch', 'Loss', 'Accuracy', 'Novel_Accuracy', 'Novel_Task_Accuracy'])

    t = tqdm.trange(total_batches)

    for stage_config in curriculum_config.curriculum_stages:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.95, 0.999))
        print('started curricular stage', stage_config['stage_name'])
        stage_train_sets = apply_curriculum_stage(model, stage_config, train_sets_dict)
        batches = stage_config['batches']
        print('training on these sets: ', stage_train_sets)

        trainable = [name for name, module in model.get_module_dict().items() if module is not None and any(p.requires_grad for p in module.parameters())]
        print('trainable modules:', trainable)
        print('module configs', model.kwargs['module_configs'])

        from experiments import init_regimes
        _, _, _, fregimes, _, _, _, _, _, _ = init_regimes()
        all_s3_regimes = fregimes['s3']
        novel_regimes = [r for r in all_s3_regimes if r not in stage_train_sets]

        novel_eval_data, novel_eval_labels = [], []
        if novel_regimes:
            for regime_name in novel_regimes[:3]:
                dir = os.path.join(load_path, regime_name)
                if os.path.exists(dir):
                    regime_data = np.load(os.path.join(dir, 'obs.npz'), mmap_mode='r')['arr_0'][:100]
                    regime_labels_raw = np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0'][:100]
                    
                    if last_timestep and len(regime_labels_raw.shape) > 2:
                        regime_labels = regime_labels_raw[..., -1, :]
                    else:
                        regime_labels = regime_labels_raw.reshape(-1, 25)
                    
                    novel_eval_data.append(regime_data)
                    novel_eval_labels.append(regime_labels)

        if novel_eval_data:
            novel_params = [np.zeros((len(d), 0)) for d in novel_eval_data]
            novel_oracles = [np.zeros((len(d), 0)) for d in novel_eval_data]
            
            novel_indices = list(range(sum(len(d) for d in novel_eval_data)))
            novel_dataset = TrainDatasetBig(novel_eval_data, novel_eval_labels, novel_params, novel_oracles, novel_indices)
            novel_loader = DataLoader(novel_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
            novel_task_loader = DataLoader(novel_task_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
                'Novel_Task_Accuracy': baseline_results['novel_task_accuracy']
            }
            new_row = pd.DataFrame([new_row_data])
            epoch_losses_df = pd.concat([epoch_losses_df, new_row], ignore_index=True)
            print(f"Acc: {baseline_results['accuracy']:.4f}, Nacc: {baseline_results['novel_accuracy']:.4f}, NTacc: {baseline_results['novel_task_accuracy']:.4f}, L: {baseline_results['loss']:.4f}, Vis:", model.vision_prob, 'P:', save_path, repetition)
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

        module_sizes = {'op_belief': 12, 'my_decision': 5, 'op_decision': 5, 'op_belief_t': 60}
        module_names = module_labels.keys()

        first_batch_processed = False
        module_ranges = {}

        for batch in range(batches):
            real_batch += 1
            total_loss = 0.0
            try:
                inputs, target_labels, _, oracles, _ = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                inputs, target_labels, _, oracles, _ = next(iter_loader)
            inputs, target_labels, oracles = inputs.to(device), target_labels.float().to(device), oracles.to(device)
            target_labels = target_labels[:,:-1]
            outputs = model(inputs, None)

            if not oracle_labels:
                loss = criterion(outputs['my_decision'], torch.argmax(target_labels, dim=1))

            if not first_batch_processed:
                #print(oracles.shape, oracles[0:100])
                flat_oracle = oracles.reshape(oracles.size(0), -1)
                total_oracle_dims = flat_oracle.size(1)
                pos = 0
                for module_name in module_labels.keys():
                    #print(module_name)
                    module_output = outputs[module_name]
                    flattened_output_size = module_sizes[module_name]
                    module_ranges[module_name] = (pos, pos + flattened_output_size)
                    pos += flattened_output_size

                first_batch_processed = True

            oracle_loss = 0.0

            #print(oracle_labels)

            if batch % 100 == 0 and batch > 0:
                with torch.no_grad():
                    original_use_neural = model.op_belief.use_neural
                    
                    model.op_belief.use_neural = False
                    hardcoded_outputs = model(inputs, None)
                    hardcoded_beliefs = hardcoded_outputs['op_belief']
                    oghardcoded_beliefs = hardcoded_outputs['og_op_belief']
                    hardcoded_decision = hardcoded_outputs['my_decision']
                    
                    model.op_belief.use_neural = True
                    neural_outputs = model(inputs, None)
                    neural_beliefs = neural_outputs['op_belief']
                    neural_decision = neural_outputs['my_decision']
                    
                    model.op_belief.use_neural = original_use_neural
                    
                    for i in range(min(3, inputs.shape[0])):
                        print(f"\nSample {i}:")
                        print(f"Oracle target indices: {target_indices[i]}")
                        print(f"OG HC beliefs:")
                        print(f"  Large: {oghardcoded_beliefs[i, 0]}")
                        print(f"  Small: {oghardcoded_beliefs[i, 1]}")
                        print(f"Hardcoded beliefs:")
                        print(f"  Large: {hardcoded_beliefs[i, 0]}")
                        print(f"  Small: {hardcoded_beliefs[i, 1]}")
                        print(f"Neural beliefs:")
                        print(f"  Large: {neural_beliefs[i, 0]}")
                        print(f"  Small: {neural_beliefs[i, 1]}")
                        print(f"Hardcoded decision: {hardcoded_decision[i].argmax().item()}")
                        print(f"Neural decision: {neural_decision[i].argmax().item()}")
                        print(f"Target decision: {torch.argmax(target_labels[i]).item()}")

            if oracle_labels:
                flat_oracle = oracles.reshape(oracles.size(0), -1)
                oracle_criterion = nn.CrossEntropyLoss()
                
                for module_name in module_labels.keys():
                    if module_name in outputs and module_name in module_ranges:
                        oracle_target = flat_oracle
                        oracle_target_reshaped = oracle_target.reshape(-1, 2, 5)
                        target_indices = torch.argmax(oracle_target_reshaped, dim=2)
                        no_treat_mask = (oracle_target_reshaped.sum(dim=2) == 0)
                        
                        if batch == 0:
                            print(f"Target indices shape: {target_indices.shape}")
                            print(f"Target indices[0]: {target_indices[0]}")
                            print(f"No treat mask[0]: {no_treat_mask[0]}")
                        
                        model_output_raw = outputs[module_name]
                        
                        if module_name == 'op_belief_t':
                            model_output = model_output_raw.permute(0, 2, 1, 3).reshape(-1, 2, 6)
                        else:
                            batch_size = model_output_raw.shape[0]
                            model_output = model_output_raw.unsqueeze(1).expand(batch_size, 5, 2, 6).reshape(-1, 2, 6)
                        
                        model_output_flat = model_output.reshape(-1, 6)
                        
                        target_indices[no_treat_mask] = 5
                        target_indices_flat = target_indices.reshape(-1)
                        
                        if batch == 0:
                            print(f"Model output shape: {model_output.shape}")
                            print(f"Model output flat shape: {model_output_flat.shape}")
                            print(f"Target indices flat shape: {target_indices_flat.shape}")
                        
                        oracle_loss += oracle_criterion(model_output_flat, target_indices_flat)

            total_loss = oracle_loss


            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            if (batch+1) % epoch_length == 0:
                if batch > 1:
                    t.update(epoch_length)

            if record_loss and (((real_batch) % epoch_length_val == 0) or (real_batch == total_batches - 1)):

                stage_results = evaluate_model_stage(model, test_loader, novel_loader, novel_task_loader, criterion, device, stage_config['stage_name'])
                
                new_row_data = {
                    'Batch': real_batch,
                    'Loss': stage_results['loss'],
                    'OLoss': oracle_loss.mean().cpu().item(),
                    'Accuracy': stage_results['accuracy'],
                    'Novel_Accuracy': stage_results['novel_accuracy'],
                    'Novel_Task_Accuracy': stage_results['novel_task_accuracy']
                }
                new_row = pd.DataFrame([new_row_data])
                
                epoch_losses_df = pd.concat([epoch_losses_df, new_row], ignore_index=True)
                model.train()

                final_accuracy = stage_results['accuracy']

                print(f"Acc: {stage_results['accuracy']:.4f}, Nacc: {stage_results['novel_accuracy']:.4f}, NTacc: {stage_results['novel_task_accuracy']:.4f}, L: {stage_results['loss']:.4f}, O: {oracle_loss.mean().cpu().item():.4f} Vis:",
                    model.vision_prob, 'Path:', save_path, repetition)

                if save_models and real_batch == total_batches - 1:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-{stage_config["stage_name"]}.pt'))

    if record_loss:
        epoch_losses_df.to_csv(os.path.join(save_path, f'losses-{repetition}.csv'), index=False)
    return final_accuracy