
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

def evaluate_model_stage(model, test_loader, novel_loader, criterion, device, stage_name=""):
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
    
    test_loss = sum(test_losses) / len(test_losses)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    accuracy = (all_preds == all_targets).float().mean().item()
    
    model.train()
    return {'accuracy': accuracy, 'novel_accuracy': novel_accuracy, 'loss': test_loss, 'stage': stage_name}

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
        #print('current temp/prob:', current_temp, current_prob)

    def get_temp(self):
        progress = min(1.0, self.rate * self.current_step / self.total_steps)
        cosine_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return self.start_temp + (self.end_temp - self.start_temp) * cosine_progress, self.vision_prob_start + (self.vision_prob_end - self.vision_prob_start) * cosine_progress


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

    model_kwargs['output_len'] = 5  # np.prod(labels.shape[1:])
    model_kwargs['channels'] = 5  # np.prod(params.shape[2])
    model_kwargs['oracle_is_target'] = oracle_is_target

    device = torch.device('cuda' if use_cuda else 'cpu')
    total_steps = 20
    epoch_steps = 20
    total_batches = sum(stage['batches'] for stage in curriculum_config.curriculum_stages)
    from supervised_learning_main import filter_indices, load_model

    module_labels = {
        #'treat_perception': ['loc-large', 'loc-small'],  
        #'op_vision': 'vision',
        #'presence_perception': 'opponents',
        #'my_belief': ['loc-large', 'loc-small'],  
        'op_belief': ['b-loc-large', 'b-loc-small'],  
        'my_decision': ['correct-loc'],
        'op_decision': ['target-loc']
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
    oracle_criterion = nn.MSELoss(reduction='mean')
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.95)
    model.vision_prob = 1.0

    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.95, 0.999))

    epoch_losses_df = pd.DataFrame(columns=['Batch', 'Loss', 'Accuracy'] + [f"{module}_mse" for module in module_labels.keys()])

    t = tqdm.trange(total_batches)

    for stage_config in curriculum_config.curriculum_stages:
        print('started curricular stage', stage_config['stage_name'])
        stage_train_sets = apply_curriculum_stage(model, stage_config, train_sets_dict)
        batches = stage_config['batches']
        print('training on these sets: ', stage_train_sets)

        trainable = [name for name, module in model.get_module_dict().items() if any(p.requires_grad for p in module.parameters())]
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

        for data_name in stage_train_sets:
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
                            #print('concating', label_name, label_data.shape)
                            label_data = label_data[..., -1, :5]
                        elif len(label_data.shape) > 2:
                            label_data = label_data.reshape(label_data.shape[0], -1)
                        if label_data.ndim == 1:
                            label_data = label_data.reshape(label_data.shape[0], 1)
                            
                        data_arrays.append(label_data)
                        #print('appending to data array', module, label_name, label_data.shape)
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
            baseline_results = evaluate_model_stage(model, test_loader, novel_loader, criterion, device, "baseline")
            new_row_data = {
                'Batch': real_batch,
                'Loss': baseline_results['loss'],
                'Accuracy': baseline_results['accuracy'],
                'Novel_Accuracy': baseline_results['novel_accuracy']
            }
            new_row = pd.DataFrame([new_row_data])
            print(f"Acc: {baseline_results['accuracy']:.4f}, Nacc: {baseline_results['novel_accuracy']:.4f}, L: {baseline_results['loss']:.4f}, Vis:",
                    model.vision_prob, 'Path:', os.path.basename(save_path), repetition)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  # can't use more on windows

        #print('oracle length:', model_kwargs['oracle_len'])
        
        #criterion = torchvision.ops.sigmoid_focal_loss
        #sigmoid_scheduler = SigmoidTempScheduler(model, start_temp=90.0, end_temp=90.0, total_steps=total_steps, vision_prob_start=model.vision_prob, vision_prob_end=model.vision_prob_base, rate=2.0)
        #scheduler = ExponentialLR(optimizer, gamma=0.92)
        #scheduler = OneCycleLR(
        #    optimizer,
        #    max_lr=1e-2,
        #    total_steps=total_steps,
        #    pct_start=0.3,
        #    div_factor=1,
        #    final_div_factor=1,
        #)

        if False:  # if loading previous model, only for progressions
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

        if save_models:
            os.makedirs(save_path, exist_ok=True)
            torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-prior.pt'))

        module_sizes = {'op_belief': 10, 'my_decision': 5, 'op_decision': 5}

        #belief_range, vision_range, decision_range = module_ranges['op_belief'], module_ranges['op_vision'], module_ranges['op_decision']
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
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #with profiler.profile(use_cuda=True) as prof:
            outputs = model(inputs, None)

            #print(inputs.shape, str(model))
            #print(prof.table())
            #print(prof.key_averages().table(sort_by="cuda_time_total"))
            #print('xxxxxxx', outputs)
            #loss = criterion(outputs['my_decision'], target_labels, reduction="mean") # used for Focal
            loss = criterion(outputs['my_decision'], torch.argmax(target_labels, dim=1)) # used for bcd

            if not first_batch_processed:
                flat_oracle = oracles.reshape(oracles.size(0), -1)
                total_oracle_dims = flat_oracle.size(1)
                
                pos = 0
                for module_name in module_labels.keys():
                    module_output = outputs[module_name]
                    #flattened_output_size = module_output.reshape(module_output.size(0), -1).size(1)
                    flattened_output_size = module_sizes[module_name]
                    #print(module_name, module_output[0], pos, flattened_output_size)
                    module_ranges[module_name] = (pos, pos + flattened_output_size)
                    pos += flattened_output_size

                first_batch_processed = True

            flat_oracle = oracles.reshape(oracles.size(0), -1)
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            #t.update(1)

            if (batch+1) % epoch_length == 0:
                if batch > 1:
                    #scheduler.step()
                    #sigmoid_scheduler.step()
                    t.update(epoch_length)

            if record_loss and (((real_batch) % epoch_length_val == 0) or (batch == batches - 1)):
                stage_results = evaluate_model_stage(model, test_loader, novel_loader, criterion, device, stage_config['stage_name'])
                
                new_row_data = {
                    'Batch': real_batch,
                    'Loss': stage_results['loss'],
                    'Accuracy': stage_results['accuracy'],
                    'Novel_Accuracy': stage_results['novel_accuracy']
                }
                #for module, mse_val in module_mse_values.items():
                #    new_row_data[f"{module}_mse"] = np.mean(mse_val)
                new_row = pd.DataFrame([new_row_data])
                
                epoch_losses_df = pd.concat([epoch_losses_df, new_row], ignore_index=True)
                model.train()

                final_accuracy = stage_results['accuracy']

                print(f"Acc: {stage_results['accuracy']:.4f}, Nacc: {stage_results['novel_accuracy']:.4f}, L: {stage_results['loss']:.4f}, Vis:",
                    model.vision_prob, 'Path:', os.path.basename(save_path), repetition)
                #print("Module MSE values:")
                #for module, mse_val in module_mse_values.items():
                #    print(f"  {module}: {np.mean(mse_val):.4f}")

                if save_models and real_batch == total_batches - 1:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-{stage_config["stage_name"]}.pt'))


    if record_loss:
        epoch_losses_df.to_csv(os.path.join(save_path, f'losses-{repetition}.csv'), index=False)
    return final_accuracy