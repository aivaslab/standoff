import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict

def plot_accuracy_with_std_data_only(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "losses*.csv"))
    
    if not csv_files:
        return None, None, None, None, None, None, None, None, None, None, None
    
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, keys=range(len(all_data)))
    
    grouped = combined_df.groupby('Batch')
    
    batches = sorted(grouped.groups.keys())
    accuracy_means = []
    accuracy_stds = []
    novel_accuracy_means = []
    novel_accuracy_stds = []
    val_accuracy_means = []
    val_accuracy_stds = []
    
    has_val_accuracy = 'Val_Accuracy' in combined_df.columns
    
    for batch in batches:
        batch_data = grouped.get_group(batch)
        accuracy_means.append(batch_data['Accuracy'].mean())
        accuracy_stds.append(batch_data['Accuracy'].std())
        novel_accuracy_means.append(batch_data['Novel_Accuracy'].mean())
        novel_accuracy_stds.append(batch_data['Novel_Accuracy'].std())
        
        if has_val_accuracy:
            val_accuracy_means.append(batch_data['Val_Accuracy'].mean())
            val_accuracy_stds.append(batch_data['Val_Accuracy'].std())
    
    final_accuracy_mean = accuracy_means[-1] if accuracy_means else None
    final_accuracy_std = accuracy_stds[-1] if accuracy_stds else None
    final_novel_accuracy_mean = novel_accuracy_means[-1] if novel_accuracy_means else None
    final_novel_accuracy_std = novel_accuracy_stds[-1] if novel_accuracy_stds else None
    
    return (final_accuracy_mean, final_accuracy_std, final_novel_accuracy_mean, final_novel_accuracy_std, 
            batches, accuracy_means, accuracy_stds, val_accuracy_means if has_val_accuracy else None, 
            novel_accuracy_means, novel_accuracy_stds, val_accuracy_stds if has_val_accuracy else None)

def get_folder_info(folder_name):
    parts = folder_name.split('_')
    
    if len(parts) >= 3:
        training_mode = parts[0]
        dataset = parts[1]
        model_type = '_'.join(parts[2:])
    else:
        training_mode = 'unknown'
        dataset = 'unknown'
        model_type = folder_name
    
    return dataset, training_mode, model_type

def plot_grouped_accuracies(grouped_folders, group_name, basepath):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_folders)))
    
    for i, (folder, data) in enumerate(grouped_folders.items()):
        if data['batches'] is not None and data['accuracy_means'] is not None:
            color = colors[i]
            batches = np.array(data['batches'])
            
            train_means = np.array(data['accuracy_means'])
            train_stds = np.array(data['accuracy_stds'])
            ax.plot(batches, train_means, label=f"{folder} (train)", linewidth=2, color=color)
            ax.fill_between(batches, train_means - train_stds, train_means + train_stds, 
                           alpha=0.3, color=color)
            
            novel_means = np.array(data['novel_accuracy_means'])
            novel_stds = np.array(data['novel_accuracy_stds'])
            ax.plot(batches, novel_means, label=f"{folder} (novel)", linewidth=2, 
                   color=color, linestyle='-.')
            ax.fill_between(batches, novel_means - novel_stds, novel_means + novel_stds, 
                           alpha=0.2, color=color)
            
            if data['val_accuracy_means'] is not None and data['dataset'] != 's3':
                val_means = np.array(data['val_accuracy_means'])
                val_stds = np.array(data['val_accuracy_stds'])
                ax.plot(batches, val_means, label=f"{folder} (val)", linewidth=2, 
                       color=color, linestyle='--')
                ax.fill_between(batches, val_means - val_stds, val_means + val_stds, 
                               alpha=0.15, color=color)
    
    ax.set_xlabel('Batch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Grouped by {group_name}')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, f"grouped_{group_name.replace(' ', '_').replace('/', '_')}.png"), bbox_inches='tight')
    plt.close()

base_path = './new/exp_1-L'
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

results = []
folder_data = {}

for folder in folders:
    print(f"Processing folder: {folder}")
    folder_path = os.path.join(base_path, folder)
    result = plot_accuracy_with_std_data_only(folder_path)
    (final_acc_mean, final_acc_std, final_novel_mean, final_novel_std, 
     batches, accuracy_means, accuracy_stds, val_accuracy_means, 
     novel_accuracy_means, novel_accuracy_stds, val_accuracy_stds) = result
    
    if final_acc_mean is not None:
        results.append({
            'folder': folder,
            'final_accuracy_mean': final_acc_mean,
            'final_accuracy_std': final_acc_std,
            'final_novel_accuracy_mean': final_novel_mean,
            'final_novel_accuracy_std': final_novel_std
        })
        
        dataset, training_mode, model_type = get_folder_info(folder)
        folder_data[folder] = {
            'batches': batches,
            'accuracy_means': accuracy_means,
            'accuracy_stds': accuracy_stds,
            'novel_accuracy_means': novel_accuracy_means,
            'novel_accuracy_stds': novel_accuracy_stds,
            'val_accuracy_means': val_accuracy_means,
            'val_accuracy_stds': val_accuracy_stds,
            'dataset': dataset,
            'training_mode': training_mode,
            'model_type': model_type
        }

dataset_training_groups = defaultdict(lambda: defaultdict(dict))
dataset_model_groups = defaultdict(lambda: defaultdict(dict))

for folder, data in folder_data.items():
    dataset_training_groups[data['dataset']][data['training_mode']][folder] = data
    dataset_model_groups[data['dataset']][data['model_type']][folder] = data

for dataset, training_groups in dataset_training_groups.items():
    for training_mode, folders_in_group in training_groups.items():
        plot_grouped_accuracies(folders_in_group, f"dataset_{dataset}/training_{training_mode}", base_path)

for dataset, model_groups in dataset_model_groups.items():
    for model_type, folders_in_group in model_groups.items():
        plot_grouped_accuracies(folders_in_group, f"dataset_{dataset}/model_{model_type}", base_path)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(base_path, 'final_accuracies_summary.csv'), index=False)
print(f"Summary saved to final_accuracies_summary.csv")
print(results_df)