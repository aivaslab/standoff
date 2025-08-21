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
    
    if len(parts) >= 3 and parts[0] == 'end2end':
        dataset = parts[1]
        remainder = '_'.join(parts[2:])
        
        if '-' in remainder:
            regime_and_rest = remainder.split('-', 1)
            regime = regime_and_rest[0]
            rest = regime_and_rest[1]
            
            if '-sym-' in rest:
                output_type = rest.split('-sym-')[0]
                model_type = rest.split('-sym-')[1]
                sym_flag = True
            else:
                rest_parts = rest.split('-')
                output_type = rest_parts[0]
                model_type = '-'.join(rest_parts[1:]) if len(rest_parts) > 1 else 'unknown'
                sym_flag = False
        else:
            regime = remainder
            output_type = 'unknown'
            model_type = 'unknown'
            sym_flag = False
    else:
        dataset = 'unknown'
        regime = 'unknown'
        output_type = 'unknown'
        model_type = folder_name
        sym_flag = False
    
    return dataset, regime, output_type, model_type, sym_flag

def get_display_name(folder_name):
    if folder_name.startswith('end2end_'):
        return folder_name[8:]
    return folder_name

def get_short_name(folder_name):
    display_name = get_display_name(folder_name)
    if '-' in display_name:
        return display_name.split('-')[-1]
    else:
        parts = display_name.split('_')
        return parts[-1] if parts else display_name

def plot_grouped_accuracies_subplots(grouped_data, group_name, basepath, datasets, color_map):
    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    
    legend_handles = []
    legend_labels = []
    
    for dataset_idx, dataset in enumerate(datasets):
        ax = axes[dataset_idx]
        
        if dataset not in grouped_data:
            ax.set_title(f'{dataset} - No Data')
            continue
        
        folders_in_dataset = grouped_data[dataset]
        
        for folder, data in folders_in_dataset.items():
            if data['batches'] is not None and data['accuracy_means'] is not None:
                short_name = get_short_name(folder)
                color = color_map[short_name]
                batches = np.array(data['batches'])
                
                train_means = np.array(data['accuracy_means'])
                train_stds = np.array(data['accuracy_stds'])
                train_line, = ax.plot(batches, train_means, linewidth=2, color=color)
                ax.fill_between(batches, train_means - train_stds, train_means + train_stds, 
                               alpha=0.3, color=color)
                
                novel_means = np.array(data['novel_accuracy_means'])
                novel_stds = np.array(data['novel_accuracy_stds'])
                ax.plot(batches, novel_means, linewidth=2, 
                       color=color, linestyle='-.')
                ax.fill_between(batches, novel_means - novel_stds, novel_means + novel_stds, 
                               alpha=0.2, color=color)
                
                if data['val_accuracy_means'] is not None and dataset != 's3':
                    val_means = np.array(data['val_accuracy_means'])
                    val_stds = np.array(data['val_accuracy_stds'])
                    ax.plot(batches, val_means, linewidth=2, 
                           color=color, linestyle='--')
                    ax.fill_between(batches, val_means - val_stds, val_means + val_stds, 
                                   alpha=0.15, color=color)
                
                if dataset_idx == 0 and short_name not in legend_labels:
                    legend_handles.append(train_line)
                    legend_labels.append(short_name)
        
        ax.set_xlabel('Batch')
        if dataset_idx == 0:
            ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset}')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    if legend_handles:
        fig.legend(legend_handles, legend_labels, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    fig.suptitle(f'Grouped by {group_name}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(basepath, f"grouped_{group_name.replace(' ', '_').replace('/', '_')}.png"), bbox_inches='tight')
    plt.close()

base_path = './new/exp_19-L'
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
        
        dataset, regime, output_type, model_type, sym_flag = get_folder_info(folder)
        folder_data[folder] = {
            'batches': batches,
            'accuracy_means': accuracy_means,
            'accuracy_stds': accuracy_stds,
            'novel_accuracy_means': novel_accuracy_means,
            'novel_accuracy_stds': novel_accuracy_stds,
            'val_accuracy_means': val_accuracy_means,
            'val_accuracy_stds': val_accuracy_stds,
            'dataset': dataset,
            'regime': regime,
            'output_type': output_type,
            'model_type': model_type,
            'sym_flag': sym_flag
        }

all_short_names = set()
for folder in folder_data.keys():
    all_short_names.add(get_short_name(folder))

color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
color_map = {name: color_list[i % len(color_list)] for i, name in enumerate(sorted(all_short_names))}

datasets = sorted(set(data['dataset'] for data in folder_data.values()))

output_type_sym_groups = defaultdict(lambda: defaultdict(dict))

for folder, data in folder_data.items():
    output_sym_key = f"{data['output_type']}_{'sym' if data['sym_flag'] else 'nosym'}"
    output_type_sym_groups[output_sym_key][data['dataset']][folder] = data

for output_sym_key, dataset_groups in output_type_sym_groups.items():
    plot_grouped_accuracies_subplots(dataset_groups, f"output_{output_sym_key}", base_path, datasets, color_map)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(base_path, 'final_accuracies_summary.csv'), index=False)
print(f"Summary saved to final_accuracies_summary.csv")
print(results_df)