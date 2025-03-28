import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np

def get_simplified_model_system():
    prefixes = ['treat', 'vision', 'presence', 'perception', 'belief', 'combiner', 'decision', 'all']
    suffixes = ['my', 'op', 'split', 'shared', 'detach']
    
    model_mapping = {
        'a-hardcoded': 'rule-based',
        'a-hardcoded-v50-b5': 'rule-based-mv',
    }
    
    for prefix in prefixes:
        for suffix in suffixes:
            model_mapping[f'a-mix-n-{prefix}-{suffix}'] = f'{prefix}-{suffix}'
            model_mapping[f'a-mix-r-{prefix}-{suffix}-100'] = f'{prefix}-{suffix}'
            model_mapping[f'a-mix-n-{prefix}-{suffix}-v50-b5'] = f'{prefix}-{suffix}-mv'
    
    model_order = ['rule-based', 'rule-based-mv']
    
    for prefix in prefixes:
        for suffix in suffixes:
            base_name = f'{prefix}-{suffix}'
            model_order.append(base_name)
            model_order.append(f'{base_name}-mv')
    
    return model_mapping, model_order

def load_and_plot_heatmap(table_path, save_dir, param='test_group', label='accuracy'):
    df = pd.read_csv(table_path)
    
    model_mapping, model_order = get_simplified_model_system()
    df['base_model'] = df['regime'].apply(lambda x: x.rsplit('-loc-', 1)[0])
    print(df['base_model'].unique())
    df['train_set'] = df['regime'].apply(lambda x: x.split('-')[-1])
    df['display_name'] = df['base_model'].apply(lambda x: model_mapping.get(x))
    
    def get_model_order(name):
        if name in model_order:
            return model_order.index(name)
        return 999  
    
    df['order'] = df['display_name'].apply(get_model_order)

    train_group_mapping = {
        's1': 's1',
        's2': 's2',
        's21': 's21',
        's3': 's3'
        }
    df['train_set'] = df['train_set'].map(lambda x: train_group_mapping.get(x, x))

    test_group_mapping = {
        's1': 'Solo',
        's2': 'Informed',
        's21': 'ToM-Simple',
        's3': 'ToM-Complex'
        }
    df['test_group'] = df['test_group'].map(lambda x: test_group_mapping.get(x, x))
    
    desired_order = ['Solo', 'Informed', 'ToM-Simple', 'ToM-Complex'] if param == 'test_group' else sorted(df[param].unique())
    train_sets = sorted(df['train_set'].unique())
    
    fig = plt.figure(figsize=(len(desired_order) * 1.5, len(train_sets) * 5))
    gs = GridSpec(len(train_sets), 1, figure=fig, hspace=0.1)
    
    for i, train_set in enumerate(train_sets):
        train_df = df[df['train_set'] == train_set].sort_values('order')
        
        ordered_models = []
        for model in model_order:
            if model in train_df['display_name'].values:
                print(model)
                ordered_models.append(model)
            print('f', model)
        
        mean_pivot = pd.pivot_table(train_df, values='accuracy mean', index='display_name', columns=param)
        std_pivot = pd.pivot_table(train_df, values='accuracy std', index='display_name', columns=param)
        
        mean_pivot = mean_pivot.reindex(columns=desired_order)
        std_pivot = std_pivot.reindex(columns=desired_order)
        mean_pivot = mean_pivot.reindex(index=ordered_models)
        std_pivot = std_pivot.reindex(index=ordered_models)
        
        formatted_df = mean_pivot.copy()
        for col in formatted_df.columns:
            for idx in formatted_df.index:
                if pd.notnull(mean_pivot.loc[idx, col]) and pd.notnull(std_pivot.loc[idx, col]):
                    formatted_df.loc[idx, col] = f"{mean_pivot.loc[idx, col]:.2f} ({std_pivot.loc[idx, col]:.2f})"
        
        ax = fig.add_subplot(gs[i])
        
        sns.heatmap(mean_pivot, annot=formatted_df, 
                   fmt='', cmap='RdBu', linewidths=0.5, linecolor='white', 
                   vmin=0, vmax=1, cbar=False, ax=ax)
        
        ax.set_ylabel(f"", fontsize=14)#{train_set}
        if i == len(train_sets) - 1:
            ax.set_xlabel("Test Set", fontsize=12)
        else:
            ax.set_xlabel("")
        ax.set_ylabel(f"{train_set}", fontsize=12)
    
    plt.tight_layout()
    
    plot_save_path = os.path.join(save_dir, f'{label}_{param}_heatmaps_by_trainset.png')
    print(f'Saving figure to {plot_save_path}')
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=150)
    plt.close()

    fig = plt.figure(figsize=(len(desired_order) * 1.5, len(train_sets) * 4))
    gs = GridSpec(len(train_sets), 1, figure=fig, hspace=0.2)

    familiar_columns = {
        0: 1, 
        1: 2, 
        2: 3,
    }

    for i, train_set in enumerate(train_sets[:-1]):
        train_df = df[df['train_set'] == train_set].sort_values('order')
        
        ordered_models = []
        for model in model_order:
            if model in train_df['display_name'].values:
                ordered_models.append(model)
        
        mean_pivot = pd.pivot_table(train_df, values='accuracy mean', index='display_name', columns=param)
        std_pivot = pd.pivot_table(train_df, values='accuracy std', index='display_name', columns=param)
        
        mean_pivot = mean_pivot.reindex(columns=desired_order)
        std_pivot = std_pivot.reindex(columns=desired_order)
        mean_pivot = mean_pivot.reindex(index=ordered_models)
        std_pivot = std_pivot.reindex(index=ordered_models)
        
        formatted_df = mean_pivot.copy()
        for col in formatted_df.columns:
            for idx in formatted_df.index:
                if pd.notnull(mean_pivot.loc[idx, col]) and pd.notnull(std_pivot.loc[idx, col]):
                    formatted_df.loc[idx, col] = f"{mean_pivot.loc[idx, col]:.2f} ({std_pivot.loc[idx, col]:.2f})"
        
        ax = fig.add_subplot(gs[i])
        
        sns.heatmap(mean_pivot, annot=formatted_df, 
                   fmt='', cmap='RdBu', linewidths=0.5, linecolor='white', 
                   vmin=0, vmax=1, cbar=False, ax=ax)
        
        familiar_count = familiar_columns.get(i, 1)
        ax.axvline(x=familiar_count, color='white', linestyle='-', linewidth=4)
        
        name = test_group_mapping[train_set] + "-Train" 
        ax.text(familiar_count/2, -0.75, 'Familiar', ha='center', va='top', fontsize=14)
        ax.text((len(desired_order) + familiar_count)/2, -0.75, 'Novel', ha='center', va='top', fontsize=14)
        ax.text(-1.1, len(train_df)/4 - 10, name, ha='center', va='top', fontsize=16, rotation=90)
        
        ax.set_ylabel(f"", fontsize=14)
        if i == len(train_sets) - 1:
            ax.set_xlabel("Test Set", fontsize=12)
        else:
            ax.set_xlabel("")

    plt.tight_layout()
    plot_save_path = os.path.join(save_dir, f'{label}_{param}_heatmaps_by_fam.png')
    print(f'Saving figure to {plot_save_path}')
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=150)
    plt.close()

    fig = plt.figure(figsize=(len(desired_order) * 3, len(train_sets) * 3.5))
    gs = GridSpec(len(train_sets), 1, figure=fig, hspace=0.01)
    
    for i, train_set in enumerate(train_sets):
        train_df = df[df['train_set'] == train_set].sort_values('order')
        
        ordered_models = []
        for model in model_order:
            if model in train_df['display_name'].values:
                ordered_models.append(model)
        
        mean_pivot = pd.pivot_table(train_df, values='accuracy mean', index='display_name', columns=param)
        std_pivot = pd.pivot_table(train_df, values='accuracy std', index='display_name', columns=param)
        
        mean_pivot = mean_pivot.reindex(columns=desired_order)
        std_pivot = std_pivot.reindex(columns=desired_order)
        mean_pivot = mean_pivot.reindex(index=ordered_models)
        std_pivot = std_pivot.reindex(index=ordered_models)
        
        # Add subplot
        ax = fig.add_subplot(gs[i])
        
        # Position bars
        bar_width = 0.8 / len(desired_order)
        x = np.arange(len(ordered_models))
        
        # Create grouped bars for each test group
        for j, test_group in enumerate(desired_order):
            if test_group in mean_pivot.columns:
                means = mean_pivot[test_group].values
                stds = std_pivot[test_group].values
                
                offset = (j - len(desired_order) / 2 + 0.5) * bar_width
                bars = ax.bar(x + offset, means, bar_width, label=test_group, 
                            yerr=stds, capsize=2, alpha=0.8)
                
                # Add value labels on top of bars
                #for k, bar in enumerate(bars):
                #    if not np.isnan(means[k]) and means[k] > 0.1:  # Only add text if value is significant
                #        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                #              f"{means[k]:.2f}", ha='center', va='bottom', fontsize=7, rotation=90)
        
        # Set labels and options
        if i == 0:
            ax.legend(title=param.replace('_', ' ').title(), loc='center right', 
                     bbox_to_anchor=(1.05, 1.15), ncol=1)
        
        ax.set_ylabel(f"{train_set}\nAccuracy", fontsize=12)
        ax.set_ylim(0, 1.0)  # Set y-axis limit
        
        # Only add x-axis labels for the bottom subplot
        if i == len(train_sets) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(ordered_models, rotation=45)
            ax.set_xlabel("Model Type", fontsize=12)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])
        
        # Add gridlines for better readability
        #ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add horizontal lines at important accuracy levels
        #for y_val in [0.25, 0.5, 0.75]:
        #    ax.axhline(y=y_val, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    bar_save_path = os.path.join(save_dir, f'{label}_{param}_bargraphs_by_trainset.png')
    print(f'Saving bar graph to {bar_save_path}')
    plt.savefig(bar_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    plt.figure(figsize=(max(15, len(model_order) * 0.7), 10))
    ax = plt.gca()
    
    focus_test_group = desired_order[0]
    
    bar_width = 0.8 / len(train_sets)
    x = np.arange(len(model_order))
    
    for i, train_set in enumerate(train_sets):
        train_df = df[df['train_set'] == train_set].sort_values('order')
        
        all_models_data = {}
        for model in model_order:
            model_data = train_df[train_df['display_name'] == model]
            if not model_data.empty and focus_test_group in model_data[param].values:
                test_data = model_data[model_data[param] == focus_test_group]
                if not test_data.empty:
                    all_models_data[model] = {
                        'mean': test_data['accuracy mean'].values[0],
                        'std': test_data['accuracy std'].values[0]
                    }
            else:
                all_models_data[model] = {'mean': np.nan, 'std': np.nan}
        
        means = [all_models_data[model]['mean'] for model in model_order]
        stds = [all_models_data[model]['std'] for model in model_order]
        
        offset = (i - len(train_sets) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, label=train_set, 
                     yerr=stds, capsize=3, alpha=0.8)
    
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Accuracy by Model Type for Test Group {focus_test_group}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=90)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Train Set', loc='center right')
    #ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    combined_bar_save_path = os.path.join(save_dir, f'{label}_{param}_combined_bargraph.png')
    print(f'Saving combined bar graph to {combined_bar_save_path}')
    plt.savefig(combined_bar_save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return plot_save_path


def create_comparison_bar_graph(table_path, save_dir):
    df = pd.read_csv(table_path)
    
    model_mapping, model_order = get_simplified_model_system()
    df['base_model'] = df['regime'].apply(lambda x: x.rsplit('-loc-', 1)[0])
    df['train_set'] = df['regime'].apply(lambda x: x.split('-')[-1])
    df['display_name'] = df['base_model'].apply(lambda x: model_mapping.get(x))
    
    test_group_mapping = {
        's1': 'Solo',
        's2': 'Informed',
        's21': 'ToM-Simple',
        's3': 'ToM-Complex'
        }
    df['test_group'] = df['test_group'].map(lambda x: test_group_mapping.get(x, x))
    
    train_sets_to_compare = ['s2', 's21']
    train_set_labels = ['Informed', 'ToM-Simple']
    
    regime_weights = {
        's1': 0.5,  
        's2': 0.5,  # For s2 training
        's21': 0.1  # For s21 training (which uses s1, s2, s21)
    }
    
    training_test_sets = {
        's2': ['s1', 's2'],  # s2 training used s1 and s2 test sets
        's21': ['s1', 's2', 's21']  # s21 training used s1, s2, and s21 test sets
    }
    
    target_suffixes = ['split', 'shared', 'shared-mv', 'detach', 'detach-mv']
    
    colors = {
        'split': ('#8dd3c7', '#4d9d91'),      # Teal for split
        'shared': ('#80b1d3', '#527a90'),     # Blue for shared 
        'shared-mv': ('#b3cde3', '#7a99b3'),  # Light blue for shared-mv
        'detach': ('#fb8072', '#a8554c'),     # Red for detach
        'detach-mv': ('#fdb462', '#a77841')   # Orange for detach-mv
    }   
    prefixes = ['treat', 'perception', 'belief', 'decision', 'all']
    
    fig, axes = plt.subplots(2, 1, figsize=(8.3,5), sharex=True)
    
    group_width = len(target_suffixes) * 0.8
    group_gap = 0.4
    
    legend_handles = []
    legend_labels = []
    
    for suffix in target_suffixes:
        legend_handles.append(plt.Rectangle((0,0), 1, 1, color=colors[suffix][0]))
        legend_labels.append(suffix + " (train)")
        legend_handles.append(plt.Rectangle((0,0), 1, 1, color=colors[suffix][1]))
        legend_labels.append(suffix + " (test)")
    
    for train_idx, train_set in enumerate(train_sets_to_compare):
        ax = axes[train_idx]
        x_start = 0
        x_ticks = []
        x_labels = []
        
        for prefix_idx, prefix in enumerate(prefixes):
            x_pos = x_start + np.arange(len(target_suffixes)) * 0.8
            bar_width = 0.35
            
            group_center = x_start + (len(target_suffixes) * 0.8) / 2
            x_ticks.append(group_center)
            x_labels.append(prefix)
            
            for i, suffix in enumerate(target_suffixes):
                if suffix == 'shared-mv':
                    model = f"{prefix}-shared-mv"
                elif suffix == 'detach-mv':
                    model = f"{prefix}-detach-mv"
                else:
                    model = f"{prefix}-{suffix}"
                
                if model not in df['display_name'].values:
                    continue
                
                train_accs, train_stds, weights = [], [], []
                for ts in training_test_sets[train_set]:
                    test_group = test_group_mapping.get(ts, ts)
                    mask = ((df['display_name'] == model) & 
                           (df['train_set'] == train_set) &
                           (df['test_group'] == test_group))
                    
                    if sum(mask) > 0:
                        train_accs.append(df.loc[mask, 'accuracy mean'].values[0])
                        train_stds.append(df.loc[mask, 'accuracy std'].values[0])
                        weights.append(regime_weights[ts])
                
                if train_accs:
                    train_mean = np.average(train_accs, weights=weights)
                    train_std = np.average(train_stds, weights=weights)
                    
                    ax.bar(x_pos[i] - bar_width/2, train_mean, bar_width, 
                           yerr=train_std, capsize=4, alpha=0.8,
                           color=colors[suffix][0])
                
                mask = ((df['display_name'] == model) & 
                        (df['train_set'] == train_set) &
                        (df['test_group'] == 'ToM-Complex')) 
                
                if sum(mask) > 0:
                    test_mean = df.loc[mask, 'accuracy mean'].values[0]
                    test_std = df.loc[mask, 'accuracy std'].values[0]
                    
                    # Plot the test performance bar
                    ax.bar(x_pos[i] + bar_width/2, test_mean, bar_width, 
                           yerr=test_std, capsize=4, alpha=0.8,
                           color=colors[suffix][1])
            
            x_start += group_width + group_gap
        
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_ylim(0, 1.05)
        if train_idx > 0:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        ax.set_title(f'{train_set_labels[train_idx]}-Train', fontsize=14)
    
    fig.text(0.5, 0.01, 'Learned Module', ha='center', fontsize=14)
    
    fig.legend(legend_handles, legend_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.0),
               ncol=5, 
               fontsize=12,
               handlelength=1,
               columnspacing=1,
               handletextpad=0.5,
               borderpad=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.12) 
    
    save_path = os.path.join(save_dir, 's2_s21_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches=None)
    plt.close()
    
    print(f"Saved comparison bar graph to: {save_path}")
    return save_path

if __name__ == "__main__":
    table_path = "C:\\Users\\Rufus\\Documents\\github\\standoff\\supervised\\exp_13-L\\c\\key_param\\test_group_accuracy_table.csv"
    save_dir = "C:\\Users\\Rufus\\Documents\\github\\standoff\\supervised\\exp_13-L\\c"
    plot_path = load_and_plot_heatmap(table_path, save_dir)
    create_comparison_bar_graph(table_path, save_dir)
    print(f"Heatmap saved to: {plot_path}")