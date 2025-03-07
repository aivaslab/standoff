import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
def load_and_plot_heatmap(table_path, save_dir, param='test_group', label='accuracy'):
    df = pd.read_csv(table_path)
    
    model_order = [
        'rule-based',
        'perception-treat',
        'perception-vision',
        'perception-presence',
        'belief-op',
        'belief-my',
        'belief-split',
        'belief-shared',
        'combiner',
        'decision-op',
        'decision-my',
        'decision-split',
        'decision-shared',
        'self',
        'other',
        'both-shared',
        'belief-comb-dec-split'
        'belief-comb-dec-shared'
        'neural-belief-shared',
        'neural-decision-shared',
        'neural-split',
        'neural-shared',
    ]
    
    model_mapping = {
        'a-hardcoded': 'rule-based',
        'a-mix-n-perception-treat': 'perception-treat',
        'a-mix-n-perception-vision': 'perception-vision',
        'a-mix-n-perception-presence': 'perception-presence',
        'a-mix-r-perception-treat-100': 'perception-treat',
        'a-mix-r-perception-vision-100': 'perception-vision',
        'a-mix-r-perception-presence-100': 'perception-presence',
        'a-mix-n-belief-op': 'belief-op',
        'a-mix-n-belief-my': 'belief-my',
        'a-mix-r-belief-op-100': 'belief-op',
        'a-mix-r-belief-my-100': 'belief-my',
        'a-mix-n-belief-split': 'belief-split',
        'a-mix-n-decision-op': 'decision-op',
        'a-mix-n-decision-my': 'decision-my',
        'a-mix-r-decision-op-100': 'decision-op',
        'a-mix-r-decision-my-100': 'decision-my',
        'a-mix-n-decision-split': 'decision-split',
        'a-mix-n-self': 'self',
        'a-mix-n-other': 'other',
        'a-neural-split': 'neural-split',
        'a-mix-n-belief-shared': 'belief-shared',
        'a-mix-n-decision-shared': 'decision-shared',
        'a-mix-n-both-shared': 'both-shared',
        'a-neural-shared': 'neural-shared',
        'a-neural-belief-shared': 'neural-belief-shared',
        'a-neural-decision-shared': 'neural-decision-shared',
        'a-mix-n-combiner': 'combiner',
        'a-mix-n-belief-comb-decision-split': 'belief-comb-dec-split',
        'a-mix-n-belief-comb-decision-shared': 'belief-comb-dec-shared',
    }
    
    df['base_model'] = df['regime'].apply(lambda x: x.rsplit('-loc-', 1)[0])
    print(df['base_model'])
    df['train_set'] = df['regime'].apply(lambda x: x.split('-')[-1])
    df['display_name'] = df['base_model'].apply(lambda x: model_mapping.get(x.replace("-v50-b5", ""), x.replace("-v50-b5", "")))
    
    def get_model_order(name):
        if name in model_order:
            return model_order.index(name)
        return 999  
    
    df['order'] = df['display_name'].apply(get_model_order)
    
    desired_order = ['s1', 's2', 's21', 's3'] if param == 'test_group' else sorted(df[param].unique())
    train_sets = sorted(df['train_set'].unique())
    
    fig = plt.figure(figsize=(len(desired_order) * 1.5, len(train_sets) * 4))
    gs = GridSpec(len(train_sets), 1, figure=fig, hspace=0.1)
    
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
        
        formatted_df = mean_pivot.copy()
        for col in formatted_df.columns:
            for idx in formatted_df.index:
                if pd.notnull(mean_pivot.loc[idx, col]) and pd.notnull(std_pivot.loc[idx, col]):
                    formatted_df.loc[idx, col] = f"{mean_pivot.loc[idx, col]:.2f} ({std_pivot.loc[idx, col]:.2f})"
        
        ax = fig.add_subplot(gs[i])
        
        sns.heatmap(mean_pivot, annot=formatted_df, 
                   fmt='', cmap='RdBu', linewidths=0.5, linecolor='white', 
                   vmin=0, vmax=1, cbar=False, ax=ax)
        
        ax.set_ylabel(f"{train_set}", fontsize=14)
        if i == len(train_sets) - 1:
            ax.set_xlabel("Test Group", fontsize=12)
        else:
            ax.set_xlabel("")
        #ax.set_ylabel("Model Type", fontsize=12)
    
    plt.tight_layout()
    
    plot_save_path = os.path.join(save_dir, f'{label}_{param}_heatmaps_by_trainset.png')
    print(f'Saving figure to {plot_save_path}')
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return plot_save_path

if __name__ == "__main__":
    table_path = "C:\\Users\\Rufus\\Documents\\github\\standoff\\supervised\\exp_11-L\\c\\key_param\\test_group_accuracy_table.csv"
    save_dir = "C:\\Users\\Rufus\\Documents\\github\\standoff\\supervised\\exp_11-L\\c"
    plot_path = load_and_plot_heatmap(table_path, save_dir)
    print(f"Heatmap saved to: {plot_path}")