import glob
import itertools

import sys
import os

import pandas as pd
import heapq

from scipy.stats import sem, t

from src.utils.plotting import save_double_param_figures, save_single_param_figures, save_fixed_double_param_figures, \
    save_fixed_triple_param_figures

sys.path.append(os.getcwd())

from src.objects import *
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import torch
from src.supervised_learning import RNNModel, CustomDataset, DiskLoadingDataset, h5Dataset, h5DatasetSlow, \
    CustomDatasetBig
import traceback
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def decode_event_name(name):
    # Split the name into the main part and the numerical suffix
    main_part, numerical_suffix = name.split('-')

    # Extract individual parameters
    visible_baits = int(main_part[main_part.index('b') + 1:main_part.index('w')])
    swaps = int(main_part[main_part.index('w') + 1:main_part.index('v')])
    visible_swaps = int(main_part[main_part.index('v') + 1])
    first_swap_is_both = 1 if 'f' in main_part else 0
    second_swap_to_first_loc = 1 if 's' in main_part else 0
    delay_2nd_bait = 1 if 'd' in main_part else 0

    # Convert the numerical suffix to binary and pad with zeroes to ensure 4 bits
    binary_suffix = format(int(numerical_suffix), '04b')

    # Extract parameters from the binary suffix
    first_bait_size = int(binary_suffix[3])
    uninformed_bait = int(binary_suffix[2])
    uninformed_swap = int(binary_suffix[1])
    first_swap = int(binary_suffix[0])

    # print(numerical_suffix, first_bait_size)

    # Calculate conditions for special parameters
    swaps_gt_0 = swaps > 0
    first_swap_is_both_false = not first_swap_is_both
    delay_2nd_bait_false = not delay_2nd_bait
    swaps_eq_2 = swaps == 2
    visible_baits_eq_1 = visible_baits == 1
    visible_swaps_eq_1 = visible_swaps == 1

    return pd.Series({
        "visible_baits": visible_baits,
        "swaps": swaps,
        "visible_swaps": visible_swaps,
        "first_swap_is_both": first_swap_is_both if swaps_gt_0 else 'N/A (swaps==0)',
        "second_swap_to_first_loc": second_swap_to_first_loc if swaps_eq_2 and delay_2nd_bait_false else 'N/A (swaps<2 or 2nd bait delayed)',
        "delay_2nd_bait": delay_2nd_bait if swaps_gt_0 and first_swap_is_both_false else 'N/A (swaps=0 or first swap both)',
        "first_bait_size": first_bait_size,
        "uninformed_bait": uninformed_bait if visible_baits_eq_1 else 'N/A (informed baits!=1)',
        "uninformed_swap": uninformed_swap if swaps_eq_2 and visible_swaps_eq_1 else 'N/A (swaps<2 or informed swaps!=1)',
        "first_swap": first_swap if swaps_gt_0 and not delay_2nd_bait and not first_swap_is_both else 'N/A (swaps=0 or 2nd bait delayed or first swap both)'
    })


def train_model(train_sets, target_label, test_sets, load_path='supervised/', save_path='', epochs=100, model_kwargs=None,
                lr=0.001, oracle_labels=[], repetition=0):
    batch_size = 256
    use_cuda = torch.cuda.is_available()
    if oracle_labels[0] == None:
        oracle_labels = []
    data, labels, params, oracles = [], [], [], []
    '''
    data_files = [os.path.join(load_path, name, 'obs.npz') for name in train_sets]
    label_files = [os.path.join(load_path, name, f'label-{target_label}.npz') for name in train_sets]
    param_files = [os.path.join(load_path, name, 'params.npz') for name in train_sets]
    oracle_files = [os.path.join(load_path, name, f'label-{oracle_label}.npz') for name in train_sets for oracle_label
                    in oracle_labels] if oracle_labels else None

    train_dataset = DiskLoadingDataset(data_files, label_files, param_files, oracle_files)

    test_loaders = []
    for val_set_name in test_sets:
        dir = os.path.join(load_path, val_set_name)
        val_data_file = os.path.join(dir, 'obs.npz')
        val_label_file = os.path.join(dir, f'label-{target_label}.npz')
        val_param_file = os.path.join(dir, 'params.npz')

        if oracle_labels:
            val_oracle_files = [os.path.join(dir, f'label-{oracle_label}.npz') for oracle_label in oracle_labels]
        else:
            val_oracle_files = None

        val_dataset = DiskLoadingDataset([val_data_file], [val_label_file], [val_param_file], val_oracle_files)
        test_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))'''

    '''
    for data_name in train_sets:
        dir = os.path.join(load_path, data_name)
        data_files.append(os.path.join(dir, 'obs.h5'))
        labels_files.append(os.path.join(dir, 'label-' + target_label + '.h5'))
        params_files.append(os.path.join(dir, 'params.h5'))

        if oracle_labels:
            oracle_data_files = []
            for oracle_label in oracle_labels:
                oracle_data_files.append(os.path.join(dir, 'label-' + oracle_label + '.h5'))
            oracles_files.append(oracle_data_files)

    train_dataset = h5Dataset(data_files, labels_files, params_files, oracles_files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # can't use more on windows , multiprocessing_context='spawn'

    test_loaders = []
    for val_set_name in test_sets:
        dir = os.path.join(load_path, val_set_name)
        val_data_file = os.path.join(dir, 'obs.h5')
        val_labels_file = os.path.join(dir, 'label-' + target_label + '.h5')
        val_params_file = os.path.join(dir, 'params.h5')

        if oracle_labels:
            oracle_data_files = []
            for oracle_label in oracle_labels:
                oracle_data_files.append(os.path.join(dir, 'label-' + oracle_label + '.h5'))
        else:
            oracle_data_files = None

        val_dataset = h5Dataset([val_data_file], [val_labels_file], [val_params_file], [oracle_data_files])
        test_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))'''


    for data_name in train_sets:
        dir = os.path.join(load_path, data_name)
        data.append(np.load(os.path.join(dir, 'obs.npz'), mmap_mode='r')['arr_0'])
        labels.append(np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0'])
        params.append(np.load(os.path.join(dir, 'params.npz'), mmap_mode='r')['arr_0'])
        if oracle_labels:
            oracle_data = []
            for oracle_label in oracle_labels:
                this_oracle = np.load(os.path.join(dir, 'label-' + oracle_label + '.npz'), mmap_mode='r')['arr_0']
                flattened_oracle = this_oracle.reshape(this_oracle.shape[0], -1)
                oracle_data.append(flattened_oracle)
            combined_oracle_data = np.concatenate(oracle_data, axis=-1)
            oracles.append(combined_oracle_data)


    '''data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    params = np.concatenate(params, axis=0)
    if oracle_labels:
        oracles = np.concatenate(oracles, axis=0)
    else:
        oracles = np.zeros((len(data), 0))'''


    train_dataset = CustomDatasetBig(data, labels, params, oracles)
    del data, labels, params, oracles
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) #can't use more on windows


    test_loaders = []
    for val_set_name in test_sets:
        dir = os.path.join(load_path, val_set_name)
        val_data = np.load(os.path.join(dir, 'obs.npz'), mmap_mode='r')['arr_0']
        val_labels = np.load(os.path.join(dir, 'label-' + target_label + '.npz'), mmap_mode='r')['arr_0']
        val_params = np.load(os.path.join(dir, 'params.npz'), mmap_mode='r')['arr_0']

        if oracle_labels:
            oracle_data = []
            for oracle_label in oracle_labels:
                this_oracle = np.load(os.path.join(dir, 'label-' + oracle_label + '.npz'))['arr_0']
                flattened_oracle = this_oracle.reshape(this_oracle.shape[0], -1)
                oracle_data.append(flattened_oracle)
            combined_oracle_data = np.concatenate(oracle_data, axis=-1)
        else:
            combined_oracle_data = np.zeros((len(val_data), 0))
        val_dataset = CustomDataset(val_data, val_labels, val_params, combined_oracle_data)
        test_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))

    # broken rn
    model_kwargs['oracle_len'] = 0 if len(oracle_labels) == 0 else len(train_dataset.oracles_list[0][0])
    model_kwargs['output_len'] = 5  # np.prod(labels.shape[1:])
    model_kwargs['channels'] = 5  # np.prod(params.shape[2])

    device = torch.device('cuda' if use_cuda else 'cpu')
    model = RNNModel(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
    special_criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #max_val_samples = 500
    num_epochs = epochs
    train_losses = []
    #val_losses = [[] for _ in range(len(test_loaders) + 1)]
    df_paths = []

    # param_losses = pd.DataFrame(columns=['param', 'epoch', 'loss'])
    param_losses_list = []

    t = tqdm.trange(num_epochs*len(train_loader))

    for epoch in range(num_epochs):
        train_loss = 0
        for i, (inputs, target_labels, _, oracle_inputs) in enumerate(train_loader):
            outputs = model(inputs, oracle_inputs)
            loss = criterion(outputs, torch.argmax(target_labels, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #train_loss += loss.item()
            t.update(1)
            #if i > len(train_loader)//5:
            #    break

        #train_losses.append(train_loss / len(train_loader))
        model.eval()
        for idx, _val_loader in enumerate(test_loaders):
            with torch.no_grad():

                for inputs, labels, params, oracle_inputs in _val_loader:
                    outputs = model(inputs, oracle_inputs)
                    losses = special_criterion(outputs, torch.argmax(labels, dim=1))
                    _, predicted = torch.max(outputs, 1)
                    corrects = (predicted == torch.argmax(labels, dim=1)).float()
                    batch_param_losses = [
                        {'param': param, 'epoch': epoch, 'loss': loss.item(), 'accuracy': correct.item()}
                        for param, loss, correct in zip(params, losses, corrects)]
                    param_losses_list.extend(batch_param_losses)

        # save dfs periodically to free up ram:
        os.makedirs(save_path, exist_ok=True)
        df = pd.DataFrame(param_losses_list)
        df = df.join(df['param'].apply(decode_event_name))
        df.replace(r'N/A.*', 'na', regex=True, inplace=True) # todo: construct na dict automatically
        df.to_csv(os.path.join(save_path, f'param_losses_{epoch}_{repetition}.csv'))
        df_paths.append(os.path.join(save_path, f'param_losses_{epoch}_{repetition}.csv'))
        param_losses_list = []
        del df

        model.train()
    # save model
    os.makedirs(save_path, exist_ok=True)
    print('saving model')
    torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{data_name}-model.pt'))

    # pd.DataFrame(param_losses_list).to_csv(path + 'param_losses.csv')
    # df = pd.read_csv('supervised/param_losses.csv')
    # Apply the decoding function to each row

    return df_paths


def calculate_ci(group):
    confidence = 0.95
    group_arr = group.values
    n = group_arr.shape[0]
    m = np.mean(group_arr)
    std_err = sem(group_arr)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    return pd.DataFrame({'lower': [m - h], 'upper': [m + h], 'mean': [m]},
                        columns=['lower', 'upper', 'mean'])


def calculate_statistics(df, last_epoch_df, params, skip_3x=False):
    avg_loss = {}
    variances = {}
    ranges_1 = {}
    ranges_2 = {}
    range_dict = {}
    range_dict3 = {}

    print('calculating statistics...')

    # hist_data = last_epoch_df.groupby('param').mean(numeric_only=True)

    param_pairs = itertools.combinations(params, 2)
    param_triples = itertools.combinations(params, 3)

    numeric_columns = last_epoch_df.select_dtypes(include=[np.number]).columns  # select only numeric columns
    variable_columns = [col for col in numeric_columns if last_epoch_df[col].std() > 0]  # filter out constant columns
    correlations = last_epoch_df[variable_columns].corr()
    target_correlations = last_epoch_df.corr()['accuracy'][variable_columns]
    stats = {
        'param_correlations': correlations,
        'accuracy_correlations': target_correlations,
        'vars': variable_columns
    }


    for param in params:
        ci_df = df.groupby([param, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
        avg_loss[param] = ci_df
        means = last_epoch_df.groupby([param]).mean(numeric_only=True)
        ranges_1[param] = means['accuracy'].max() - means['accuracy'].min()

    for param1, param2 in param_pairs:
        ci_df = df.groupby([param1, param2, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
        avg_loss[(param1, param2)] = ci_df

        means = last_epoch_df.groupby([param1, param2]).mean(numeric_only=True)
        # variances[(param1, param2)] = grouped.var().mean()
        ranges_2[(param1, param2)] = means['accuracy'].max() - means['accuracy'].min()

        for value1 in df[param1].unique():
            subset = last_epoch_df[last_epoch_df[param1] == value1]
            if len(subset[param2].unique()) > 1:
                new_means = subset.groupby(param2)['accuracy'].mean()
                range_dict[(param1, value1, param2)] = new_means.max() - new_means.min()

    if not skip_3x:
        for param1, param2, param3 in param_triples:
            ci_df = df.groupby([param1, param2, param3, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
            avg_loss[(param1, param2, param3)] = ci_df

            for value1 in df[param1].unique():
                for value2 in df[param2].unique():
                    subset = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == param1)]
                    if len(subset[param3].unique()) > 1:
                        new_means = subset.groupby(param3)['accuracy'].mean()
                        range_dict3[(param1, value1, param2, value2, param3)] = new_means.max() - new_means.min()

    return avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats


def save_figures(path, df, avg_loss, ranges, range_dict, range_dict3, params, last_epoch_df, num=10):
    top_pairs = sorted(ranges.items(), key=lambda x: x[1], reverse=True)[:num]
    top_n_ranges = heapq.nlargest(num, range_dict, key=range_dict.get)
    top_n_ranges3 = heapq.nlargest(num, range_dict3, key=range_dict3.get)

    save_double_param_figures(path, top_pairs, df, avg_loss, last_epoch_df)
    save_single_param_figures(path, params, df, avg_loss, last_epoch_df)
    save_fixed_double_param_figures(path, top_n_ranges, df, avg_loss, last_epoch_df)
    save_fixed_triple_param_figures(path, top_n_ranges3, df, avg_loss, last_epoch_df)


def write_metrics_to_file(filepath, df, ranges, params, stats):
    df2 = df[df['epoch'] == df['epoch'].max()]
    with open(filepath, 'w') as f:
        mean_accuracy = df2['accuracy'].mean()
        std_accuracy = df2['accuracy'].std()
        f.write(f"Average accuracy: {mean_accuracy} ({std_accuracy})\n")

        param_stats = df2.groupby('param')['accuracy'].agg(['mean', 'std', 'count'])
        top_10 = param_stats.nlargest(10, 'mean')
        bottom_10 = param_stats.nsmallest(10, 'mean')

        f.write("Top 10 tasks based on average accuracy:\n")
        for param, row in top_10.iterrows():
            f.write(f"Task: {param} (n={row['count']}), Mean accuracy: {row['mean']}, Std deviation: {row['std']}\n")
        f.write("\nBottom 10 tasks based on average accuracy:\n")
        for param, row in bottom_10.iterrows():
            f.write(f"Task: {param} (n={row['count']}), Mean accuracy: {row['mean']}, Std deviation: {row['std']}\n")

        sorted_ranges = sorted(ranges.items(), key=lambda x: x[1], reverse=True)
        f.write("Top 10 parameters based on sensitivity to accuracy across values:\n")
        for i in range(10):
            f.write(f"Parameter: {sorted_ranges[i][0]}, Range: {sorted_ranges[i][1]}\n")

        f.write("\nBottom 10 parameters based on sensitivity to accuracy across values:\n")
        for i in range(10):
            f.write(f"Parameter: {sorted_ranges[-(i + 1)][0]}, Range: {sorted_ranges[-(i + 1)][1]}\n")

        param_value_stats = []
        for param in params:
            param_stats_mean = df2.groupby(param)['accuracy'].mean().reset_index().values
            param_stats_std = df2.groupby(param)['accuracy'].std().reset_index().values
            # Include the parameter's name in the values
            param_stats_with_name = [(param, value, mean, std) for (value, mean), (_, std) in
                                     zip(param_stats_mean, param_stats_std)]
            param_value_stats.extend(param_stats_with_name)

        param_value_stats.sort(key=lambda x: x[2], reverse=True)  # sort by mean accuracy
        top_10_values = param_value_stats[:10]
        bottom_10_values = param_value_stats[-10:]

        f.write("Top 10 parameter values based on average accuracy:\n")
        for param, value, mean_accuracy, std_dev in top_10_values:
            f.write(f"Parameter: {param}, Value: {value}, Mean accuracy: {mean_accuracy}, Std. deviation: {std_dev}\n")

        f.write("\nBottom 10 parameter values based on average accuracy:\n")
        for param, value, mean_accuracy, std_dev in bottom_10_values:
            f.write(f"Parameter: {param}, Value: {value}, Mean accuracy: {mean_accuracy}, Std. deviation: {std_dev}\n")

        f.write("\nCorrelations between parameters and accuracy:\n")
        for param in stats['vars']:
            f.write(f"Correlation between {param} and acc: {stats['accuracy_correlations'][param]}\n")

        f.write("\nCorrelations between parameters:\n")
        for i in range(len(stats['vars'])):
            for j in range(i + 1, len(stats['vars'])):
                f.write(f"Correlation between {params[i]} and {params[j]}: {stats['param_correlations'].iloc[i, j]}\n")


def find_df_paths(directory, file_pattern):
    """Find all CSV files in a directory based on a pattern."""
    return glob.glob(f"{directory}/{file_pattern}")
# train_model('random-2500', 'exist')
def run_supervised_session(save_path, repetitions=1, epochs=5, train_sets=None, eval_name='a1',
                           load_path='supervised', oracle_labels=[], skip_train=True):
    # labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target', 'correctSelection']

    model_kwargs_base = {'hidden_size': [6, 8, 12, 16, 32],
                         'num_layers': [1, 2, 3],
                         'kernels': [4, 8, 16, 24, 32],
                         'kernel_size1': [1, 3, 5],
                         'kernel_size2': [1, 3, 5],
                         'stride1': [1, 2],
                         'pool_kernel_size': [2, 3],
                         'pool_stride': [1, 2],
                         'padding1': [0, 1],
                         'padding2': [0, 1],
                         'use_pool': [True, False],
                         'use_conv2': [True, False],
                         'kernels2': [8, 16, 32, 48],
                         }

    test_names = []

    num_random_tests = 1
    lr = 0.001

    test = 0
    while test < num_random_tests:
        try:
            # model_kwargs = {x: random.choice(model_kwargs_base[x]) for x in model_kwargs_base.keys()}
            model_kwargs = {'hidden_size': 16, 'num_layers': 2, 'output_len': 5, 'pool_kernel_size': 3,
                            'pool_stride': 2, 'channels': 7, 'kernels': 8, 'padding1': 1, 'padding2': 0,
                            'use_pool': False, 'stride1': 1, 'use_conv2': True, 'kernel_size1': 3, 'kernels2': 16,
                            'kernel_size2': 3}
            '''model_kwargs = {'hidden_size': 6, 'num_layers': 1, 'output_len': 5, 'pool_kernel_size': 3,
                            'pool_stride': 2, 'channels': 4, 'kernels': 8, 'padding1': 1, 'padding2': 0,
                            'use_pool': False, 'stride1': 1, 'use_conv2': True, 'kernel_size1': 3, 'kernels2': 8,
                            'kernel_size2': 3}'''

            model_name = "".join([str(x) + "," for x in model_kwargs.values()])

            dfs_paths = []
            last_epoch_df_paths = []
            if not skip_train:
                for repetition in range(repetitions):
                    train_df_paths = train_model(train_sets, 'correctSelection', [eval_name], load_path=load_path,
                                                     save_path=save_path, epochs=epochs, model_kwargs=model_kwargs,
                                                     lr=lr, oracle_labels=oracle_labels, repetition=repetition)
                    dfs_paths.extend(train_df_paths)
                    last_epoch_df_paths.append(train_df_paths[-1])
            else:
                dfs_paths = find_df_paths(save_path, 'param_losses_*_*.csv')
                max_epoch = max(int(path.split('_')[-2]) for path in dfs_paths)
                last_epoch_df_paths = [path for path in dfs_paths if int(path.split('_')[-2]) == max_epoch]

            print('loading dfs...')
            df_list = [pd.read_csv(df_path) for df_path in dfs_paths]
            combined_df = pd.concat(df_list, ignore_index=True)
            last_df_list = [pd.read_csv(df_path) for df_path in last_epoch_df_paths]
            last_epoch_df = pd.concat(last_df_list, ignore_index=True)
            replace_dict = {'1': 1, '0': 0}
            combined_df.replace(replace_dict, inplace=True)
            last_epoch_df.replace(replace_dict, inplace=True)
            params = ['visible_baits', 'swaps', 'visible_swaps', 'first_swap_is_both',
                      'second_swap_to_first_loc', 'delay_2nd_bait', 'first_bait_size',
                      'uninformed_bait', 'uninformed_swap', 'first_swap']


            avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats = calculate_statistics(
                combined_df, last_epoch_df, params, skip_3x=True)

            write_metrics_to_file(os.path.join(save_path, 'metrics.txt'), last_epoch_df, ranges_1, params, stats)

            save_figures(os.path.join(save_path, 'figs'), combined_df, avg_loss, ranges_2, range_dict, range_dict3,
                         params, last_epoch_df, num=12)

            test_names.append(model_name)
            test += 1
        except BaseException as e:
            print(e)
            traceback.print_exc()
    return dfs_paths, last_epoch_df_paths


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return value