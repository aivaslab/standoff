import heapq
import pickle
import os
import numpy as np
from itertools import combinations
import gzip

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from src.utils.activation_models import MLP2


class ComplexModel(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(ComplexModel, self).__init__()
        self.layers = nn.ModuleList()
        cur_size = input_size
        for layer_size in layer_sizes:
            self.layers.append(MLP2(cur_size, 32, layer_size))
            cur_size = layer_size

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


def train_mlp_tree(inputs, target_outputs, epochs=10):
    output_sizes = [x.shape[-1] for x in target_outputs]
    model = ComplexModel(inputs.shape[-1], output_sizes)
    learning_rate = 1e-3

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []

    dataset_size = inputs.shape[0]
    batch_size = 128
    validation_split = 0.2
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    dataset = TensorDataset(inputs, *target_outputs)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        epoch_train_losses = [[] for _ in target_outputs]

        for batch in train_loader:
            inputs, *targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)

            losses = [criterion(output, target) for output, target in zip(outputs, targets)]
            total_loss = sum(losses)
            total_loss.backward()
            optimizer.step()

            for i, loss in enumerate(losses):
                epoch_train_losses[i].append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        epoch_val_losses = [[] for _ in target_outputs]
        with torch.no_grad():
            for batch in val_loader:
                inputs, *targets = batch
                outputs = model(inputs)
                losses = [criterion(output, target) for output, target in zip(outputs, targets)]
                # val_losses.append(sum(losses).item())
                for i, loss in enumerate(losses):
                    epoch_val_losses[i].append(loss.item())

        train_losses.append([sum(losses) / len(losses) for losses in epoch_train_losses])
        val_losses.append([sum(losses) / len(losses) for losses in epoch_val_losses])

    return train_losses, val_losses


def normalize_feature_data(feature_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)
    return scaled_data


def get_feature_combinations(feature_keys, combine=True):
    all_features = [(x,) for x in feature_keys]
    if combine:
        for combo in combinations(feature_keys, 2):
            all_features.append(combo)
    return all_features


def load_data(path, epoch_numbers, repetitions, timesteps=5, skip_imaginary=False):
    created_data = False

    for epoch_number in epoch_numbers:
        for repetition in repetitions:
            with open(os.path.join(path, f'activations_{epoch_number}_{repetition}.pkl'), 'rb') as f:
                loaded_activation_data = pickle.load(f)

                if not created_data:
                    data = {}
                    for size in ['all']:  # "['multi', 'all', 'single', 'last']:
                        data[size] = {}
                        for f_type in ['abstract', 'treat', 'box', 'image']:
                            data[size][f_type] = {}
                    for feature in loaded_activation_data.keys():
                        if len(loaded_activation_data[feature]) == 0:
                            continue
                        concatenated_array = np.concatenate(loaded_activation_data[feature], axis=0)
                        feature_shape = concatenated_array.shape[1]
                        if feature_shape == 0:
                            continue
                        new_key = feature.replace("act_label_", "")
                        concatenated_array = normalize_feature_data(concatenated_array)

                        if new_key in ['pred', 'activations_out', 'activations_hidden_short', 'activations_hidden_long', 'oracles']:
                            continue

                        # Add the "Full" features

                        if feature_shape % 49 == 0:
                            print('image', 'all', new_key, feature_shape)
                            data['all']['image'][new_key] = concatenated_array  # for now, we don't need to reshape at all!
                            '''channels = feature_shape // (49 * timesteps)
                            for c in range(channels):
                                data['all']['image'][f'{new_key}_c{c}'] = concatenated_array.reshape((-1, timesteps, channels, 7, 7))[:, :, c, :, :].reshape((-1, channels * 7 * 7))
                                print(f'{new_key}_c{c}', data['all']['image'][f'{new_key}_c{c}'].shape)'''
                        elif (feature_shape % 25 == 0) or (new_key == 'labels'):
                            print('box', 'all', new_key, feature_shape)
                            data['all']['box'][new_key] = concatenated_array
                        elif new_key in ['exist', 'b-exist']:
                            data['all']['treat'][new_key] = concatenated_array
                            print('treat', 'all', new_key, data['all']['treat'][new_key].shape)
                        else:
                            # some data doesn't have all 5 timesteps
                            if new_key in ['opponents']:
                                data['all']['abstract'][new_key] = np.tile(concatenated_array, (1, 5))

                            elif new_key in ['labels', 'pred']:
                                data['all']['abstract'][new_key] = np.zeros((concatenated_array.shape[0], concatenated_array.shape[1] * 5))
                                data['all']['abstract'][new_key][:, -concatenated_array.shape[1]:] = concatenated_array
                            else:
                                data['all']['abstract'][new_key] = concatenated_array
                            print('abs', 'all', new_key, data['all']['abstract'][new_key].shape)

                        # Convert "all" features to "Partial" features

                    created_data = True
    return data


def print_feature_tree(tree, root, level=0, visited=None):
    indent = "  " * level
    if visited is None:
        visited = set()

    if not isinstance(root, tuple):
        root = (root,)

    if root in visited:
        return

    visited.add(root)

    if root[0] not in tree.keys() or not tree[root[0]]:
        # print(f"{indent}{' & '.join(root)}: ?")
        return

    for k, (predictor, mse, sum) in enumerate(tree[root[0]]):
        predictor_str = ' & '.join(predictor)
        print(f"{indent}  {root[0]}:={predictor_str} (MSE {mse:.3f}, G {sum:.3f})")
        for pred in predictor:
            if (pred,) not in visited:
                print_feature_tree(tree, (pred,), level + 1, visited)
        if k > 5 or mse >= 0.5:  # mse > 0.15 and k < len(tree[root[0]])-1 or
            print(f"{indent}  ... ({len(tree[root[0]]) - k - 1} more children)")
            break

def make_gen_error(path, keys):
    ## MAKE gen-error
    df_inf = pd.read_csv(os.path.join(path, f'f2f-val-special-{True}.csv'))
    df_all = pd.read_csv(os.path.join(path, f'f2f-val-special-{False}.csv'))

    for df in [df_inf, df_all]:
        df[['First Key', 'Second Key']] = pd.DataFrame(df['Combo'].apply(lambda x: extract_keys(x, keys)).tolist(), index=df.index)

    df_merged = pd.merge(df_inf, df_all, on=['First Key', 'Second Key'], suffixes=('_inf', '_all'))
    df_merged['gen_error'] = df_merged['Validation Loss_all'] - df_merged['Validation Loss_inf']
    output_path = os.path.join(path, 'gen_error_results.csv')
    df_merged.to_csv(output_path, index=False)

    x_group = df_merged.groupby("Second Key")['Validation Loss_inf'].agg(['mean', 'std'])
    y_group = df_merged.groupby("Second Key")['Validation Loss_all'].agg(['mean', 'std'])

    for inout, firstsecond in [('Input', 'First'), ('Output', 'Second')]:
        unique_keys = df_merged[f'{firstsecond} Key'].unique()
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_keys)))
        color_dict = dict(zip(unique_keys, colors))

        for size in [0, 1]:
            plt.figure(figsize=(10, 10))
            mean_coords = {}
            for key in unique_keys:
                subset = df_merged[df_merged[f'{firstsecond} Key'] == key]
                mean_x = subset['Validation Loss_inf'].mean()
                mean_y = subset['Validation Loss_all'].mean()
                mean_coords[key] = (mean_x, mean_y)
                plt.scatter(subset['Validation Loss_inf'], subset['Validation Loss_all'], color=color_dict[key], label=key, alpha=0.5)
                if size == 1:
                    for _, row in subset.iterrows():
                        plt.text(row['Validation Loss_inf'], row['Validation Loss_all'], f'{row["First Key"]}, {row["Second Key"]}', fontsize=7, color='black', rotation=45)


            if size == 0:
                for key, (mean_x, mean_y) in mean_coords.items():
                    plt.text(mean_x, mean_y, key, fontsize=11, ha='center', va='center', bbox=dict(facecolor=color_dict[key], alpha=0.2, edgecolor='none'))


            plt.xlabel('Contrastive trained loss')
            plt.ylabel('Direct trained loss')
            plt.title(f'Each F2F {inout} feature generalization')
            plt.ylim((0.85*size, 1))
            plt.xlim((0, 1-0.85*size))
            plt.savefig(os.path.join(path, f'{inout}_generalization_{size}.png'))
            plt.close()



def extract_keys(combo, keys):
    for key1 in keys:
        for key2 in keys:
            if combo == f"{key1}-{key2}":
                return key1, key2
    return None, None

def f2f_best_first(path, epoch_numbers, repetitions, timesteps=5, train_mlp=None):
    # data is structured as follows: size (partial/full), type (abstract, box, image), feature (...)

    split_inf = False
    run_f2f = True
    joint = False
    get_diff = True

    print('loading data')
    data = load_data(path, epoch_numbers, repetitions, timesteps=5)
    regime_data = data['all']['abstract']['i-informedness']
    opponents = data['all']['abstract']['opponents']
    unique_regimes = np.unique(regime_data[:, -2:], axis=0) if split_inf else [None]

    if run_f2f:
        print('running f2f', 'split_inf', split_inf)
        results = []
        min_val_loss_per_combo = {}
        for regime_values in unique_regimes:
            if regime_values is None:
                regime_data_r = None
                opponents_r = None
            else:
                regime_data_r = regime_data
                opponents_r = opponents

            #print(len(regime), len(regime_indices))
            #regime_indices = np.where(np.all(regime_data == regime, axis=1))[0]

            ### Start F2F REDO
            real_data = data['all']
            flat_data = {}
            for key1 in real_data.keys():
                for key2 in real_data[key1].keys():
                    flat_data[key2] = real_data[key1][key2]#[regime_indices]

            print('keys:', flat_data.keys())

            combos = [(x, y) for x in flat_data.keys() for y in flat_data.keys() if x != y and x != 'labels' and y != 'inputs']
            for combo in tqdm(combos):
                best_val_loss, _, _, _, model = train_mlp(flat_data[combo[0]], flat_data[combo[1]], regime_data=regime_data_r, regime=regime_values, opponents_data=opponents_r, num_epochs=25, model_type='mlp2', )
                combo_key = (combo[0], combo[1])
                if combo_key not in min_val_loss_per_combo or best_val_loss < min_val_loss_per_combo[combo_key]:
                    min_val_loss_per_combo[combo_key] = best_val_loss
                    path2 = os.path.join(path, f"f2f-{combo[0]}-{combo[1]}-{split_inf}.pkl.gz")
                    with gzip.open(path2, 'wb') as f:
                        pickle.dump(model.state_dict(), f)

                    #torch.save(model.state_dict(), path )
                #results.append({'Regime': f"{regime_values[0]}-{regime_values[1]}", 'Combo': f'{combo[0]}-{combo[1]}', 'Validation Loss': best_val_loss})
        results = [{'Combo': f'{key[0]}-{key[1]}', 'Validation Loss': val} for key, val in min_val_loss_per_combo.items()]
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(path, f'f2f-val-special-{split_inf}.csv'), index=False)
        print('done')


    regime_values = unique_regimes[0]
    if split_inf:
        regime_indices = np.where((regime_data[:, -2] == regime_values[0]) & (regime_data[:, -1] == regime_values[1]))[0]
    real_data = data['all']
    flat_data = {}
    indexed_flat_data = {}
    for key1 in real_data.keys():
        for key2 in real_data[key1].keys():
            flat_data[key2] = real_data[key1][key2]
            if split_inf:
                indexed_flat_data[key2] = real_data[key1][key2][regime_indices]

    keys = list(flat_data.keys())
    df = pd.read_csv(os.path.join(path, f'f2f-val-special-{split_inf}.csv'))
    if get_diff:
        make_gen_error(path, keys)

    df[['First Key', 'Second Key']] = pd.DataFrame(df['Combo'].apply(lambda x: extract_keys(x, keys)).tolist(), index=df.index)
    filtered_df = df[df['Validation Loss'] < 0.1]
    #sorted_df = filtered_df.sort_values(by=['Second Key', 'Validation Loss'])

    trios = []
    for _, row1 in filtered_df.iterrows():
        matches = filtered_df[filtered_df['First Key'] == row1['Second Key']]
        for _, row2 in matches.iterrows():
            trio = (row1['First Key'], row1['Second Key'], row2['Second Key'])
            loss1 = row1['Validation Loss']
            loss2 = row2['Validation Loss']
            if row1['Second Key'] == 'inputs' or row2['Second Key'] == 'inputs' or row1['First Key'] == 'labels' or row2['First Key'] == 'labels':
                continue
            if row1['First Key'] == row2['Second Key']:
                continue
            if loss1 + loss2 < 0.2:
                trios.append((trio, loss1, loss2))
    print(len(trios))

    model_cache = {}
    results_df = pd.DataFrame(columns=['f1', 'f2', 'f3', 'XY_Error', 'YZ_Error', 'XZ_Error', 'Chained_Error', 'XZDifference', 'XYDifference', 'YZDifference', 'train'])
    for trio in tqdm(trios):
        first_model_name = trio[0][0]+'-'+trio[0][1]
        second_model_name = trio[0][1]+'-'+trio[0][2]
        chained_model_name = trio[0][0]+'-'+trio[0][2]
        input_data_eval, input_train = flat_data[trio[0][0]], indexed_flat_data[trio[0][0]]
        middle_data_eval, middle_train = flat_data[trio[0][1]], indexed_flat_data[trio[0][1]]
        output_data_eval, output_train = flat_data[trio[0][2]], indexed_flat_data[trio[0][2]]

        if chained_model_name not in df['Combo'].values:
            continue

        xy_error = df.loc[df['Combo'] == first_model_name, 'Validation Loss'].values[0]
        yz_error = df.loc[df['Combo'] == second_model_name, 'Validation Loss'].values[0]
        xz_error = df.loc[df['Combo'] == chained_model_name, 'Validation Loss'].values[0]

        size1 = input_train.shape[1]
        size2 = middle_train.shape[1]
        size3 = output_train.shape[1]

        if (size1, size2) not in model_cache:
            model1 = MLP2(size1, 32, size2)
            model_cache[(size1, size2)] = model1
        else:
            model1 = model_cache[(size1, size2)]
        if (size2, size3) not in model_cache:
            model2 = MLP2(size2, 32, size3)
            model_cache[(size2, size3)] = model2
        else:
            model2 = model_cache[(size2, size3)]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor_train = torch.tensor(input_train, dtype=torch.float32, device=device)
        input_tensor_eval = torch.tensor(input_data_eval, dtype=torch.float32, device=device)
        output_tensor_train = torch.tensor(output_train, dtype=torch.float32, device=device)
        output_tensor_eval = torch.tensor(output_data_eval, dtype=torch.float32, device=device)
        intermediate_tensor_train = torch.tensor(middle_train, dtype=torch.float32, device=device)
        intermediate_tensor_eval = torch.tensor(middle_data_eval, dtype=torch.float32, device=device)

        if not joint:
            path1 = os.path.join(path, f'f2f-{first_model_name}-{split_inf}.pkl.gz')
            path2 = os.path.join(path, f'f2f-{second_model_name}-{split_inf}.pkl.gz')
            with gzip.open(path1, 'rb') as f:
                state_dict = pickle.load(f)
                state_dict = {k: v.to(device) for k, v in state_dict.items()}
                model1.load_state_dict(state_dict)
                model1.eval()
                model1.to(device)
            with gzip.open(path2, 'rb') as f:
                state_dict = pickle.load(f)
                state_dict = {k: v.to(device) for k, v in state_dict.items()}
                model2.load_state_dict(state_dict)
                model2.eval()
                model2.to(device)
        else:
            model1.to(device)
            model2.to(device)
            optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-3)

            criterion = nn.MSELoss()

            dataset_train = TensorDataset(input_tensor_train, intermediate_tensor_train, output_tensor_train)
            loader = DataLoader(dataset_train, batch_size=128, shuffle=True)
            for epoch in range(25):
                total_loss2 = 0
                for inputs, intermediate_targets, targets in loader:
                    inputs, intermediate_targets, targets = inputs.to(device), intermediate_targets.to(device), targets.to(device)
                    optimizer.zero_grad()
                    first_model_outputs = model1(inputs)
                    loss1 = criterion(first_model_outputs, intermediate_targets)
                    second_model_outputs = model2(first_model_outputs)
                    loss2 = criterion(second_model_outputs, targets)
                    total_loss = loss1 + loss2
                    total_loss2 += total_loss.cpu().detach().numpy()
                    total_loss.backward()
                    optimizer.step()
                #print(total_loss2/len(loader))
        with torch.no_grad():
            first_model_output = model1(input_tensor_eval)
            second_model_output = model2(first_model_output)
            mse = torch.mean((second_model_output - output_tensor_eval) ** 2).cpu().numpy()
            mse2 = torch.mean((first_model_output - intermediate_tensor_eval) ** 2).cpu().numpy()
            train_loss = total_loss2/len(loader)
        egg = {
            'f1': trio[0][0],
            'f2': trio[0][1],
            'f3': trio[0][2],
            'XY_Error': xy_error,
            'YZ_Error': yz_error,
            'XZ_Error': xz_error,
            'Chained_Error': mse,
            'XZDifference': mse-xz_error,
            'XYDifference': mse2-xy_error,
            'YZDifference': mse-yz_error,
            'train': train_loss,
        }
        print(egg)
        results_df = results_df.append(egg, ignore_index=True)
    results_df.to_csv(os.path.join(path, f'chain_results_{split_inf}{"" if not joint else "-j"}.csv'), index=False)


    top_10_losses = sorted_df.groupby('Second Key').head(10)
    #print(top_10_losses)
    model_files = [f for f in os.listdir(path) if f.endswith('.model')]
    #print(model_files)


    ### START SPECIAL TRAIN
    if False:
        real_data = data['all']

        inputs = torch.from_numpy(real_data['image']['inputs']).float()
        outputs = [
            torch.from_numpy(real_data['box']['loc']).float(),
            torch.from_numpy(real_data['box']['b-loc']).float(),
            torch.from_numpy(real_data['box']['labels']).float(),
        ]
        train_losses, val_losses = train_mlp_tree(inputs, outputs, epochs=3)

        # train_losses_by_output = list(zip(*train_losses))
        # val_losses_by_output = list(zip(*val_losses))

        # train_losses_by_output = [[loss.item() for loss in losses] for losses in train_losses_by_output]
        # val_losses_by_output = [[loss.item() for loss in losses] for losses in val_losses_by_output]
        labels = ['loc', 'b-loc', 'labels']

        plt.figure(figsize=(10, 6))
        for i, (t, v) in enumerate(zip(train_losses, val_losses)):
            plt.plot(t, label=labels[i] + ' train')
            plt.plot(v, label=labels[i] + ' val')
        # total_losses = [sum(losses) for losses in zip(*train_losses_by_output)]
        # plt.plot(total_losses, label='Total Loss', color='black', linewidth=2, linestyle='--')
        plt.ylim([0, 3.2])

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Output across Epochs')
        plt.legend()
        plt.show()

        exit()

    ### END SPECIAL TRAIN

    # push the labels onto the graph
    first_item = ('all', 'box', ('labels',))
    heap = [(0, first_item, [])]

    goal = 'inputs'
    model_type = 'mlp2'
    visited = set()

    feature_tree = {}
    done_dict = {}  # stores mse for both input and output

    # val_loss_matrix_f2f = pd.read_csv(os.path.join(path, f"f2f_loss_matrix_mlp2.csv"), index_col=0, header=0)
    # print(val_loss_matrix_f2f)

    heuristics = {}  # stores MSE(inputs, key)

    while heap:
        current_mse, current_node, path = heapq.heappop(heap)

        if current_node in visited or goal in current_node[-1]:
            continue

        visited.add(current_node)
        current_size, current_type, current_feature = current_node

        if goal in current_node:
            print("Goal reached! Path:", path, "with MSE:", current_mse)
            continue

        same_size_data = data[current_size]
        # we don't check features in different size datas?
        for current_feature_indy in current_feature:
            if current_feature_indy in visited:
                continue
            visited.add(current_feature_indy)
            output_data = same_size_data[current_type][current_feature_indy]
            for data_type in same_size_data.keys():
                typed_data = same_size_data[data_type]
                # make this only do f2f
                for feature in get_feature_combinations(typed_data.keys(), combine=True):
                    # print(feature)

                    next_feature_typed = (current_size, data_type, feature)
                    if next_feature_typed in visited:
                        continue
                    continuing = False
                    for f in feature:
                        if (current_size, data_type, (f,)) in visited or f in current_feature:
                            continuing = True
                            break
                    if continuing:
                        continue
                    feature_data = typed_data[feature[0]] if len(feature) == 1 else np.concatenate([typed_data[feature[0]], typed_data[feature[1]]], axis=-1)
                    # for mlp we can just concatenate the vectors
                    key = str(feature) + str(current_feature_indy)
                    if key in done_dict.keys():
                        print('skipping', current_feature_indy, 'from', feature)
                        best_val_loss = done_dict[key]
                    elif False:
                        best_val_loss = 100
                        if current_feature_indy in val_loss_matrix_f2f.columns:  # rows are inputs, and current_feature_indy is our output
                            if feature[0] in val_loss_matrix_f2f.index:
                                best_val_loss = val_loss_matrix_f2f.at[feature[0], current_feature_indy]
                                done_dict[key] = best_val_loss
                    else:
                        best_val_loss, _, _, _, _ = train_mlp(feature_data, output_data, regime_data=None, regime=None, opponents_data=None, num_epochs=5, model_type=model_type, )
                        print('predicting', current_feature_indy, feature, best_val_loss)
                        done_dict[key] = best_val_loss
                    # print(next_feature_typed, current_node, best_val_loss)

                    predictors = feature_tree.get(current_feature_indy, [])
                    predictors.append((feature, best_val_loss, best_val_loss + current_mse + 0.01 * (len(feature) > 1)))
                    feature_tree[current_feature_indy] = sorted(predictors, key=lambda x: x[2])
                    # print(f'Updated feature_tree for {current_feature_indy}:', feature_tree[current_feature_indy])
                    # print(feature_tree[current_feature_indy])

                    if best_val_loss < 0.95 or True:
                        next_feature_typed_indy = (current_size, data_type, feature)
                        heapq.heappush(heap, (best_val_loss + current_mse + 0.01 * (len(feature) > 1), next_feature_typed_indy, path + [feature]))
                        # print(heap)
        print(f'{model_type} tree from {first_item}:')
        print_feature_tree(feature_tree, ('labels',))
    print('finished')
    exit()
