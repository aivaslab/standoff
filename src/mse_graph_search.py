import heapq
import pickle
import os
import numpy as np
from itertools import combinations

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

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
                #val_losses.append(sum(losses).item())
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
                    for size in ['all']:#"['multi', 'all', 'single', 'last']:
                        data[size] = {}
                        for f_type in ['abstract', 'treat', 'box', 'image']:
                            data[size][f_type] = {}
                    for feature in loaded_activation_data.keys():
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
        if k > 5 or mse >= 0.5: #mse > 0.15 and k < len(tree[root[0]])-1 or
            print(f"{indent}  ... ({len(tree[root[0]])-k-1} more children)")
            break


def f2f_best_first(path, epoch_numbers, repetitions, timesteps=5, train_mlp=None):
    # data is structured as follows: size (partial/full), type (abstract, box, image), feature (...)



    print('loading data')
    data = load_data(path, epoch_numbers, repetitions, timesteps=5)

    ### START SPECIAL TRAIN
    real_data = data['all']

    inputs = torch.from_numpy(real_data['image']['inputs']).float()
    outputs = [
        torch.from_numpy(real_data['box']['loc']).float(),
        torch.from_numpy(real_data['box']['b-loc']).float(),
        torch.from_numpy(real_data['box']['labels']).float(),
    ]
    train_losses, val_losses = train_mlp_tree(inputs, outputs, epochs=3)

    #train_losses_by_output = list(zip(*train_losses))
    #val_losses_by_output = list(zip(*val_losses))

    #train_losses_by_output = [[loss.item() for loss in losses] for losses in train_losses_by_output]
    #val_losses_by_output = [[loss.item() for loss in losses] for losses in val_losses_by_output]
    labels = ['loc', 'b-loc', 'labels']

    plt.figure(figsize=(10, 6))
    for i, (t, v) in enumerate(zip(train_losses, val_losses)):
        plt.plot(t, label=labels[i] + ' train')
        plt.plot(v, label=labels[i] + ' val')
    #total_losses = [sum(losses) for losses in zip(*train_losses_by_output)]
    #plt.plot(total_losses, label='Total Loss', color='black', linewidth=2, linestyle='--')
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
    done_dict = {} # stores mse for both input and output

    val_loss_matrix_f2f = pd.read_csv(os.path.join(path, f"f2f_loss_matrix_mlp2.csv"), index_col=0, header=0)
    #print(val_loss_matrix_f2f)

    heuristics = {} # stores MSE(inputs, key)

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
                        if current_feature_indy in val_loss_matrix_f2f.columns: # rows are inputs, and current_feature_indy is our output
                            if feature[0] in val_loss_matrix_f2f.index:
                                best_val_loss = val_loss_matrix_f2f.at[feature[0], current_feature_indy]
                                done_dict[key] = best_val_loss
                    else:
                        print('predicting', current_feature_indy, feature)
                        best_val_loss, _, _, _ = train_mlp(feature_data, output_data, regime_data=None, regime=None, opponents_data=None, num_epochs=5, model_type=model_type, )
                        done_dict[key] = best_val_loss
                    # print(next_feature_typed, current_node, best_val_loss)


                    predictors = feature_tree.get(current_feature_indy, [])
                    predictors.append((feature, best_val_loss, best_val_loss + current_mse + 0.01*(len(feature) > 1)))
                    feature_tree[current_feature_indy] = sorted(predictors, key=lambda x: x[2])
                    # print(f'Updated feature_tree for {current_feature_indy}:', feature_tree[current_feature_indy])
                    # print(feature_tree[current_feature_indy])

                    if best_val_loss < 0.95 or True:
                        next_feature_typed_indy = (current_size, data_type, feature)
                        heapq.heappush(heap, (best_val_loss + current_mse + 0.01*(len(feature) > 1), next_feature_typed_indy, path + [feature]))
                        #print(heap)
        print(f'{model_type} tree from {first_item}:')
        print_feature_tree(feature_tree, ('labels',))
    print('finished')
    exit()
