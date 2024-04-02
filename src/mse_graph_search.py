import heapq
import pickle
import os
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler


def normalize_feature_data(feature_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)
    return scaled_data


def get_feature_combinations(feature_keys):
    all_features = [(x,) for x in feature_keys]
    for combo in combinations(feature_keys, 2):
        all_features.append(combo)
    return all_features


def load_data(path, epoch_numbers, repetitions, timesteps=5):
    created_data = False

    for epoch_number in epoch_numbers:
        for repetition in repetitions:
            with open(os.path.join(path, f'activations_{epoch_number}_{repetition}.pkl'), 'rb') as f:
                loaded_activation_data = pickle.load(f)

                if not created_data:
                    data = {}
                    for size in ['multi', 'all', 'single', 'last']:
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
                            channels = feature_shape // (49 * timesteps)
                            for c in range(channels):
                                data['all']['image'][f'{new_key}_c{c}'] = concatenated_array.reshape((-1, timesteps, channels, 7, 7))[:, :, c, :, :].reshape((-1, channels * 7 * 7))
                                print(f'{new_key}_c{c}', data['all']['image'][f'{new_key}_c{c}'].shape)
                        elif (feature_shape % 25 == 0) or (new_key == 'labels'):
                            print('box', 'all', new_key, feature_shape)
                            data['all']['box'][new_key] = concatenated_array
                        elif new_key in ['informedness', 'exist', 'b-exist']:
                            if new_key in ['informedness']:
                                data['all']['treat'][new_key] = np.zeros((concatenated_array.shape[0], concatenated_array.shape[1] * 5))
                                data['all']['treat'][new_key][:, -concatenated_array.shape[1]:] = concatenated_array
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
        if mse > 0.15 and k < len(tree[root[0]])-1 or k > 5:
            print(f"{indent}  ... ({len(tree[root[0]])-k-1} more children)")
            break


def f2f_best_first(path, epoch_numbers, repetitions, timesteps=5, train_mlp=None):
    # data is structured as follows: size (partial/full), type (abstract, box, image), feature (...)

    print('loading data')
    data = load_data(path, epoch_numbers, repetitions, timesteps=5)

    # push the labels onto the graph
    first_item = ('all', 'box', ('labels',))
    heap = [(0, first_item, [])]

    goal = 'inputs'
    visited = set()

    feature_tree = {}
    done_dict = {} # stores mse for both input and output

    while heap:
        current_mse, current_node, path = heapq.heappop(heap)

        if current_node in visited:
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
                for feature in get_feature_combinations(typed_data.keys()):
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
                        print('skipping', current_feature_indy, feature)
                        best_val_loss = done_dict[key]
                    else:
                        print('predicting', current_feature_indy, feature)
                        best_val_loss, _, _, _ = train_mlp(feature_data, output_data, regime_data=None, regime=None, opponents_data=None, num_epochs=5, model_type='mlp2', )
                        done_dict[key] = best_val_loss
                    # print(next_feature_typed, current_node, best_val_loss)


                    predictors = feature_tree.get(current_feature_indy, [])
                    predictors.append((feature, best_val_loss, best_val_loss + current_mse))
                    feature_tree[current_feature_indy] = sorted(predictors, key=lambda x: x[2])
                    # print(f'Updated feature_tree for {current_feature_indy}:', feature_tree[current_feature_indy])
                    # print(feature_tree[current_feature_indy])

                    print_feature_tree(feature_tree, ('labels',))
                    if best_val_loss < 0.35:
                        next_feature_typed_indy = (current_size, data_type, feature)
                        heapq.heappush(heap, (best_val_loss + current_mse, next_feature_typed_indy, path + [feature]))
                        #print(heap)
