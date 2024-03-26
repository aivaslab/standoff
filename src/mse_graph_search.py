import heapq
import pickle
import os
import numpy as np
from itertools import combinations

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
                    for size in ['partial', 'full']:
                        data[size] = {}
                        for f_type in ['abstract', 'box', 'image']:
                            data[size][f_type] = {}
                    for feature in loaded_activation_data.keys():
                        for size in ['partial', 'full']:
                            concatenated_array = np.concatenate(loaded_activation_data[feature], axis=0)
                            feature_shape = concatenated_array.shape[1]
                            new_key = feature.replace("act_label_", "")

                            if size == 'partial':
                                # temporary because I have not yet coded partial slices
                                continue

                            if new_key in ['pred', 'activations_out', 'activations_hidden_short', 'activations_hidden_long', 'oracles']:
                                continue

                            if (feature_shape % 49 == 0 and feature_shape > 0):
                                print('image', size, new_key, feature_shape)
                                data[size]['image'][new_key] = concatenated_array # for now, we don't need to reshape at all!
                            elif (feature_shape % 25 == 0) or (new_key == 'labels' and (size == 'full')):
                                print('box', size, new_key, feature_shape)
                                data[size]['box'][new_key] = concatenated_array
                            elif (feature_shape == 0):
                                print('abs', size, new_key, feature_shape)
                                data[size]['abstract'][new_key] = concatenated_array

                    created_data = True
    return data


def print_feature_tree(tree, root, level=0, visited=None):

    #print('tree:', tree)
    indent = "  " * level  # Two spaces per level of indentation
    if visited is None:
        visited = set()  # Initialize the set of visited nodes

    # Ensure root is a tuple for consistency with your tree keys
    if not isinstance(root, tuple):
        root = (root,)

    if root in visited:
        #print(f"{indent}{' & '.join(root)}: [Already Visited]")
        return

    visited.add(root)  # Add the current node to the set of visited nodes

    # Check if the root has predictors
    if root[0] not in tree.keys() or not tree[root[0]]:
        #print(f"{indent}{' & '.join(root)}: ?")
        return

    # Print the current root feature
    for predictor, mse in tree[root[0]]:
        # Ensure we're handling predictor tuples correctly
        predictor_str = ' & '.join(predictor)  # Convert all predictors to string format for printing
        print(f"{indent}  {predictor_str} (MSE {mse}):")
        # Recursive call with tuples
        if len(predictor) == 1:
            print_feature_tree(tree, predictor, level + 1, visited.copy())

def f2f_best_first(path, epoch_numbers, repetitions, timesteps=5, train_mlp=None):
    # data is structured as follows: size (partial/full), type (abstract, box, image), feature (...)

    print('loading data')
    data = load_data(path, epoch_numbers, repetitions, timesteps=5)

    # push the labels onto the graph
    first_item = ('full', 'box', ('labels',))
    heap = [(0, first_item, [])]

    goal = 'inputs'
    visited = set()

    feature_tree = {}

    while heap:
        current_mse, current_node, path = heapq.heappop(heap)

        if current_node in visited:
            continue

        visited.add(current_node)
        current_size, current_type, current_feature = current_node

        if current_node == goal:
            print("Goal reached! Path:", path, "with MSE:", current_mse)

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
                    #print(feature)

                    next_feature_typed = (current_size, data_type, feature)
                    if next_feature_typed in visited:
                        continue
                    continuing = False
                    for f in feature:
                        if (current_size, data_type, (f, )) in visited or f in current_feature:
                            continuing = True
                            break
                    if continuing:
                        continue
                    feature_data = typed_data[feature[0]] if len(feature) == 1 else np.concatenate([typed_data[feature[0]], typed_data[feature[1]]], axis=-1)
                    # for mlp we can just concatenate the vectors
                    best_val_loss, train_losses, val_losses, last_epoch_val_losses = \
                        train_mlp(feature_data, output_data, regime_data=None, regime=None, opponents_data=None, num_epochs=1, model_type='mlp2', )
                    #print(next_feature_typed, current_node, best_val_loss)

                    print('predicting', current_feature_indy, feature)

                    predictors = feature_tree.get(current_feature_indy, [])
                    predictors.append((feature, best_val_loss))
                    feature_tree[current_feature_indy] = sorted(predictors, key=lambda x: x[1])
                    #print(f'Updated feature_tree for {current_feature_indy}:', feature_tree[current_feature_indy])
                    #print(feature_tree[current_feature_indy])

                    print_feature_tree(feature_tree, ('labels',))
                    if best_val_loss < 0.35:
                        next_feature_typed_indy = (current_size, data_type, feature)
                        heapq.heappush(heap, (best_val_loss + current_mse, next_feature_typed_indy, path + [feature]))
                        print(heap)

