import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats
import tqdm
from scipy.stats import pearsonr, entropy
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split

class VectorDataset(Dataset):
    def __init__(self, act_vectors, other_vectors):
        self.act_vectors = act_vectors
        self.other_vectors = other_vectors

    def __len__(self):
        return len(self.act_vectors)

    def __getitem__(self, idx):
        return self.act_vectors[idx], self.other_vectors[idx]

class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_mlp(activations, other_data, patience=5, num_prints=5, num_epochs=25, model_type="linear"):
    # Parameters
    input_size = activations.shape[-1]
    output_size = other_data.shape[1]
    hidden_size = 32
    learning_rate = 1e-3
    batch_size = 128

    print('size:', activations.shape, other_data.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print('example:', activations[0], other_data[0])

    act_train, act_val, other_train, other_val = train_test_split(
        activations, other_data, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.tensor(act_train, dtype=torch.float32).to(device), torch.tensor(other_train, dtype=torch.float32).to(device))
    val_dataset = TensorDataset(torch.tensor(act_val, dtype=torch.float32).to(device), torch.tensor(other_val, dtype=torch.float32).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model_type == "mlp1":
        model = MLP(input_size, hidden_size, output_size).to(device)
    elif model_type == "linear":
        model = LinearClassifier(input_size, output_size).to(device)
    elif model_type == "lstm":
        model = LSTMClassifier(input_size, hidden_size, output_size).to(device)
    elif model_type == "mlp2":
        model = MLP2(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    slowcriterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    last_epoch_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        for act_vector, other_vector in train_loader:
            outputs = model(act_vector)
            loss = criterion(outputs, other_vector)
            epoch_train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            if epoch == num_epochs - 1:
                for act_vector, other_vector in val_loader:
                    outputs = model(act_vector)
                    val_loss = criterion(outputs, other_vector)
                    val_loss_indy = slowcriterion(outputs, other_vector).mean(dim=1)
                    epoch_val_losses.append(val_loss.item())
                    last_epoch_val_losses.extend(val_loss_indy.tolist())
            else:
                for act_vector, other_vector in val_loader:
                    outputs = model(act_vector)
                    val_loss = criterion(outputs, other_vector)
                    epoch_val_losses.append(val_loss.item())

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        #if (epoch + 1) % (num_epochs // num_prints) == 0:
        #    print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}' )


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
        if avg_val_loss <= 0.005:
            break
    return best_val_loss, train_losses, val_losses, last_epoch_val_losses


def compute_vector_correlation(activation_scalars, other_data_vectors):
    act_size, vector_size = activation_scalars.shape[1], other_data_vectors.shape[1]
    correlations = np.zeros((act_size, vector_size))

    for i in range(act_size):
        scalar = activation_scalars[:, i]
        for j in range(vector_size):
            vec = other_data_vectors[:, j]
            correlation_matrix = np.corrcoef(scalar, vec)
            correlations[i, j] = correlation_matrix[0, 1]

    return correlations


def get_unique_arrays(data):
    seen = set()
    unique_arrays = []

    for item in data:
        # Convert arrays to tuple for hashable comparison
        item_tuple = tuple(item)

        if item_tuple not in seen:
            seen.add(item_tuple)
            unique_arrays.append(item)

    return unique_arrays


def calculate_entropy(prob):
    if prob == 0 or prob == 1:
        return 0
    return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)


def information_gain(binary_correctness, model_a, model_b):
    correct_a = binary_correctness[model_a]
    correct_b = binary_correctness[model_b]
    prob_a_correct = np.mean(correct_a)
    prob_a_incorrect = 1 - prob_a_correct
    prob_b_correct = np.mean(correct_b)

    entropy_b = calculate_entropy(prob_b_correct)

    correct_b_given_a = correct_b[correct_a]
    incorrect_b_given_not_a = correct_b[~correct_a]
    prob_b_given_a = np.mean(correct_b_given_a) if len(correct_b_given_a) > 0 else 0
    prob_b_given_not_a = np.mean(incorrect_b_given_not_a) if len(incorrect_b_given_not_a) > 0 else 0

    entropy_b_given_a = prob_a_correct*calculate_entropy(prob_b_given_a)
    entropy_b_given_not_a = calculate_entropy(prob_b_given_not_a)

    conditional_entropy_b_given_a = (prob_a_correct * entropy_b_given_a) + (prob_a_incorrect * entropy_b_given_not_a)

    ig_b_given_a = entropy_b - conditional_entropy_b_given_a

    return ig_b_given_a
def calculate_conditional_probability(binary_correctness, model_a, model_b):
    both_correct = np.logical_and(binary_correctness[model_a], binary_correctness[model_b])
    model_a_correct = binary_correctness[model_a]
    model_b_correct = binary_correctness[model_b]
    model_a_incorrect = np.logical_not(model_a_correct)
    b_correct_a_incorrect = np.logical_and(model_b_correct, model_a_incorrect)
    p_b_given_a = np.sum(both_correct) / np.sum(model_a_correct)
    p_b_given_not_a = np.sum(b_correct_a_incorrect) / np.sum(model_a_incorrect)
    return p_b_given_a, p_b_given_not_a

def get_integer_labels_for_data(data, unique_arrays):
    """Map each data point to its corresponding unique label."""
    integer_labels = []

    for item in data:
        # Find the index of the item in unique_arrays using a custom comparison for numpy arrays
        idx = next(i for i, unique_array in enumerate(unique_arrays) if np.array_equal(item, unique_array))
        integer_labels.append(idx)

    return integer_labels

def run_mlp_test(correlation_data2, act_key, activations, path):
    all_individual_val_losses = {}
    all_val_losses = {}
    for other_key, other_data in tqdm.tqdm(correlation_data2.items()):
        if other_key != act_key:
            print(other_key, len(activations), len(other_data))

            assert activations.shape[0] == other_data.shape[0]
            val_loss, train_losses, val_losses, val_losses_indy = train_mlp(activations, other_data)
            '''plt.figure(figsize=(10, 6))
            #plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim([0, 1.05])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"{other_key}-mlp-loss.png"))'''
            all_individual_val_losses[other_key] = val_losses_indy
            all_val_losses[other_key] = val_losses

    plt.figure(figsize=(12, 8))
    for other_key, val_losses in all_val_losses.items():
        plt.plot(val_losses, label=f'Val Loss ({other_key})')
    plt.ylim([0, 0.6])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"all-mlp-loss.png"))

    models = list(all_individual_val_losses.keys())
    num_models = len(models)
    confusion_matrix = np.zeros((num_models, num_models))
    confusion_matrix_n = np.zeros((num_models, num_models))
    ig_matrix = np.zeros((num_models, num_models))
    #mean_thresholds = {key: np.mean(val_losses_indy) for key, val_losses_indy in all_individual_val_losses.items()}
    percentile = 50
    median_thresholds = {key: np.percentile(val_losses_indy, percentile) for key, val_losses_indy in all_individual_val_losses.items()}
    binary_correctness = {key: np.array(val_losses_indy) < median_thresholds[key] for key, val_losses_indy in
                          all_individual_val_losses.items()}
    for i, model_a in enumerate(all_individual_val_losses.keys()):
        for j, model_b in enumerate(list(all_individual_val_losses.keys())):
            p_b_given_a, p_b_given_not_a = calculate_conditional_probability(binary_correctness, model_a, model_b)
            confusion_matrix[i, j] = p_b_given_a
            confusion_matrix_n[i, j] = p_b_given_not_a
            ig_matrix[i, j] = information_gain(binary_correctness, model_a, model_b)
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="coolwarm_r",
                xticklabels=models, yticklabels=models, vmin=0, vmax=1)
    plt.title('Probability B is Correct, Given A is Correct')
    plt.xlabel('Feature B')
    plt.ylabel(f'Feature A (given condition correct, i.e. <{percentile} percentile MSE)')
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix_n, annot=True, fmt=".2f", cmap="coolwarm_r",
                xticklabels=models, yticklabels=models, vmin=0, vmax=1)
    plt.title('Probability B is Correct, Given A is Incorrect')
    plt.xlabel('Feature B')
    plt.ylabel(f'Feature A (given condition incorrect, i.e. >= {percentile} percentile MSE)')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"mlp-confusion.png"))

    plt.figure(figsize=(10, 8))
    sns.heatmap((confusion_matrix - confusion_matrix_n) / (confusion_matrix + confusion_matrix_n), annot=True, fmt=".2f", cmap="coolwarm_r",
                xticklabels=models, yticklabels=models, vmin=-1, vmax=1)
    plt.title('Influence of A on B')
    plt.xlabel('Feature B')
    plt.ylabel('Feature A')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"mlp-influence.png"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(ig_matrix, annot=True, fmt=".2f", cmap="coolwarm_r", xticklabels=models, yticklabels=models, vmin=-1, vmax=1)
    plt.title('Information gain of B given A')
    plt.xlabel('Feature B')
    plt.ylabel('Feature A')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"mlp-infogain.png"))
def process_activations(path, epoch_numbers, repetitions, timesteps=5):
    for epoch_number in epoch_numbers:
        for repetition in repetitions:
            with open(os.path.join(path, f'activations_{epoch_number}_{repetition}.pkl'), 'rb') as f:
                loaded_activation_data = pickle.load(f)

            correlation_data = {}
            correlation_data2 = {}
            correlation_data_lstm = {}

            keys_to_process = ['activations_out', 'activations_hidden_short', 'activations_hidden_long']

            skip_model_dependent = True

            for key in keys_to_process:
                if key in loaded_activation_data:
                    concatenated_array = np.concatenate(loaded_activation_data[key], axis=0)
                    correlation_data[key] = concatenated_array
                    if not skip_model_dependent:
                        correlation_data2[key] = concatenated_array
                    print(f'{key} shape:', concatenated_array.shape)
                    length = concatenated_array.shape[0]

            keys_to_process2 = ['inputs', 'labels', 'pred']
            for key in keys_to_process2:
                if key in loaded_activation_data:
                    if key == "pred":
                        if skip_model_dependent:
                            continue
                        arrays = []
                        for index in range(len(loaded_activation_data[key])):
                            data_array = np.array(loaded_activation_data[key][index]).astype(int)
                            one_hot = np.eye(5)[data_array.reshape(-1)].reshape(data_array.shape[0], -1)
                            arrays.append(one_hot[:, -5:])
                        correlation_data2[key] = np.concatenate(arrays, axis=0)
                    else:
                        concatenated_array = np.concatenate(loaded_activation_data[key], axis=0)
                        correlation_data2[key] = concatenated_array
                    print(f'{key} shape:', concatenated_array.shape)

            for key in loaded_activation_data:
                new_key = key.replace("act_label_", "")
                if "act_label_" in key:
                    arrays = []
                    hist_arrays = []
                    for index in range(len(loaded_activation_data[key])):
                        data_array = np.array(loaded_activation_data[key][index]).astype(int)
                        a_len = data_array.shape[-1]
                        if key in ["act_label_opponents", "act_label_vision"]:
                            one_hot = np.eye(2)[data_array.reshape(-1)].reshape(data_array.shape[0], -1)
                            arrays.append(one_hot[:, -2:])
                            hist_arrays.append(one_hot[:, :])
                        elif key == "act_label_informedness":
                            one_hot = np.eye(3)[data_array.reshape(-1)].reshape(data_array.shape[0], data_array.shape[1] * 3)
                            arrays.append(one_hot[:, -6:])
                            hist_arrays.append(one_hot[:, :])
                        else:
                            arrays.append(data_array[:, -(a_len // timesteps):])
                            hist_arrays.append(data_array[:, :])
                    if key != "act_label_vision":
                        correlation_data2[new_key] = np.concatenate(arrays, axis=0)
                    if key != "act_label_opponents" and key != "act_label_informedness":
                        correlation_data2[new_key + "_h"] = np.concatenate(hist_arrays, axis=0)
                        value = correlation_data2[new_key + "_h"]
                        correlation_data_lstm[new_key + "_h"] = value.reshape(value.shape[0], 5, value.shape[1] // 5)
                        print('corlstmshape', new_key, correlation_data_lstm[new_key + "_h"].shape)
                    print('cor2shape', new_key, correlation_data2[new_key].shape)

            correlation_data2["rand_vec5"] = np.random.randint(2, size=(length, 5))
            #pred_d = correlation_data2["pred"]
            #random_indices = np.random.choice(len(pred_d), size=10, replace=False)
            #print("pred datapoints:", pred_d[random_indices])
            activation_keys = ['activations_out']#, 'activations_hidden_short', 'activations_hidden_long']

            # MLP F2F DATA
            run = True
            #models = ['linear', 'mlp1', 'mlp2', 'lstm']
            models = ['lstm']
            compose = True

            if run:
                for model_type in models:
                    #if compose and model_type == "lstm":
                    #    continue
                    keys = sorted(correlation_data2.keys())
                    size = len(keys)
                    if model_type == "lstm":
                        keys1 = sorted(correlation_data_lstm.keys())
                        size1 = len(keys1)
                    else:
                        size1 = size
                        keys1 = keys

                    val_loss_matrix = np.zeros((size1, size))
                    with tqdm.tqdm(total=size1*size, desc='Processing key pairs') as pbar:
                        for i, key1 in enumerate(keys1):
                            if model_type == "lstm":
                                input_data = correlation_data_lstm[key1]
                                realkeys = keys1
                            else:
                                input_data = correlation_data2[key1]
                                realkeys = keys
                            for j, key2 in enumerate(realkeys):
                                if compose:
                                    if model_type == "lstm":
                                        if key2 not in keys1:
                                            continue
                                        input_data = np.concatenate([correlation_data_lstm[key1], correlation_data_lstm[key2]], axis=2)
                                    else:
                                        input_data = np.concatenate([correlation_data2[key1], correlation_data2[key2]], axis=1)
                                    output_data = correlation_data2["labels"]
                                else:
                                    output_data = correlation_data2[key2]
                                assert input_data.shape[0] == output_data.shape[0]
                                val_loss, train_losses, val_losses, val_losses_indy = train_mlp(input_data, output_data, num_epochs=25, model_type=model_type)
                                val_loss_matrix[i, j] = val_loss
                                print(key1, key2, input_data.shape, val_loss)
                                pbar.update(1)

                    if compose:
                        np.save(os.path.join(path, f"ff2l_loss_matrix_{model_type}.npy"), val_loss_matrix)
                    else:
                        np.save(os.path.join(path, f"f2f_loss_matrix_{model_type}.npy"), val_loss_matrix)

            for model_type in models:

                keys = sorted(correlation_data2.keys())
                size = len(keys)
                if model_type == "lstm":
                    keys1 = sorted(correlation_data_lstm.keys())
                else:
                    keys1 = keys

                val_loss_matrix = np.load(os.path.join(path, f"f2f_loss_matrix_{model_type}.npy"))
                plt.figure(figsize=(12, 8))
                sns.heatmap(val_loss_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                            xticklabels=keys, yticklabels=keys1, vmin=0, vmax=0.25)
                plt.title(f'{model_type} Validation MSE')
                plt.xlabel('Target')
                plt.ylabel('Input')
                plt.tight_layout()
                plt.savefig(os.path.join(path, f"f2f-{model_type}.png"))


                val_loss_matrix = np.load(os.path.join(path, f"ff2l_loss_matrix_{model_type}.npy"))
                order = keys1
                if model_type == "lstm":
                    order = ['b-loc_h', 'loc_h', 'target_h', 'vision_h']
                    order_indices = [keys1.index(label) for label in order]
                    val_loss_matrix = val_loss_matrix[:, :4][np.ix_(order_indices, order_indices)]
                plt.figure(figsize=(12, 8))
                sns.heatmap(val_loss_matrix, annot=True, fmt=".2f", cmap="coolwarm",xticklabels=keys if model_type != "lstm" else order, yticklabels=order, vmin=0, vmax=0.25)
                plt.title(f'{model_type} FF2L Validation MSE')
                plt.xlabel('Input2')
                plt.ylabel('Input1')
                plt.tight_layout()
                plt.savefig(os.path.join(path, f"ff2l-{model_type}.png"))

            matrices = {model: np.load(os.path.join(path, f"f2f_loss_matrix_{model}.npy")) for model in models}
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if j <= i:  # This ensures we only consider each pair once, avoiding redundancy
                        continue

                    keys = sorted(correlation_data2.keys())
                    if model1 == "lstm" or model2 == "lstm":
                        keys1 = sorted(correlation_data_lstm.keys())
                    else:
                        keys1 = keys

                    val_loss_matrix_full = np.full((len(keys), len(keys)), np.nan)
                    shared_keys_indices = [keys.index(key) for key in keys1]
                    all_indices = [keys.index(key) for key in keys]
                    lstm_keys_indices = [keys1.index(key) for key in keys1]

                    for idx1, ldx1 in zip(shared_keys_indices, lstm_keys_indices):
                        for idx2 in all_indices:
                            if keys[idx1] in keys1:
                                model1_index = ldx1 if model1 == "lstm" else idx1
                                model2_index = ldx1 if model2 == "lstm" else idx1
                                val_loss_matrix_full[idx1, idx2] = matrices[model2][model2_index, idx2] - matrices[model1][model1_index, idx2]
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(val_loss_matrix_full, annot=True, fmt=".2f", cmap="coolwarm",
                                xticklabels=keys, yticklabels=keys, vmin=-0.06, vmax=0.06)
                    plt.title(f'F2F Validation MSE ({model2} - {model1})')
                    plt.xlabel('Target')
                    plt.ylabel('Input')
                    plt.tight_layout()
                    plt.savefig(os.path.join(path, f"f2f-difference-{model1}-{model2}.png"))

            # END

            for act_key in activation_keys:
                activations = correlation_data[act_key]
                correlation_results = {}

                #tsne_model = TSNE(n_components=2, perplexity=30, n_iter=300)
                #tsne_values = tsne_model.fit_transform(activations)
                egg = 1

                run_mlp_test(correlation_data2, act_key, activations, path)
                continue

                if egg == 0:

                        if egg == 0:
                            unique_vectors = get_unique_arrays(other_data)
                            integer_labels = get_integer_labels_for_data(other_data, unique_vectors)

                            plt.figure(figsize=(10, 6))
                            labels = correlation_data2[other_key]
                            plt.scatter(tsne_values[:, 0], tsne_values[:, 1], c=integer_labels, edgecolors='black', cmap='tab10')
                            plt.title(f't-SNE of {act_key} colored by {other_key}')
                            plt.xlabel('Dimension 1')
                            plt.ylabel('Dimension 2')
                            plt.tight_layout()
                            plt.savefig(os.path.join(path, f"{act_key}-tsne-colored-by-{other_key}.png"))
                            plt.close()


                            result_key = f"{other_key}"

                            #print('data', other_data[0], len(get_unique_arrays(other_data)))

                            temp_results = compute_vector_correlation(activations, other_data)
                            non_nan_columns = ~np.isnan(temp_results).any(axis=0)
                            correlation_results[result_key] = temp_results[:, non_nan_columns]

                            # Plotting the heatmap for each matrix
                            plt.figure(figsize=(10, 10))
                            ax = sns.heatmap(correlation_results[result_key], cmap='coolwarm', center=0, vmin=-1, vmax=1)
                            plt.xlabel(f"{other_key} Environment Features")
                            plt.ylabel(f"{act_key} Neuron")
                            plt.title(f"{act_key} vs {result_key}")
                            if act_key == 'activations_out':
                                y_ticks = np.arange(0, 128, 4)
                                y_ticklabels = [(str(i % 32)) for i in y_ticks]
                                ax.set_yticks(y_ticks)
                                ax.set_yticklabels(y_ticklabels)

                                max_width = max([t.get_window_extent().width for t in ax.get_yticklabels()])
                                offset = max_width / ax.figure.dpi * 1.5

                                plot_width = ax.get_window_extent().width / ax.figure.dpi
                                text_offset = -plot_width * 0.05

                                for ts in range(0, 4):
                                    plt.axhline(32 * ts, color='white', linestyle='--')
                                    ax.text(text_offset, 32 * ts + 16, f"Timestep {ts + 1}", va='center',
                                            ha='left', color='black', backgroundcolor='white', rotation=90)
                            plt.tight_layout()
                            y_label_pos = ax.yaxis.get_label().get_position()
                            ax.yaxis.get_label().set_position((y_label_pos[0] + offset, y_label_pos[1]))
                            plt.savefig(os.path.join(path, f"{other_key}-{act_key}.png"))
                            plt.close()


                sum_results = []
                for result_key, matrix in correlation_results.items():
                    sums = np.sum(np.abs(matrix), axis=1) / matrix.shape[1]
                    sum_results.append(sums)
                sum_matrix = np.column_stack(sum_results)

                plt.figure(figsize=(15, 10))
                ax = sns.heatmap(sum_matrix, cmap='viridis')
                plt.xlabel("Result Key")
                plt.ylabel("Neuron")
                plt.title("Mean of Absolute Neuron Activation Correlations")
                plt.xticks(ticks = np.arange(0.5, len(correlation_results), 1), labels=correlation_results.keys(), rotation=90)

                if act_key == 'activations_out':
                    y_ticks = np.arange(0, 128, 4)
                    y_ticklabels = [(str(i % 32)) for i in y_ticks]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_ticklabels)

                    max_width = max([t.get_window_extent().width for t in ax.get_yticklabels()])
                    offset = max_width / ax.figure.dpi * 1.5
                    plot_width = ax.get_window_extent().width / ax.figure.dpi
                    text_offset = -plot_width * 0.05

                    for ts in range(0, 4):
                        plt.axhline(32 * ts, color='white', linestyle='--')
                        ax.text(text_offset, 32 * ts + 16, f"Timestep {ts + 1}", va='center',
                                ha='left', color='black', backgroundcolor='white', rotation=90)
                plt.tight_layout()
                y_label_pos = ax.yaxis.get_label().get_position()
                ax.yaxis.get_label().set_position((y_label_pos[0] + offset, y_label_pos[1]))
                plt.savefig(os.path.join(path, f"{act_key}-entropy_heatmap.png"))
                plt.close()