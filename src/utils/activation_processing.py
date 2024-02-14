import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats
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

def train_mlp(activations, other_data, patience=25, num_prints=5):
    # Parameters
    input_size = activations.shape[1]
    output_size = other_data.shape[1]
    hidden_size = 32
    learning_rate = 1e-3
    num_epochs = 50
    batch_size = 64

    print('size:', activations.shape, other_data.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #print('example:', activations[0], other_data[0])

    act_train, act_val, other_train, other_val = train_test_split(
        activations, other_data, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.tensor(act_train, dtype=torch.float32).to(device), torch.tensor(other_train, dtype=torch.float32).to(device))
    val_dataset = TensorDataset(torch.tensor(act_val, dtype=torch.float32).to(device), torch.tensor(other_val, dtype=torch.float32).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_size, hidden_size, output_size).to(device)
    #linear_model = LinearClassifier(input_size, output_size)
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
        if (epoch + 1) % (num_epochs // num_prints) == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}' )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
    return 1 - best_val_loss, train_losses, val_losses, last_epoch_val_losses


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
    for other_key, other_data in correlation_data2.items():
        if other_key != act_key:
            print(other_key, len(activations), len(other_data))

            assert activations.shape[0] == other_data.shape[0]
            val_loss, train_losses, val_losses, val_losses_indy = train_mlp(activations, other_data)
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim([0, 1.05])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"{other_key}-mlp-loss.png"))
            all_individual_val_losses[other_key] = val_losses_indy
            all_val_losses[other_key] = val_losses

    plt.figure(figsize=(12, 8))
    for other_key, val_losses in all_val_losses.items():
        plt.plot(val_losses, label=f'Val Loss ({other_key})')
    plt.ylim([0, 1.05])
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
    mean_thresholds = {key: np.mean(val_losses_indy) for key, val_losses_indy in all_individual_val_losses.items()}
    binary_correctness = {key: np.array(val_losses_indy) < 0.1 for key, val_losses_indy in
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
    plt.ylabel('Feature A (given condition correct, i.e. <mean MSE)')
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix_n, annot=True, fmt=".2f", cmap="coolwarm_r",
                xticklabels=models, yticklabels=models, vmin=0, vmax=1)
    plt.title('Probability B is Correct, Given A is Incorrect')
    plt.xlabel('Feature B')
    plt.ylabel('Feature A (given condition incorrect, i.e. >=mean MSE)')
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

            keys_to_process = ['activations_out']#, 'activations_hidden_short', 'activations_hidden_long']
            for key in keys_to_process:
                if key in loaded_activation_data:
                    concatenated_array = np.concatenate(loaded_activation_data[key], axis=0)
                    correlation_data[key] = concatenated_array
                    print(f'{key} shape:', concatenated_array.shape)

            keys_to_process2 = ['inputs', 'labels']
            for key in keys_to_process2:
                if key in loaded_activation_data:
                    concatenated_array = np.concatenate(loaded_activation_data[key], axis=0)
                    correlation_data2[key] = concatenated_array
                    print(f'{key} shape:', concatenated_array.shape)

            for key in loaded_activation_data:
                if "act_label_" in key:
                    arrays = []
                    hist_arrays = []
                    for index in range(len(loaded_activation_data[key])):
                        data_array = np.array(loaded_activation_data[key][index])
                        data_array = data_array.astype(int)
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
                    correlation_data2[key] = np.concatenate(arrays, axis=0)
                    if key != "act_label_opponents" and key != "act_label_informedness":
                        correlation_data2[key + "_h"] = np.concatenate(hist_arrays, axis=0)
                    print('cor2shape', key, correlation_data2[key].shape)


            informedness_datapoints = correlation_data2["act_label_vision"]
            random_indices = np.random.choice(len(informedness_datapoints), size=10, replace=False)
            print("vision datapoints:", informedness_datapoints[random_indices])
            activation_keys = ['activations_out']#, 'activations_hidden_short', 'activations_hidden_long']


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