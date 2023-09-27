import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score
from torch.utils.data import Dataset, DataLoader
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

def train_mlp(activations, other_data):
    # Parameters
    input_size = activations.shape[1]  # Size of act_vector
    output_size = other_data.shape[1]  # Size of other_vector
    hidden_size = 128  # you can adjust this
    learning_rate = 1e-2
    num_epochs = 20
    batch_size = 32

    act_train, act_val, other_train, other_val = train_test_split(
        activations, other_data, test_size=0.2, random_state=42
    )
    act_train_tensor = torch.tensor(act_train, dtype=torch.float32)
    other_train_tensor = torch.tensor(other_train, dtype=torch.float32)
    act_val_tensor = torch.tensor(act_val, dtype=torch.float32)
    other_val_tensor = torch.tensor(other_val, dtype=torch.float32)

    train_dataset = VectorDataset(act_train_tensor, other_train_tensor)
    val_dataset = VectorDataset(act_val_tensor, other_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss and optimizer
    model = MLP(input_size, hidden_size, output_size)
    linear_model = LinearClassifier(input_size, output_size)
    criterion = nn.MSELoss()  # Using Mean Squared Error as loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        linear_model.train()  # Set the model to training mode
        for act_vector, other_vector in train_loader:
            outputs = linear_model(act_vector)
            loss = criterion(outputs, other_vector)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        linear_model.eval()  # Set the model to evaluation mode
        val_losses = []
        with torch.no_grad():
            for act_vector, other_vector in val_loader:
                outputs = linear_model(act_vector)
                val_loss = criterion(outputs, other_vector)
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}')


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


def get_integer_labels_for_data(data, unique_arrays):
    """Map each data point to its corresponding unique label."""
    integer_labels = []

    for item in data:
        # Find the index of the item in unique_arrays using a custom comparison for numpy arrays
        idx = next(i for i, unique_array in enumerate(unique_arrays) if np.array_equal(item, unique_array))
        integer_labels.append(idx)

    return integer_labels

def process_activations(path, epoch_numbers, repetitions):
    for epoch_number in epoch_numbers:
        for repetition in repetitions:
            with open(os.path.join(path, f'activations_{epoch_number}_{repetition}.pkl'), 'rb') as f:
                loaded_activation_data = pickle.load(f)

            correlation_data = {}
            correlation_data2 = {}

            correlation_data['activations_out'] = loaded_activation_data['activations_out'][0]
            correlation_data['activations_hidden_short'] = loaded_activation_data['activations_hidden_short'][0]
            correlation_data['activations_hidden_long'] = loaded_activation_data['activations_hidden_long'][0]

            correlation_data2['inputs'] = loaded_activation_data['inputs'][0]
            correlation_data2['labels'] = loaded_activation_data['labels'][0]
            '''if loaded_activation_data['oracles'] and len(loaded_activation_data['oracles']) > 0 and len(loaded_activation_data['oracles'][0]) > 0:
                correlation_data2['oracles'] = loaded_activation_data['oracles'][0]
                print('found oracles?', loaded_activation_data['oracles'][0])'''

            for key in loaded_activation_data:
                if "act_label_" in key:
                    correlation_data2[key] = loaded_activation_data[key][0]


            activation_keys = ['activations_out', 'activations_hidden_short', 'activations_hidden_long']


            for act_key in activation_keys:
                activations = correlation_data[act_key]
                correlation_results = {}

                tsne_model = TSNE(n_components=2, perplexity=30, n_iter=300)
                tsne_values = tsne_model.fit_transform(activations)


                for other_key, other_data in correlation_data2.items():
                    if other_key != act_key and True:
                        #print(act_key, other_key, len(activations), len(other_data))

                        assert activations.shape[0] == other_data.shape[0]
                        #train_mlp(activations, other_data)

                        unique_vectors = get_unique_arrays(other_data)
                        integer_labels = get_integer_labels_for_data(other_data, unique_vectors)

                        plt.figure(figsize=(10, 6))
                        labels = correlation_data2[other_key]
                        plt.scatter(tsne_values[:, 0], tsne_values[:, 1], c=integer_labels, edgecolors='black', cmap='viridis')
                        plt.title(f't-SNE of {act_key} colored by {other_key}')
                        plt.xlabel('Dimension 1')
                        plt.ylabel('Dimension 2')
                        plt.tight_layout()
                        plt.savefig(os.path.join(path, f"{act_key}-tsne-colored-by-{other_key}.png"))


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