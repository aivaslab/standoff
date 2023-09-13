import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt


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

                for other_key, other_data in correlation_data2.items():
                    if other_key != act_key and True:
                        print(act_key, other_key, len(activations), len(other_data))

                        assert activations.shape[0] == other_data.shape[0]


                        result_key = f"{other_key}"

                        print('data', other_data[0], len(get_unique_arrays(other_data)))

                        temp_results = compute_vector_correlation(activations, other_data)
                        non_nan_columns = ~np.isnan(temp_results).any(axis=0)
                        correlation_results[result_key] = temp_results[:, non_nan_columns]

                        # Plotting the heatmap for each matrix
                        plt.figure(figsize=(10, 10))
                        ax = sns.heatmap(correlation_results[result_key], cmap='coolwarm', center=0)
                        plt.xlabel(f"{other_key} Environment Features")
                        plt.ylabel(f"{act_key} Neuron")
                        plt.title(f"{act_key} vs {result_key}")
                        if act_key == 'activations_out':
                            y_ticks = np.arange(0, 128, 4)
                            y_ticklabels = [(str(i % 32)) for i in y_ticks]
                            ax.set_yticks(y_ticks)
                            ax.set_yticklabels(y_ticklabels)

                            # Annotating with timesteps
                            for ts in range(0, 4):
                                plt.axhline(32 * ts, color='white', linestyle='--')
                                ax.text(-1.5, 32 * ts + 16, f"Timestep {ts + 1}", va='center',
                                        ha='left', color='black', backgroundcolor='white', rotation=90)
                        plt.tight_layout()
                        plt.savefig(os.path.join(path, f"{other_key}-{act_key}.png"))

                std_dev_results = []
                for result_key, matrix in correlation_results.items():
                    row_std_devs = np.std(matrix, axis=1)
                    std_dev_results.append(row_std_devs)
                std_dev_matrix = np.column_stack(std_dev_results)

                plt.figure(figsize=(15, 10))
                ax = sns.heatmap(std_dev_matrix, cmap='viridis')
                plt.xlabel("Result Key")
                plt.ylabel("Neuron")
                plt.title("Variance of Neuron Activation Correlations")
                plt.xticks(ticks = np.arange(0.5, len(correlation_results), 1), labels=correlation_results.keys(), rotation=90)

                if act_key == 'activations_out':
                    y_ticks = np.arange(0, 128, 4)
                    y_ticklabels = [(str(i % 32)) for i in y_ticks]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_ticklabels)

                    # Annotating with timesteps
                    for ts in range(0, 4):
                        plt.axhline(32 * ts, color='white', linestyle='--')
                        ax.text(-1.5, 32 * ts + 16, f"Timestep {ts + 1}", va='center',
                                ha='left', color='black', backgroundcolor='white', rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(path, f"{act_key}-entropy_heatmap.png"))