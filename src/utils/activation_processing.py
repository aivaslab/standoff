import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt


def compute_scalar_vector_correlation(activation_scalars, other_data_vectors):
    """
    Calculate the correlation between each scalar from activation_scalars
    and each vector from other_data_vectors.

    Args:
    - activation_scalars (ndarray): Array of shape (batch_size, )
    - other_data_vectors (ndarray): Array of shape (batch_size, vector_size)

    Returns:
    - correlations (ndarray): Array of shape (batch_size, vector_size)
    """
    act_size, vector_size = activation_scalars.shape[1], other_data_vectors.shape[1]
    correlations = np.zeros((act_size, vector_size))

    for i in range(act_size):
        scalar = activation_scalars[:, i]
        for j in range(vector_size):
            vec = other_data_vectors[:, j]
            correlation_matrix = np.corrcoef(scalar, vec)
            correlations[i, j] = correlation_matrix[0, 1]

    return correlations

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

            correlation_results = {}
            activation_keys = ['activations_out', 'activations_hidden_short', 'activations_hidden_long']


            for act_key in activation_keys:
                activations = correlation_data[act_key]

                for other_key, other_data in correlation_data2.items():
                    if other_key != act_key and True:
                        print(act_key, other_key, len(activations), len(other_data))

                        assert activations.shape[0] == other_data.shape[0]


                        result_key = f"{act_key}_vs_{other_key}"

                        correlation_results[result_key] = compute_scalar_vector_correlation(activations, other_data)

                        # Plotting the heatmap for each matrix
                        plt.figure(figsize=(10, 10))
                        sns.heatmap(correlation_results[result_key], cmap='coolwarm', center=0)
                        plt.xlabel(f"{other_key} Environment Features")
                        plt.ylabel(f"{act_key} Neuron Activations")
                        plt.title(result_key)
                        plt.show()
