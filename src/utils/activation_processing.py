import os
import pickle
import time

import h5py
import numpy as np
import pandas as pd
#import scipy.stats
import tqdm
import umap

#from scipy.stats import pearsonr, entropy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.spatial.distance import pdist, squareform

from torch.optim.lr_scheduler import ExponentialLR

#from sklearn.manifold import TSNE
#from sklearn.metrics import mutual_info_score
#from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split

from src.mse_graph_search import f2f_best_first
from src.utils.activation_models import MLP, LinearClassifier, MLP2d, MLP2c, MLP2bn, MLP3, BasicCNN2m, BasicCNN2, BasicCNN1, MLP2, LSTMClassifier, MLP2ln, TinyAttentionMLP, LSTMClassifierDrop, RNNClassifier
from src.utils.activation_plotting import plot_info_matrices, plot_histogram, plot_scatter, plot_bars


class VectorDataset(Dataset):
    def __init__(self, act_vectors, other_vectors):
        self.act_vectors = act_vectors
        self.other_vectors = other_vectors

    def __len__(self):
        return len(self.act_vectors)

    def __getitem__(self, idx):
        return self.act_vectors[idx], self.other_vectors[idx]


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2) / (torch.var(y_true) + 1e-8)


def get_model_type(model_type, input_size, input_size2, output_size, hidden_size, device, num_epochs):
    l2_reg = False
    if "l2reg" in model_type:
        model_type = model_type.replace("l2reg", "")
        l2_reg = True

    if 'e50' in model_type:
        num_epochs = 50
        model_type = model_type.replace("e50", "")
    elif 'e100' in model_type:
        num_epochs = 100
        model_type = model_type.replace("e100", "")

    if model_type == "mlp1":
        model = MLP(input_size, hidden_size, output_size).to(device)
    elif model_type == "linear":
        model = LinearClassifier(input_size, output_size).to(device)
    elif model_type == "lstm":
        model = LSTMClassifier(input_size, hidden_size, output_size).to(device)
    elif model_type == "mlp2":
        model = MLP2(input_size, hidden_size, output_size).to(device)
    elif model_type == "conv1":
        model = BasicCNN1(input_size, output_size).to(device)
    elif model_type == "conv2":
        model = BasicCNN2(input_size, output_size).to(device)
    elif model_type == "conv2m":
        model = BasicCNN2m(input_size, output_size).to(device)
    elif model_type == "mlp3":
        model = MLP3(input_size, hidden_size, output_size).to(device)
    elif model_type == "mlp2bn":
        model = MLP2bn(input_size, hidden_size, output_size).to(device)
    elif model_type == "mlp2s":
        model = MLP2bn(input_size, 16, output_size).to(device)
    elif model_type == "mlp2c5":
        model = MLP2c(input_size, input_size2, 5, output_size).to(device)
    elif model_type == "mlp2c10":
        model = MLP2c(input_size, input_size2, 10, output_size).to(device)
    elif model_type == "mlp2c16":
        model = MLP2c(input_size, input_size2, 16, output_size).to(device)
    elif model_type == "mlp2d":
        model = MLP2d(input_size, hidden_size, output_size).to(device)
    elif model_type == "mlp2d2":
        model = MLP2d(input_size, hidden_size, output_size, amount=0.2).to(device)
    elif model_type == "mlp2d3":
        model = MLP2d(input_size, hidden_size, output_size, amount=0.3).to(device)
    elif model_type == "mlp2ln":
        model = MLP2ln(input_size, hidden_size, output_size).to(device)
    elif model_type == "attn":
        model = TinyAttentionMLP(input_size, output_size, hidden_size, 1).to(device)
    elif model_type == "lstmd1":
        model = LSTMClassifierDrop(input_size, hidden_size, output_size, 1, 0.1).to(device)
    elif model_type == "lstmd5":
        model = LSTMClassifierDrop(input_size, hidden_size, output_size, 1, 0.5).to(device)
    elif model_type == "rnn":
        model = RNNClassifier(input_size, hidden_size, output_size, 1, 0.0).to(device)
    elif model_type == "rnnd1":
        model = RNNClassifier(input_size, hidden_size, output_size, 1, 0.1).to(device)
    elif model_type == "rnnd5":
        model = RNNClassifier(input_size, hidden_size, output_size, 1, 0.5).to(device)
    return model, l2_reg, num_epochs


def calculate_accuracy(outputs, labels, threshold=0.5):
    binarized_preds = (outputs > threshold).float()
    correct_elements = (binarized_preds == labels)
    correct_datapoints = correct_elements.all(dim=1)
    return correct_datapoints.float()

def train_mlp(inputs, other_data, regime_data, regime, opponents_data, datapoint_ids=None, patience=5, num_batches=10000,
              model_type="linear", input_data2=None):
    # Parameters
    input_size = inputs.shape[-1] if "conv" not in model_type else inputs.shape[1]
    input_size2 = 0
    if input_data2 is not None:
        input_size2 = input_data2.shape[-1] if "conv" not in model_type else inputs.shape[1]
    output_size = other_data.shape[1]
    hidden_size = 32
    learning_rate = 1e-3
    batch_size = 256

    num_epochs = num_batches // batch_size

    all_indices = np.arange(len(inputs))
    train_indices, val_indices = train_test_split(all_indices, test_size=0.10, random_state=42)


    if regime is not None:
        # print(opponents_data.shape, np.mean(opponents_data, axis=0), opponents_data[0], opponents_data[-1], opponents_data[5000], np.unique(opponents_data))
        zero_opponents_indices = np.where(np.all(opponents_data[:, -1:] == -1, axis=1))[0]
        # print(regime_data.shape, regime.shape)
        regime_indices = np.where(np.all(regime_data[:, -2:] == regime, axis=1))[0]
        combined_indices = np.union1d(regime_indices, zero_opponents_indices)
        # print('regime split effect', len(regime_data), len(regime_indices), len(combined_indices))
        regime_train_indices = np.intersect1d(train_indices, combined_indices)
        act_train = inputs[regime_train_indices]
        other_train = other_data[regime_train_indices]
        if input_data2 is not None:
            input2_train = input_data2[regime_train_indices]
    else:
        act_train = inputs[train_indices]
        other_train = other_data[train_indices]
        if input_data2 is not None:
            input2_train = input_data2[train_indices]

    act_val = inputs[val_indices]
    other_val = other_data[val_indices]
    if datapoint_ids is not None:
        individual_losses = []
        individual_accuracies = []
        validation_indices = []
        datapoint_val = datapoint_ids[val_indices]

    if input_data2 is not None:
        input2_val = input_data2[val_indices]

    # print('size:', regime, activations.shape, other_data.shape, len(train_indices), len(act_train), len(act_val))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if input_data2 is not None:
        train_dataset = TensorDataset(torch.tensor(act_train, dtype=torch.float32).to(device),
                                      torch.tensor(input2_train, dtype=torch.float32).to(device),
                                      torch.tensor(other_train, dtype=torch.float32).to(device))
        val_dataset = TensorDataset(torch.tensor(act_val, dtype=torch.float32).to(device),
                                    torch.tensor(input2_val, dtype=torch.float32).to(device),
                                    torch.tensor(other_val, dtype=torch.float32).to(device))
    else:
        train_dataset = TensorDataset(torch.tensor(act_train, dtype=torch.float32).to(device),
                                      torch.tensor(other_train, dtype=torch.float32).to(device))
        if datapoint_ids is None:
            val_dataset = TensorDataset(torch.tensor(act_val, dtype=torch.float32).to(device),
                                        torch.tensor(other_val, dtype=torch.float32).to(device))
        else:
            val_dataset = TensorDataset(torch.tensor(act_val, dtype=torch.float32).to(device),
                                        torch.tensor(datapoint_val, dtype=torch.float32).to(device),
                                        torch.tensor(other_val, dtype=torch.float32).to(device))


    is_windows = True  # os.name == 'nt'

    if is_windows:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model, l2_reg, num_epochs = get_model_type(model_type, input_size, input_size2, output_size, hidden_size, device, num_epochs)

    criterion = nn.MSELoss()
    slowcriterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5 if l2_reg else 0)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    last_epoch_val_losses = []
    val_accuracies = []


    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        for data in train_loader:
            #act_vector, *optional_data = data[0]
            #other_vector = data[-1]
            act_vector, other_vector = data[0], data[1]
            outputs = model(act_vector, *optional_data) if input_data2 is not None else model(act_vector)
            loss = criterion(outputs, other_vector)
            epoch_train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            #print(epoch, num_epochs)
            if epoch == num_epochs - 1:
                for data in val_loader:
                    act_vector, other_vector = data[0], data[2]
                    index_vector = data[1]
                    #act_vector, *optional_data = data[:-1]
                    #other_vector = data[-1]
                    #outputs = model(act_vector, *optional_data) if input_data2 is not None else model(act_vector)
                    outputs = model(act_vector)
                    val_loss = criterion(outputs, other_vector)
                    val_loss_indy = slowcriterion(outputs, other_vector).mean(dim=1)
                    epoch_val_losses.append(val_loss.item())
                    last_epoch_val_losses.extend(val_loss_indy.tolist())
                    val_acc_indy = calculate_accuracy(outputs, other_vector)
                    '''flat_indices = index_vector.cpu().numpy().flatten()
                    flat_losses = val_loss_indy.cpu().numpy().flatten()
                    flat_accs = val_acc_indy.cpu().numpy().flatten()

                    for idx, loss, acc in zip(flat_indices, flat_losses, flat_accs):
                        individual_losses[int(idx)] = float(loss)
                        individual_accuracies[int(idx)] = float(acc)'''

                    individual_losses.append(val_loss_indy)
                    individual_accuracies.append(val_acc_indy)
                    #print('appended')
                    validation_indices.append(index_vector)

                    val_accuracies.append(val_acc_indy.mean().item())
            else:
                for data in val_loader:
                    #act_vector, *optional_data = data[:-1]
                    #other_vector = data[-1]
                    act_vector, other_vector = data[0], data[2]
                    index_vector = data[1]
                    #outputs = model(act_vector, *optional_data) if input_data2 is not None else model(act_vector)
                    outputs = model(act_vector)
                    val_loss = criterion(outputs, other_vector)
                    epoch_val_losses.append(val_loss.item())

                    val_acc = calculate_accuracy(outputs, other_vector)
                    val_accuracies.append(val_acc.mean().item())

        if epoch == num_epochs - 1:
            individual_losses = torch.cat(individual_losses).cpu().numpy()
            individual_accuracies = torch.cat(individual_accuracies).cpu().numpy()
            all_validation_indices = torch.cat(validation_indices).cpu().numpy()
            individual_losses_dict = dict(zip(all_validation_indices.flatten(), individual_losses))
            individual_accuracies_dict = dict(zip(all_validation_indices.flatten(), individual_accuracies))

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        val_accuracies = [acc.item() if torch.is_tensor(acc) else acc for acc in val_accuracies]
        avg_val_acc = sum(val_accuracies) / len(val_accuracies)
        # if (epoch + 1) % (num_epochs // num_prints) == 0:
        #    print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}' )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            '''if epochs_no_improve == patience:
                break'''
        '''if avg_val_loss <= 0.005:
            break'''
    return best_val_loss, train_losses, val_losses, last_epoch_val_losses, avg_val_acc, model, individual_losses_dict, individual_accuracies_dict


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

    entropy_b_given_a = prob_a_correct * calculate_entropy(prob_b_given_a)
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
            val_loss, train_losses, val_losses, val_losses_indy, val_acc, model, _ = train_mlp(activations, other_data)
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
    # mean_thresholds = {key: np.mean(val_losses_indy) for key, val_losses_indy in all_individual_val_losses.items()}
    percentile = 50
    median_thresholds = {key: np.percentile(val_losses_indy, percentile) for key, val_losses_indy in
                         all_individual_val_losses.items()}
    binary_correctness = {key: np.array(val_losses_indy) < median_thresholds[key] for key, val_losses_indy in
                          all_individual_val_losses.items()}
    for i, model_a in enumerate(all_individual_val_losses.keys()):
        for j, model_b in enumerate(list(all_individual_val_losses.keys())):
            p_b_given_a, p_b_given_not_a = calculate_conditional_probability(binary_correctness, model_a, model_b)
            confusion_matrix[i, j] = p_b_given_a
            confusion_matrix_n[i, j] = p_b_given_not_a
            ig_matrix[i, j] = information_gain(binary_correctness, model_a, model_b)
    plot_info_matrices(confusion_matrix, confusion_matrix_n, models, percentile, path, ig_matrix)


def get_keys(model_type, used_cor_inputs, correlation_data2, correlation_data_conv, correlation_data_lstm, correlation_data_lstm_inputs, correlation_data_lstm_outputs, compose,
             use_conv_inputs=False, image_output=False):
    output_keys = sorted(used_cor_inputs.keys())
    input_keys = sorted(used_cor_inputs.keys())
    output_size = len(output_keys)
    input_size = len(output_keys)
    second_input_keys = input_keys
    if not compose:
        cor2keys = sorted(correlation_data2.keys())
        input_size = len(cor2keys)
        if "conv" in model_type or use_conv_inputs:
            input_keys = sorted(correlation_data_conv.keys())
            output_size = len(input_keys)
    if "lstm" in model_type or "rnn" in model_type:
        input_keys = sorted(correlation_data_lstm_inputs.keys())
        return input_keys, input_keys, len(input_keys), len(input_keys), input_keys
    else:
        output_size = input_size
        second_input_keys = input_keys
    if image_output:
        output_keys = sorted(correlation_data_conv.keys())
    else:
        output_keys = sorted(correlation_data2.keys())
    return input_keys, output_keys, output_size, input_size, second_input_keys


def correlation_thing(activation_keys, correlation_data_inputs, correlation_data_outputs, test_keys, path, umapping=False, cca=False, rsa=False, pearson_figs=False, max_samples=-1, mlp_test=True, repetition=-1, epoch=-1, use_inputs=True):
    # final heatmap is feature vs layer, mean neuron abs correlation
    major_cor_results = {}
    cca_results = {}
    rsa_results = {}
    mlp_results = {}
    all_individual_losses = {}
    all_individual_accs = {}
    identity_data = np.asarray(correlation_data_inputs['act_id'])[:max_samples]

    print('all keys', correlation_data_inputs.keys())

    start_time = time.time()

    activation_keys = [x for x in correlation_data_inputs.keys() if "_output" in x or "_state" in x]
    input_keys = [x for x in correlation_data_inputs.keys() if "_input" in x]
    activation_keys.append('scalar')
    num_samples = len(correlation_data_inputs[activation_keys[0]])
    print('keys', activation_keys, num_samples)

    #print(np.mean(correlation_data2['act_label_exist']))

    correlation_data_inputs['all_activations'] = [
        np.concatenate([correlation_data_inputs[key][i] for key in activation_keys if 'input' not in key and 'scalar' not in key])
        for i in range(num_samples)
    ]
    correlation_data_inputs['final_layer_activations'] = [
        np.concatenate([correlation_data_inputs[key][i] for key in activation_keys if "fc" in key])
        for i in range(num_samples)
    ]

    real_act_keys = ['all_activations', 'final_layer_activations', 'input_activations']
    if not use_inputs:
        real_act_keys = [x for x in real_act_keys if 'input' not in x]
    else:
        correlation_data_inputs['input_activations'] = [
            np.concatenate([correlation_data_inputs[key][i] for key in input_keys])
            for i in range(num_samples)
        ]

    #if repetition == 0 and epoch == 0:
    #    real_act_keys.append('scalar')

    csv_path = os.path.join(path, f'ifr_df.csv')
    h5_path = os.path.join(path, f'indy_ifr{repetition}.h5')
    if os.path.exists(csv_path):
        ifr_df = pd.read_csv(csv_path)
    else:
        ifr_df = pd.DataFrame()

    for act_key in real_act_keys:#activation_keys:
        if act_key == 'labels':
            continue
        # each of our activations
        data = correlation_data_inputs[act_key]
        activations = np.asarray(data[:max_samples])
        correlation_results = {}

        if umapping:
            tsne_model = umap.UMAP(n_neighbors=50, min_dist=0.5, metric='euclidean', n_epochs=100)
            tsne_values = tsne_model.fit_transform(activations)
            print('finished umap')

        # this is for model activations
        #run_mlp_test(correlation_data2, act_key, activations, path)
        #continue

        for other_key in tqdm.tqdm(test_keys):
            other_data = np.asarray(correlation_data_outputs[other_key])[:max_samples]
            if other_key == '' or 'oracle' in other_key:
                continue
            if umapping:
                unique_vectors, integer_labels = np.unique(other_data, axis=0, return_inverse=True)
            other_key = other_key.replace('act_', '')

            if umapping:
                plt.figure(figsize=(10, 6))
                plt.scatter(tsne_values[:, 0], tsne_values[:, 1], c=integer_labels, edgecolors='black', cmap='tab10')
                plt.title(f'umap of {act_key} colored by {other_key}')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.tight_layout()
                os.makedirs(os.path.join(path, 'umaps'), exist_ok=True)
                plt.savefig(os.path.join(path, 'umaps', f"{act_key}-umap-colored-by-{other_key}.png"))
                plt.close()

            if mlp_test and other_key != act_key:
                for ifr_rep in [0, 1, 2]:
                    assert activations.shape[0] == other_data.shape[0]
                    val_loss, train_losses, val_losses, val_losses_indy, val_acc, _, val_loss_indy, val_acc_indy = train_mlp(activations, other_data, None, None, None, model_type='mlp1', num_batches=10000, datapoint_ids=identity_data)
                    mlp_results[(act_key, other_key, ifr_rep)] = val_loss
                    all_individual_losses[(act_key, other_key, epoch, repetition, ifr_rep)] = val_loss_indy
                    all_individual_accs[(act_key, other_key, epoch, repetition, ifr_rep)] = val_acc_indy
                    #all_individual_val_losses[other_key] = val_losses_indy
                    #all_val_losses[other_key] = val_losses
                    ifr_df = ifr_df.append({'epoch': epoch, 'act': act_key, 'feature': other_key, 'rep': repetition, 'type': 'mlp', 'loss': val_loss, 'val_acc': val_acc, 'ifr_rep': ifr_rep}, ignore_index=True)
                    print(mlp_results)

            #print('data', other_key, other_data[0], len(get_unique_arrays(other_data)))

            # pearson correlation
            temp_results = compute_vector_correlation(activations, other_data)
            non_nan_columns = ~np.isnan(temp_results).any(axis=0)
            correlation_results[other_key] = temp_results[:, non_nan_columns]

            # cca
            if cca:
                cca = CCA(n_components=1)
                X_c, Y_c = cca.fit_transform(activations, other_data)
                cca_correlation = np.corrcoef(X_c.T, Y_c.T)[0, 1]
                cca_results[(act_key, other_key)] = cca_correlation

            # Compute RSA
            if rsa:
                activation_sim = 1 - squareform(pdist(activations, metric='correlation'))
                feature_sim = 1 - squareform(pdist(other_data, metric='correlation'))
                rsa_correlation = np.corrcoef(activation_sim.flatten(), feature_sim.flatten())[0, 1]
                rsa_results[(act_key, other_key)] = rsa_correlation

            # Plotting the heatmap for each matrix
            if pearson_figs:
                plt.figure(figsize=(10, 10))
                ax = sns.heatmap(correlation_results[other_key], cmap='coolwarm', center=0, vmin=-1, vmax=1)
                plt.xlabel(f"{other_key} components")
                plt.ylabel(f"{act_key} Neuron")
                plt.title(f"{act_key} vs {other_key}")
                plt.tight_layout()
                os.makedirs(os.path.join(path, 'neuron-component'), exist_ok=True)
                plt.savefig(os.path.join(path, 'neuron-component', f"{other_key}-{act_key}.png"))
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
        plt.title("Mean of Absolute Neuron Activation Correlations over Unique Feature Values")
        plt.xticks(ticks=np.arange(0.5, len(correlation_results), 1), labels=correlation_results.keys(), rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{act_key}-entropy_heatmap.png"))
        plt.close()

        mean_neurons = np.mean(sum_matrix, axis=0)
        major_cor_results[act_key] = mean_neurons

    print('saving csv', csv_path)
    ifr_df.to_csv(csv_path, index=False)

    print('saving individual losses h5', h5_path)
    with h5py.File(h5_path, 'w') as f:
        for (act_key, other_key, epoch, repetition, ifr_rep) in all_individual_losses.keys():
            dataset_name = f"{act_key}_{other_key}_{epoch}_{repetition}_{ifr_rep}"

            losses = all_individual_losses[(act_key, other_key, epoch, repetition, ifr_rep)]
            accs = all_individual_accs[(act_key, other_key, epoch, repetition, ifr_rep)]

            indices = np.array(list(losses.keys()))
            values = np.array(list(losses.values()))
            acc_values = np.array(list(accs.values()))
            f.create_dataset(f"{dataset_name}_indices", data=indices)
            f.create_dataset(f"{dataset_name}_values", data=values)
            f.create_dataset(f"{dataset_name}_acc_values", data=acc_values)

    elapsed_time = time.time() - start_time
    time_struct = time.gmtime(elapsed_time)
    print(f"Finished correlation, took: {time.strftime('%H:%M:%S', time_struct)}")

    if train_mlp:
        cca_matrix = np.zeros((len(real_act_keys), len(correlation_results)))
        for i, act_key in enumerate(real_act_keys):
            for j, other_key in enumerate(correlation_results.keys()):
                if (act_key, other_key) in mlp_results:
                    cca_matrix[i, j] = mlp_results[(act_key, other_key)]
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(cca_matrix, cmap='viridis', annot=True, fmt=".3f", vmin=0, vmax=0.25)
        plt.xlabel("Feature")
        plt.ylabel("Layer")
        plt.title("MLP Validation Loss")
        plt.xticks(ticks=np.arange(0.5, len(correlation_results), 1), labels=correlation_results.keys(), rotation=90)
        plt.yticks(ticks=np.arange(0.5, len(real_act_keys), 1), labels=real_act_keys, rotation=0)
        plt.tight_layout()

        path2 = os.path.join(path, 'mlp_results')
        os.makedirs(path2, exist_ok=True)
        plt.savefig(os.path.join(path2, f"layer_mlp_heatmap.png"))
        plt.close()

    # Plotting CCA results
    if cca:
        cca_matrix = np.zeros((len(activation_keys), len(correlation_results)))
        for i, act_key in enumerate(activation_keys):
            for j, other_key in enumerate(correlation_results.keys()):
                if (act_key, other_key) in cca_results:
                    cca_matrix[i, j] = cca_results[(act_key, other_key)]
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(cca_matrix, cmap='viridis')
        plt.xlabel("Feature")
        plt.ylabel("Layer")
        plt.title("Canonical Correlation Analysis")
        plt.xticks(ticks=np.arange(0.5, len(correlation_results), 1), labels=correlation_results.keys(), rotation=90)
        plt.yticks(ticks=np.arange(0.5, len(activation_keys), 1), labels=activation_keys, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"layer_cca_heatmap.png"))
        plt.close()

    # Plotting RSA results
    if rsa:
        rsa_matrix = np.zeros((len(activation_keys), len(correlation_results)))
        for i, act_key in enumerate(activation_keys):
            for j, other_key in enumerate(correlation_results.keys()):
                if (act_key, other_key) in rsa_results:
                    rsa_matrix[i, j] = rsa_results[(act_key, other_key)]
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(rsa_matrix, cmap='viridis')
        plt.xlabel("Feature")
        plt.ylabel("Layer")
        plt.title("Representational Similarity Analysis")
        plt.xticks(ticks=np.arange(0.5, len(correlation_results), 1), labels=correlation_results.keys(), rotation=90)
        plt.yticks(ticks=np.arange(0.5, len(activation_keys), 1), labels=activation_keys, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"layer_rsa_heatmap.png"))
        plt.close()

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(np.array(list(major_cor_results.values())).T, cmap='viridis')
    plt.xlabel("Layer")
    plt.ylabel("Feature")
    plt.title("Mean of Layers' Mean Neuron Correlations")
    plt.xticks(ticks=np.arange(0.5, len(activation_keys), 1), labels=activation_keys, rotation=90)
    plt.yticks(ticks=np.arange(0.5, len(correlation_results.keys()), 1), labels=list(correlation_results.keys()), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"layer_heatmap.png"))
    plt.close()


def process_activations(path, epoch_numbers, repetitions, timesteps=5, do_best_first=False, do_sequences_and_conv=False, use_inputs=False):
    if do_best_first:
        print('doing best first search')
        f2f_best_first(path, epoch_numbers, repetitions, timesteps=5, train_mlp=train_mlp)
        print('done')

    use_i = False
    use_non_h = False

    print('repetitions', repetitions)

    for epoch_number in epoch_numbers:
        for repetition in repetitions:
            print(path, f'activations_{epoch_number}_{repetition}.pkl')
            try:
                if epoch_number == 0:
                    with open(os.path.join(path, f'activations_prior_{repetition}.pkl'), 'rb') as f:
                        loaded_activation_data = pickle.load(f)
                else:
                    with open(os.path.join(path, f'activations_{epoch_number}_{repetition}.pkl'), 'rb') as f:
                        loaded_activation_data = pickle.load(f)
            except Exception as e:
                print('failed', e)
                continue

            correlation_data = {}
            correlation_data2 = {}
            correlation_data_lstm = {}
            correlation_data_last_ts = {}
            correlation_data_lstm_inputs = {}
            correlation_data_lstm_outputs = {}
            correlation_data_conv = {}
            correlation_data_conv_flat = {}

            keys_to_process = list(loaded_activation_data.keys())#['activations_out', 'activations_hidden_short', 'activations_hidden_long']

            skip_model_dependent = True

            for key in keys_to_process:
                if key in loaded_activation_data and len(loaded_activation_data[key]):
                    concatenated_array = np.concatenate(loaded_activation_data[key], axis=0)
                    correlation_data[key] = concatenated_array
                    if not skip_model_dependent:
                        correlation_data2[key] = concatenated_array
                    #print(f'{key} shape:', concatenated_array.shape)
                    length = concatenated_array.shape[0]

            keys_to_process2 = ['inputs', 'labels', 'pred', 'id']

            for key in keys_to_process2:
                if key in loaded_activation_data:
                    if key == "pred":
                        #if skip_model_dependent:
                        #    continue
                        arrays = []
                        for index in range(len(loaded_activation_data[key])):
                            data_array = np.array(loaded_activation_data[key][index]).astype(int)
                            one_hot = np.eye(5)[data_array.reshape(-1)].reshape(data_array.shape[0], -1)
                            arrays.append(one_hot[:, -5:])
                        correlation_data[key] = np.concatenate(arrays, axis=0)

                    elif key == "inputs":
                        # print(np.concatenate(loaded_activation_data[key], axis=0).shape)
                        real_arrays = np.concatenate(loaded_activation_data[key], axis=0).reshape((-1, 5, 5, 7, 7))


                        if do_sequences_and_conv:
                            correlation_data_conv[key + '_stacked'] = real_arrays.reshape((-1, 25, 7, 7))
                            correlation_data_conv_flat[key + '_stacked'] = real_arrays.reshape((-1, 25 * 7 * 7))
                            for t in range(5):
                                correlation_data_conv[f'{key}_t{t}'] = real_arrays[:, t, :, :, :].reshape((-1, 5, 7, 7))
                                correlation_data_conv_flat[f'{key}_t{t}'] = real_arrays[:, t, :, :, :].reshape((-1, 5 * 7 * 7))
                            for c in range(real_arrays.shape[1]):
                                correlation_data_conv[f'{key}_c{c}_last'] = real_arrays[:, -1, c, :, :].reshape((-1, 1, 7, 7))
                                correlation_data_conv_flat[f'{key}_c{c}_last'] = real_arrays[:, -1, c, :, :].reshape((-1, 1 * 7 * 7))
                                correlation_data_conv[f'{key}_c{c}_h'] = real_arrays[:, :, c, :, :].reshape((-1, 5, 7, 7))
                                correlation_data_conv_flat[f'{key}_c{c}_h'] = real_arrays[:, :, c, :, :].reshape((-1, 5 * 7 * 7))
                            for k in correlation_data_conv.keys():
                                pass
                                # print("conv shape", k, correlation_data_conv[k].shape)
                    else:
                        concatenated_array = np.concatenate(loaded_activation_data[key], axis=0)
                        correlation_data2[key] = concatenated_array
                        #print(f'{key} shape:', concatenated_array.shape)

            for key in loaded_activation_data:
                if not use_i and "i-" in key:
                    continue
                new_key = key.replace("act_", "")
                if "act_" in key:
                    arrays = []
                    hist_arrays = []
                    for index in range(len(loaded_activation_data[key])):
                        data_array = np.array(loaded_activation_data[key][index]).astype(int)
                        a_len = data_array.shape[-1]
                        if new_key == 'inputs':
                            reshaped_array = data_array.reshape((5, 5, 7, 7))
                            arrays.append(reshaped_array)

                        if key in ["act_opponents", "act_label_vision"]:
                            one_hot = np.eye(2)[data_array.reshape(-1)].reshape(data_array.shape[0], -1)
                            arrays.append(one_hot[:, -2:])
                            hist_arrays.append(one_hot[:, :])
                        elif key == "act_i-informedness":
                            one_hot = np.eye(3)[data_array.reshape(-1)].reshape(data_array.shape[0], data_array.shape[1] * 3)
                            arrays.append(one_hot[:, -6:])
                            hist_arrays.append(one_hot[:, :])
                        else:
                            arrays.append(data_array[:, -(a_len // timesteps):])
                            hist_arrays.append(data_array[:, :])
                    if key != "act_vision" and key != "act_box-updated" and key != "act_exist":
                        # these variables are always the same at the end of the task, but differ during
                        if use_non_h or new_key == "i-informedness" or new_key == "opponents":  # this one gets used otherwise
                            correlation_data2[new_key] = np.concatenate(arrays, axis=0)
                        # print('cor2shape', new_key, correlation_data2[new_key].shape)
                    if key != "act_opponents" and key != "act_informedness" and key != 'act_id' and key != 'act_vision':
                        correlation_data2[new_key + "_h"] = np.concatenate(hist_arrays, axis=0)
                        value = correlation_data2[new_key + "_h"]
                        num_sequences, feature_dim_times_timesteps = value.shape
                        feature_dim = feature_dim_times_timesteps // timesteps
                        #print(key)
                        new_value = value.reshape((num_sequences, timesteps, feature_dim))
                        correlation_data_lstm[new_key + "_h"] = new_value

                        correlation_data_last_ts[new_key] = correlation_data_lstm[new_key + "_h"][:, -1, :]
                        #print('corlstmshape', new_key, correlation_data_last_ts[new_key].shape)

                        if do_sequences_and_conv:
                            modified_sequences = np.zeros((num_sequences * timesteps, timesteps, feature_dim))
                            modified_final = np.zeros((num_sequences * timesteps, feature_dim))
                            for i in range(num_sequences):
                                for num_steps in range(1, timesteps + 1):
                                    sequence = np.zeros((timesteps, feature_dim))
                                    sequence[-num_steps:] = new_value[i, :num_steps]
                                    modified_sequences[i * timesteps + (num_steps - 1)] = sequence
                                    modified_final[i * timesteps + (num_steps - 1)] = sequence[-1]

                            correlation_data_lstm_inputs[new_key] = modified_sequences
                            correlation_data_lstm_outputs[new_key] = modified_final
                        # print('corstepsshape', new_key, correlation_data_lstm_inputs[new_key].shape)
                        # print('corlastshape', new_key, correlation_data_lstm_outputs[new_key].shape)
                    #elif key == 'act_opponents':
                    #    correlation_data_last_ts[key] = correlation_data[key]


            correlation_data2["scalar"] = np.random.randint(2, size=(1,))
            correlation_data["scalar"] = np.random.randint(2, size=(8844, 1,))
            # correlation_data_conv["rand_vec5"] = np.random.randint(2, size=(8, 1, 7, 7))
            # correlation_data_conv_flat["rand_vec5"] = np.random.randint(2, size=(length, 1 * 7 * 7))
            print(correlation_data.keys())
            pred_d = correlation_data["pred"]
            labels_d = correlation_data["labels"]
            #pred_d2 = correlation_data_lstm_outputs["vision"]
            random_indices = np.random.choice(len(pred_d), size=3, replace=False)

            #correlation_data_last_ts["pred"] = correlation_data["pred"]
            #print("pred datapoints:", pred_d[random_indices], labels_d[random_indices], correlation_data["act_big-loc"][random_indices], correlation_data["act_big-b-loc"][random_indices])
            # print("pred datapoints:", pred_d2[random_indices])
            # print(correlation_data_lstm_inputs.keys())

            test_keys = ['act_opponents', 'labels', 'pred',
                         'act_big-loc', 'act_small-loc',
                         'act_big-box', 'act_small-box',
                         'act_b-loc', 'act_b-box',
                         'act_target-loc', 'act_target-box',
                         "act_fb-loc", "act_vision", "act_fb-exist"]

            for key in test_keys:
                if key not in correlation_data_last_ts.keys():
                    correlation_data_last_ts[key] = correlation_data[key]
            #print(correlation_data["inputs"].shape, correlation_data_last_ts['act_vision'].shape)
            ### Correlation thing
            if True:
                print('running IFR calculation', path)
                activation_keys = [x for x in correlation_data.keys() if 'act_' not in x and x not in ['inputs', 'oracles', '', ]]
                correlation_thing(activation_keys, correlation_data, correlation_data_last_ts, test_keys, path, repetition=repetition, epoch=epoch_number, use_inputs=epoch_number==epoch_numbers[0] and use_inputs)

            # MLP F2F DATA
            remove_labels = []
            run = False
            models = ['mlp2']
            # models = ['mlp2', 'mlp2bn', 'mlp2d', 'mlp2d2', 'mlp2d3', 'mlp2ln', 'mlp2s', 'mlp3', 'mlp1', 'mlp2l2reg', 'linear', 'mlp2c5', 'mlp2c10', 'mlp2c16', 'linearl2reg', 'mlp2dl2reg', 'mlp3l2reg', 'mlp1l2reg', 'mlp2e50', 'mlp2l2rege50', 'mlp3l2rege50', 'mlp1l2rege50']
            # models = ['mlp2l2reg', 'mlp2ln', 'mlp2d']
            # models = ['rnn', 'rnnd1', 'rnnd5', 'lstm', 'lstmd1', 'lstmd5']

            compose = False
            split_by_regime = False
            compose_targets = [None]
            # compose_targets = ['b-loc', 'target-size', 'target-loc', 'loc']
            # remove_labels = ['labels']
            # compose_targets = ['b-loc_h','target-size_h', 'target-loc_h', 'loc_h']
            compose_targets = ['labels', 'b-loc', 'target-size', 'target-loc', 'loc']
            image_inputs = False
            image_outputs = False
            num_epochs = 25
            get_stats = True

            # skip down-stream labels for ff2f tests
            if compose:
                for label in remove_labels:
                    correlation_data2.pop(label)
                    if label in correlation_data_lstm.keys():
                        correlation_data_lstm.pop(label)
            else:
                compose_targets = [None]

            if image_inputs:
                if "conv" in models[0]:
                    used_cor_inputs = correlation_data_conv
                else:
                    used_cor_inputs = correlation_data_conv_flat
            else:
                used_cor_inputs = correlation_data2

            if split_by_regime:
                unique_regimes = np.unique(correlation_data2['i-informedness'], axis=0)
                # we don't have this for lstm data yet
            else:
                unique_regimes = [None]


            if run:
                print('running', "ff2" + str(compose_targets) if compose else "f2f", str(models), 'regime-split:', split_by_regime, 'epochs:', num_epochs)
                for model_type in models:
                    keys1, output_keys, size1, size, second_input_keys = get_keys(model_type, used_cor_inputs, correlation_data2,
                                                                                  correlation_data_conv, correlation_data_lstm, correlation_data_lstm_inputs, correlation_data_lstm_outputs, compose,
                                                                                  use_conv_inputs=image_inputs, image_output=image_outputs)

                    loss_matrices = {str(regime): pd.DataFrame(index=keys1, columns=output_keys) for regime in
                                     unique_regimes}
                    # val_loss_matrix = np.zeros((size1, size))
                    #print(keys1, output_keys)
                    with tqdm.tqdm(total=size1 * size * len(unique_regimes) * len(compose_targets),
                                   desc='Processing key pairs') as pbar:
                        for target in compose_targets:
                            print("fitting combinations to target feature:", target)
                            for i, key1 in enumerate(keys1):
                                if "lstm" in model_type or "rnn" in model_type:
                                    input_data = correlation_data_lstm_inputs[key1]
                                    regime_data = None  # correlation_data_lstm_outputs["informedness"]
                                    opponents_data = None  # correlation_data_lstm_outputs["opponents"]
                                    realkeys = keys1
                                else:
                                    input_data = used_cor_inputs[key1]
                                    regime_data = correlation_data2["i-informedness"]
                                    opponents_data = correlation_data2["opponents"]
                                    realkeys = output_keys  # we want this to be cor2 keys but only if not comparative

                                if compose:
                                    realkeys = second_input_keys
                                cat = "mlp2c" not in model_type
                                print('concatenating:', cat)

                                for j, key2 in enumerate(realkeys):
                                    if compose:
                                        if j < i:
                                            continue
                                        output_data = correlation_data2[target]
                                        if "lstm" in model_type or "rnn" in model_type:
                                            if key2 not in keys1:
                                                continue
                                            input_data = np.concatenate([correlation_data_lstm[key1], correlation_data_lstm[key2]], axis=2)
                                        elif cat:
                                            print(key1, key2)
                                            input_data = np.concatenate([used_cor_inputs[key1], used_cor_inputs[key2]], axis=1)
                                        elif not cat:
                                            input_data = used_cor_inputs[key1]
                                            input_data2 = used_cor_inputs[key2]
                                    elif image_outputs:
                                        output_data = correlation_data_conv_flat[key2]
                                    elif "lstm" in model_type or "rnn" in model_type:
                                        output_data = correlation_data_lstm_outputs[key2]
                                    else:
                                        output_data = correlation_data2[key2]
                                    # if not cat:
                                    # print("using key2", key2)
                                    for regime in unique_regimes:
                                        assert input_data.shape[0] == output_data.shape[0]
                                        val_loss, train_losses, val_losses, val_losses_indy, val_acc, model, _ = \
                                            train_mlp(input_data, output_data, regime_data=regime_data, regime=regime, opponents_data=opponents_data,
                                                      num_epochs=num_epochs, model_type=model_type, input_data2=input_data2 if not cat else None, )

                                        loss_matrices[str(regime)].at[key1, key2] = val_loss
                                        print(key1, key2, input_data.shape, output_data.shape, val_loss, val_acc)
                                        pbar.update(1)

                            for regime in unique_regimes:
                                name = str(regime) if regime is not None else ""
                                if image_inputs:
                                    name += "c"
                                    if not image_outputs:
                                        name += "v"
                                if compose:
                                    loss_matrices[str(regime)].to_csv(os.path.join(path, f"{name}ff2l_{target}_loss_matrix_{model_type}.csv"))
                                else:
                                    loss_matrices[str(regime)].to_csv(os.path.join(path, f"{name}f2f_loss_matrix_{model_type}.csv"))

            histogram = []
            histogram2 = []
            if compose:
                histogram = {target: [] for target in compose_targets}
                histogram2 = {target: [] for target in compose_targets}

            models_used = []
            for model_type in models:
                keys1, keys, size1, size, input2keys = get_keys(model_type, used_cor_inputs, correlation_data2,
                                                                correlation_data_conv, correlation_data_lstm, correlation_data_lstm_inputs, correlation_data_lstm_outputs, compose,
                                                                use_conv_inputs=image_inputs, image_output=image_outputs)

                min_matrix_f2f = pd.DataFrame(np.inf, index=keys1, columns=keys)
                min_matrix_ff2l = {}
                for target in compose_targets:
                    min_matrix_ff2l[target] = pd.DataFrame(np.inf, index=keys1, columns=input2keys)

                for regime in unique_regimes:
                    name = str(regime) if regime is not None else ""
                    name += "c" * bool(image_inputs) + "v" * (bool(image_inputs) and not image_outputs)
                    if not compose:
                        matrix_path = os.path.join(path, f"{name}f2f_loss_matrix_{model_type}.csv")
                        if os.path.exists(matrix_path):
                            val_loss_matrix_f2f = pd.read_csv(matrix_path, index_col=0, header=0)
                            min_matrix_f2f = min_matrix_f2f.combine(val_loss_matrix_f2f, np.minimum)
                            if not get_stats:
                                print(f"trying to make {name}f2f-{model_type}.png")
                                plt.figure(figsize=(12, 8))
                                sns.heatmap(val_loss_matrix_f2f, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1.)
                                plt.title(f'{model_type} Validation MSE')
                                plt.xlabel('Target')
                                plt.ylabel('Input')
                                plt.tight_layout()
                                plt.savefig(os.path.join(path, f"{name}f2f-{model_type}.png"))
                            else:
                                if not split_by_regime:
                                    mat = val_loss_matrix_f2f.to_numpy().flatten()
                                    mat = np.nan_to_num(mat, nan=np.nan, posinf=np.nan, neginf=np.nan)[np.isfinite(mat)]

                                    print(model_type, mat.mean(), mat.std(), (mat < 0.1).sum(), (mat < 0.05).sum(), (mat < 0.02).sum(), (mat < 0.01).sum())
                                    histogram.append(mat)
                                    models_used.append(model_type)

                                if str(regime) == str(unique_regimes[0]):
                                    pass
                                    mat = min_matrix_f2f.to_numpy().flatten()
                                    # mat = np.nan_to_num(mat, nan=np.nan, posinf=np.nan, neginf=np.nan)[np.isfinite(mat)]
                                    # models_used.append(model_type)
                                    # print(model_type, mat.mean(), mat.std(), (mat < 0.1).sum(), (mat < 0.05).sum(), (mat < 0.02).sum(), (mat < 0.01).sum())
                                    # histogram.append(mat)
                        else:
                            print("couldn't find file", matrix_path)
                    if compose:
                        for target in compose_targets:
                            matrix_path = os.path.join(path, f"{name}ff2l_{target}_loss_matrix_{model_type}.csv")
                            if os.path.exists(matrix_path):
                                val_loss_matrix_ff2l = pd.read_csv(matrix_path, index_col=0, header=0)
                                min_matrix_ff2l[target] = min_matrix_ff2l[target].combine(val_loss_matrix_ff2l, np.minimum)

                                if model_type == "lstm":
                                    pass
                                    # order = ['b-loc_h', 'loc_h', 'target_h', 'vision_h']
                                    # order_indices = [keys1.index(label) for label in order]
                                    # val_loss_matrix_ff2l = val_loss_matrix_ff2l[:, :len(order)][np.ix_(order_indices, order_indices)]
                                if not get_stats:
                                    print('rendering matrix for', target)
                                    plt.figure(figsize=(12, 8))
                                    sns.heatmap(val_loss_matrix_ff2l, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=0.25)
                                    plt.title(f'{model_type} FF2F {target} Validation MSE')
                                    plt.xlabel('Input2')
                                    plt.ylabel('Input1')
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(path, f"{name}ff2f-{target}-{model_type}.png"))
                                else:
                                    mat = min_matrix_f2f.to_numpy().flatten()
                                    mat = np.nan_to_num(mat, nan=np.nan, posinf=np.nan, neginf=np.nan)[np.isfinite(mat)]
                                    print(model_type, mat.mean(), mat.std(), (mat < 0.1).sum(), (mat < 0.05).sum(), (mat < 0.02).sum(), (mat < 0.01).sum())
                                    histogram[target].append(mat)
                                    models_used.append(model)
                            else:
                                print("couldn't find file", matrix_path)

                name = "c" * bool(image_inputs) + "v" * (bool(image_inputs) and not image_outputs)
                # minimum matrices
                if split_by_regime:
                    if not compose:
                        if not get_stats:
                            plt.figure(figsize=(12, 8))
                            sns.heatmap(min_matrix_f2f, annot=True, fmt=".2f", cmap="coolwarm",
                                        xticklabels=keys, yticklabels=keys1, vmin=0, vmax=0.25)
                            plt.title(f'Minimum of Regimes {model_type} Validation MSE (F2F)')
                            plt.xlabel('Target')
                            plt.ylabel('Input')
                            plt.tight_layout()
                            plt.savefig(os.path.join(path, f"min_{name}f2f-{model_type}.png"))
                        else:
                            mat = min_matrix_f2f.to_numpy().flatten()
                            mat = np.nan_to_num(mat, nan=np.nan, posinf=np.nan, neginf=np.nan)[np.isfinite(mat)]
                            print(model_type, mat.mean(), mat.std(), (mat < 0.1).sum(), (mat < 0.05).sum(), (mat < 0.02).sum(), (mat < 0.01).sum(), mat.shape)
                            # we save both this min histogram... and we try to get the "None" histogram for this model type
                            matrix_path = os.path.join(path, f"f2f_loss_matrix_{model_type}.csv")
                            if os.path.exists(matrix_path):
                                val_loss_matrix_f2f = pd.read_csv(matrix_path, index_col=0, header=0)
                                mat2 = val_loss_matrix_f2f.to_numpy().flatten()
                                mat2 = np.nan_to_num(mat2, nan=np.nan, posinf=np.nan, neginf=np.nan)[np.isfinite(mat2)]
                            if mat.shape[0] != 0:
                                histogram.append(mat)
                                models_used.append(model_type)
                                if os.path.exists(matrix_path):
                                    histogram2.append(mat2)

                    else:
                        for target in compose_targets:
                            if not get_stats:
                                print('rendering min matrix for', target)
                                plt.figure(figsize=(12, 8))
                                sns.heatmap(min_matrix_ff2l[target], annot=True, fmt=".2f", cmap="coolwarm",
                                            xticklabels=keys1, yticklabels=keys1, vmin=0, vmax=0.25)
                                plt.title(f'Minimum of Regimes {model_type} FF2F {target} Validation MSE')
                                plt.xlabel('Input2')
                                plt.ylabel('Input1')
                                plt.tight_layout()
                                plt.savefig(os.path.join(path, f"min_{name}ff2f-{target}-{model_type}.png"))
                            else:
                                mat = min_matrix_ff2l[target].to_numpy().flatten()
                                mat = np.nan_to_num(mat, nan=np.nan, posinf=np.nan, neginf=np.nan)
                                mat = mat[np.isfinite(mat)]
                                print(model_type, mat.mean(), mat.std(), (mat < 0.1).sum(), (mat < 0.05).sum(), (mat < 0.02).sum(), (mat < 0.01).sum())
                                histogram[target].append(mat)
                                models_used.append(model)

            if get_stats:
                # make a histogram of the min matrices
                if compose:
                    for target in compose_targets:
                        try:
                            name = "c" * bool(image_inputs) + "v" * (bool(image_inputs) and not image_outputs) + "r" * bool(split_by_regime)
                            plot_bars(histogram[target], histogram2, path, name, target, models_used)
                        except BaseException as e:
                            print("failed", target, e)
                else:
                    try:
                        print('len', len(histogram))
                        name = "c" * bool(image_inputs) + "v" * (bool(image_inputs) and not image_outputs) + "r" * bool(split_by_regime)

                        plot_histogram(path, name, target, histogram, models_used)
                        plot_scatter(histogram, histogram2, keys, keys1, min_matrix_f2f, models_used, path, name, target)
                        plot_bars(histogram, histogram2, path, name, target, models_used)

                    except BaseException as e:
                        print("failed", target, e)

            print('finished')

            if run:

                name = "c" * bool(image_inputs) + "v" * (bool(image_inputs) and not image_outputs)
                matrices = {model: pd.read_csv(os.path.join(path, f"{name}f2f_loss_matrix_{model}.csv")) for model in models}
                for i, model1 in enumerate(models):
                    for j, model2 in enumerate(models):
                        if j <= i:  # This ensures we only consider each pair once, avoiding redundancy
                            continue

                        keys = sorted(correlation_data2.keys())
                        if model1 == "lstm" or model2 == "lstm":
                            keys1 = sorted(correlation_data_lstm.keys())
                        else:
                            keys1 = keys

                        # val_loss_matrix_full = np.full((len(keys), len(keys)), np.nan)
                        val_loss_matrix_full = pd.DataFrame(index=keys, columns=keys, data=np.nan)
                        shared_keys_indices = [keys.index(key) for key in keys1]
                        all_indices = [keys.index(key) for key in keys]
                        lstm_keys_indices = [keys1.index(key) for key in keys1]

                        for idx1, ldx1 in zip(shared_keys_indices, lstm_keys_indices):
                            for idx2 in all_indices:
                                if keys[idx1] in keys1:
                                    model1_index = ldx1 if model1 == "lstm" else idx1
                                    model2_index = ldx1 if model2 == "lstm" else idx1
                                    val_loss_matrix_full[idx1, idx2] = matrices[model2][model2_index, idx2] - \
                                                                       matrices[model1][model1_index, idx2]
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(val_loss_matrix_full, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.06, vmax=0.06)
                        plt.title(f'F2F Validation MSE ({model2} - {model1})')
                        plt.xlabel('Target')
                        plt.ylabel('Input')
                        plt.tight_layout()
                        plt.savefig(os.path.join(path, f"f2f-difference-{model1}-{model2}.png"))

            # END
