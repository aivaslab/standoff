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
        return torch.mean((y_pred - y_true)**2) / (torch.var(y_true) + 1e-8)

def get_model_type(model_type,  input_size, input_size2, output_size, hidden_size, device, num_epochs):

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

def train_mlp(inputs, other_data, regime_data, regime, opponents_data, patience=2, num_prints=5, num_epochs=25,
              model_type="linear", input_data2=None):
    # Parameters
    input_size = inputs.shape[-1] if "conv" not in model_type else inputs.shape[1]
    input_size2 = 0
    if input_data2 is not None:
        input_size2 = input_data2.shape[-1] if "conv" not in model_type else inputs.shape[1]
    output_size = other_data.shape[1]
    hidden_size = 32
    learning_rate = 1e-3
    batch_size = 128

    all_indices = np.arange(len(inputs))
    train_indices, val_indices = train_test_split(all_indices, test_size=0.10, random_state=42)

    if regime is not None:
        zero_opponents_indices = np.where(np.all(opponents_data == 0, axis=1))[0]
        regime_indices = np.where(np.all(regime_data == regime, axis=1))[0]
        combined_indices = np.union1d(regime_indices, zero_opponents_indices)
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
        val_dataset = TensorDataset(torch.tensor(act_val, dtype=torch.float32).to(device),
                                    torch.tensor(other_val, dtype=torch.float32).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model, l2_reg, num_epochs = get_model_type(model_type, input_size, input_size2, output_size, hidden_size, device, num_epochs)

    criterion = nn.MSELoss()
    slowcriterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= 1e-5 if l2_reg else 0)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    last_epoch_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        for data in train_loader:
            act_vector, *optional_data = data[:-1]
            other_vector = data[-1]
            outputs = model(act_vector, *optional_data) if input_data2 is not None else model(act_vector)
            loss = criterion(outputs, other_vector)
            epoch_train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            if epoch == num_epochs - 1:
                for data in val_loader:
                    act_vector, *optional_data = data[:-1]
                    other_vector = data[-1]
                    outputs = model(act_vector, *optional_data) if input_data2 is not None else model(act_vector)
                    val_loss = criterion(outputs, other_vector)
                    val_loss_indy = slowcriterion(outputs, other_vector).mean(dim=1)
                    epoch_val_losses.append(val_loss.item())
                    last_epoch_val_losses.extend(val_loss_indy.tolist())
            else:
                for data in val_loader:
                    act_vector, *optional_data = data[:-1]
                    other_vector = data[-1]
                    outputs = model(act_vector, *optional_data) if input_data2 is not None else model(act_vector)
                    val_loss = criterion(outputs, other_vector)
                    epoch_val_losses.append(val_loss.item())

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        # if (epoch + 1) % (num_epochs // num_prints) == 0:
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


def get_keys(model_type, used_cor_inputs, correlation_data2, correlation_data_conv, correlation_data_lstm,  correlation_data_lstm_inputs, correlation_data_lstm_outputs, compose,
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



def process_activations(path, epoch_numbers, repetitions, timesteps=5):

    f2f_best_first(path, epoch_numbers, repetitions, timesteps=5, train_mlp=train_mlp)
    print('done')

    use_i = False
    use_non_h = False

    for epoch_number in epoch_numbers:
        for repetition in repetitions:
            with open(os.path.join(path, f'activations_{epoch_number}_{repetition}.pkl'), 'rb') as f:
                loaded_activation_data = pickle.load(f)

            correlation_data = {}
            correlation_data2 = {}
            correlation_data_lstm = {}
            correlation_data_lstm_inputs = {}
            correlation_data_lstm_outputs = {}
            correlation_data_conv = {}
            correlation_data_conv_flat = {}

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

                    elif key == "inputs":
                        #print(np.concatenate(loaded_activation_data[key], axis=0).shape)
                        real_arrays = np.concatenate(loaded_activation_data[key], axis=0).reshape((-1, 5, 5, 7, 7))

                        correlation_data_conv[key + '_stacked'] = real_arrays.reshape((-1, 25, 7, 7))
                        correlation_data_conv_flat[key + '_stacked'] = real_arrays.reshape((-1, 25 * 7 * 7))

                        for t in range(5):
                            correlation_data_conv[f'{key}_t{t}'] = real_arrays[:, t, :, :, :].reshape((-1, 5, 7, 7))
                            correlation_data_conv_flat[f'{key}_t{t}'] = real_arrays[:, t, :, :, :].reshape((-1, 5 * 7 * 7))
                        for c in range(real_arrays.shape[1]):
                            correlation_data_conv[f'{key}_c{c}_last'] = real_arrays[:, -1, c, :, :].reshape( (-1, 1, 7, 7))
                            correlation_data_conv_flat[f'{key}_c{c}_last'] = real_arrays[:, -1, c, :, :].reshape( (-1, 1 * 7 * 7))
                            correlation_data_conv[f'{key}_c{c}_h'] = real_arrays[:, :, c, :, :].reshape((-1, 5, 7, 7))
                            correlation_data_conv_flat[f'{key}_c{c}_h'] = real_arrays[:, :, c, :, :].reshape((-1, 5 * 7 * 7))
                        for k in correlation_data_conv.keys():
                            pass
                            #print("conv shape", k, correlation_data_conv[k].shape)
                    else:
                        concatenated_array = np.concatenate(loaded_activation_data[key], axis=0)
                        correlation_data2[key] = concatenated_array
                    #print(f'{key} shape:', concatenated_array.shape)

            for key in loaded_activation_data:
                if not use_i and "i-" in key:
                    continue
                new_key = key.replace("act_label_", "")
                if "act_label_" in key:
                    arrays = []
                    hist_arrays = []
                    for index in range(len(loaded_activation_data[key])):
                        data_array = np.array(loaded_activation_data[key][index]).astype(int)
                        a_len = data_array.shape[-1]
                        if new_key == 'inputs':
                            reshaped_array = data_array.reshape((5, 5, 7, 7))
                            arrays.append(reshaped_array)

                        if key in ["act_label_opponents", "act_label_vision"]:
                            one_hot = np.eye(2)[data_array.reshape(-1)].reshape(data_array.shape[0], -1)
                            arrays.append(one_hot[:, -2:])
                            hist_arrays.append(one_hot[:, :])
                        elif key == "act_label_informedness":
                            one_hot = np.eye(3)[data_array.reshape(-1)].reshape(data_array.shape[0],data_array.shape[1] * 3)
                            arrays.append(one_hot[:, -6:])
                            hist_arrays.append(one_hot[:, :])
                        else:
                            arrays.append(data_array[:, -(a_len // timesteps):])
                            hist_arrays.append(data_array[:, :])
                    if key != "act_label_vision" and key != "act_label_box-updated" and key != "act_label_exist":
                        # these variables are always the same at the end of the task, but differ during
                        if use_non_h or new_key == "informedness" or new_key == "opponents": #this one gets used otherwise
                            correlation_data2[new_key] = np.concatenate(arrays, axis=0)
                        #print('cor2shape', new_key, correlation_data2[new_key].shape)
                    if key != "act_label_opponents" and key != "act_label_informedness":
                        correlation_data2[new_key + "_h"] = np.concatenate(hist_arrays, axis=0)
                        value = correlation_data2[new_key + "_h"]
                        num_sequences, feature_dim_times_timesteps = value.shape
                        feature_dim = feature_dim_times_timesteps // timesteps
                        new_value = value.reshape((num_sequences, timesteps, feature_dim))
                        correlation_data_lstm[new_key + "_h"] = new_value
                        #print('corlstmshape', new_key, correlation_data_lstm[new_key + "_h"].shape)

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
                        #print('corstepsshape', new_key, correlation_data_lstm_inputs[new_key].shape)
                        #print('corlastshape', new_key, correlation_data_lstm_outputs[new_key].shape)

            correlation_data2["rand_vec5"] = np.random.randint(2, size=(length, 5))
            correlation_data_conv["rand_vec5"] = np.random.randint(2, size=(length, 1, 7, 7))
            correlation_data_conv_flat["rand_vec5"] = np.random.randint(2, size=(length, 1 * 7 * 7))
            pred_d = correlation_data_lstm_inputs["vision"]
            pred_d2 = correlation_data_lstm_outputs["vision"]
            random_indices = np.random.choice(len(pred_d), size=3, replace=False)
            #print("pred datapoints:", pred_d[random_indices])
            #print("pred datapoints:", pred_d2[random_indices])
            print(correlation_data_lstm_inputs.keys())
            activation_keys = ['activations_out']  # , 'activations_hidden_short', 'activations_hidden_long']

            # MLP F2F DATA
            remove_labels = []
            run = True
            models = ['mlp2']
            #models = ['mlp2', 'mlp2bn', 'mlp2d', 'mlp2d2', 'mlp2d3', 'mlp2ln', 'mlp2s', 'mlp3', 'mlp1', 'mlp2l2reg', 'linear', 'mlp2c5', 'mlp2c10', 'mlp2c16', 'linearl2reg', 'mlp2dl2reg', 'mlp3l2reg', 'mlp1l2reg', 'mlp2e50', 'mlp2l2rege50', 'mlp3l2rege50', 'mlp1l2rege50']
            #models = ['mlp2l2reg', 'mlp2ln', 'mlp2d']
            #models = ['rnn', 'rnnd1', 'rnnd5', 'lstm', 'lstmd1', 'lstmd5']

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
                unique_regimes = np.unique(correlation_data2['informedness'], axis=0)
                # we don't have this for lstm data yet
            else:
                unique_regimes = [None]

            if run:
                print('running', "ff2" + str(compose_targets) if compose else "f2f", str(models), 'regime-split:',
                      split_by_regime, 'epochs:', num_epochs)
                for model_type in models:
                    keys1, output_keys, size1, size, second_input_keys = get_keys(model_type, used_cor_inputs, correlation_data2,
                                                               correlation_data_conv, correlation_data_lstm, correlation_data_lstm_inputs, correlation_data_lstm_outputs, compose,
                                                               use_conv_inputs=image_inputs, image_output=image_outputs)


                    loss_matrices = {str(regime): pd.DataFrame(index=keys1, columns=output_keys) for regime in
                                     unique_regimes}
                    # val_loss_matrix = np.zeros((size1, size))
                    print(keys1, output_keys)
                    with tqdm.tqdm(total=size1 * size * len(unique_regimes) * len(compose_targets),
                                   desc='Processing key pairs') as pbar:
                        for target in compose_targets:
                            print("fitting combinations to target feature:", target)
                            for i, key1 in enumerate(keys1):
                                if "lstm" in model_type or "rnn" in model_type:
                                    input_data = correlation_data_lstm_inputs[key1]
                                    regime_data = None#correlation_data_lstm_outputs["informedness"]
                                    opponents_data = None#correlation_data_lstm_outputs["opponents"]
                                    realkeys = keys1
                                else:
                                    input_data = used_cor_inputs[key1]
                                    regime_data = correlation_data2["informedness"]
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
                                    #if not cat:
                                        #print("using key2", key2)
                                    for regime in unique_regimes:
                                        assert input_data.shape[0] == output_data.shape[0]
                                        val_loss, train_losses, val_losses, val_losses_indy = \
                                            train_mlp(input_data, output_data, regime_data=regime_data, regime=regime, opponents_data=opponents_data,
                                                      num_epochs=num_epochs, model_type=model_type, input_data2=input_data2 if not cat else None,)
                                        loss_matrices[str(regime)].at[key1, key2] = val_loss
                                        print(key1, key2, input_data.shape, output_data.shape, val_loss)
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
                                    #mat = np.nan_to_num(mat, nan=np.nan, posinf=np.nan, neginf=np.nan)[np.isfinite(mat)]
                                    #models_used.append(model_type)
                                    #print(model_type, mat.mean(), mat.std(), (mat < 0.1).sum(), (mat < 0.05).sum(), (mat < 0.02).sum(), (mat < 0.01).sum())
                                    #histogram.append(mat)
                        else:
                            print("couldn't find file", matrix_path)
                    if compose:
                        for target in compose_targets:
                            matrix_path = os.path.join(path, f"{name}ff2l_{target}_loss_matrix_{model_type}.csv")
                            if os.path.exists(matrix_path):
                                val_loss_matrix_ff2l = pd.read_csv( matrix_path, index_col=0,header=0)
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

            for act_key in activation_keys:
                activations = correlation_data[act_key]
                correlation_results = {}

                # tsne_model = TSNE(n_components=2, perplexity=30, n_iter=300)
                # tsne_values = tsne_model.fit_transform(activations)
                egg = 1

                # this is for model activations
                run_mlp_test(correlation_data2, act_key, activations, path)
                continue

                if egg == 0:

                    if egg == 0:
                        unique_vectors = get_unique_arrays(other_data)
                        integer_labels = get_integer_labels_for_data(other_data, unique_vectors)

                        plt.figure(figsize=(10, 6))
                        labels = correlation_data2[other_key]
                        plt.scatter(tsne_values[:, 0], tsne_values[:, 1], c=integer_labels, edgecolors='black',
                                    cmap='tab10')
                        plt.title(f't-SNE of {act_key} colored by {other_key}')
                        plt.xlabel('Dimension 1')
                        plt.ylabel('Dimension 2')
                        plt.tight_layout()
                        plt.savefig(os.path.join(path, f"{act_key}-tsne-colored-by-{other_key}.png"))
                        plt.close()

                        result_key = f"{other_key}"

                        # print('data', other_data[0], len(get_unique_arrays(other_data)))

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
                plt.xticks(ticks=np.arange(0.5, len(correlation_results), 1), labels=correlation_results.keys(),
                           rotation=90)

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
