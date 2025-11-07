import hashlib
import pickle
from functools import lru_cache

import h5py
import sys
import os


import pandas as pd

from .utils.conversion import calculate_informedness
from .utils.evaluation import get_relative_direction
from torch.utils.data._utils.collate import default_collate

sys.path.append(os.getcwd())

from .objects import *
from .agents import GridAgentInterface
from .pz_envs import env_from_config
# import src.pz_envs
from torch.utils.data import Dataset
import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch


def identify_mismatches(info, env, informedness, params, data_name, configName, eName, loc, b_loc, counts):
    inf = calculate_informedness(loc, b_loc)
    if informedness != inf and params["num_puppets"] > 0:
        print('informedness mismatch', informedness, inf, loc, b_loc, params["num_puppets"],
              env.current_param_group_pos, env.current_param_group, configName, eName)
    if params["num_puppets"] == 0:
        if info['p_0']['shouldGetBig'] == False:
            print('opponentless big avoid', env.current_param_group_pos, env.current_param_group, configName, eName)
        if info['p_0']['shouldGetSmall'] != False:
            print('opponentless small avoid', env.current_param_group_pos, env.current_param_group, configName, eName)
    if params["num_puppets"] != int(data_name[-1]):
        print('puppet mismatch', env.current_param_group_pos, env.current_param_group, configName, eName)
    # look for accidentals:

    counts['total'] += 1
    if info['p_0']['target-size'][0] and inf[0] != 2:
        counts['big'] += 1
        #weirdly high for regime Nt1?
    if info['p_0']['target-size'][1] and inf[1] != 2:
        counts['small'] += 1

    # next we check for mismatches in target location/size
    if inf[0] == 2:
        # we should get big here
        if info['p_0']['shouldGetBig'] == True or info['p_0']['shouldGetSmall'] == False:
            print('informed size mismatch', env.current_param_group_pos, env.current_param_group, configName, eName)
    elif inf[0] < 2 and inf[1] == 2:
        if (info['p_0']['shouldGetBig'] == False or info['p_0']['shouldGetSmall'] == True):
            print('uninformed size mismatch', env.current_param_group_pos, env.current_param_group, configName, eName)
    if info['p_0']['shouldGetBig']:
        if info['p_0']['correct-loc'] != env.big_food_locations[-1]:
            print('selection mismatch', env.big_food_locations, eName)
    else:
        if info['p_0']['correct-loc'] != env.small_food_locations[-1]:
            print('selection mismatch', env.small_food_locations, eName)
@lru_cache(maxsize=None)
def one_hot(size, data):
    return np.eye(size)[data]





def serialize_data(datapoint):
    return pickle.dumps(datapoint)
def write_to_h5py(data, filename, key='data'):
    with h5py.File(filename, 'w') as f:
        f.create_dataset(key, data=data, chunks=True)

def get_h5py_file(filename):
    # Open the file
    h5f = h5py.File(filename, 'r')

    # Get the file's access property list
    fapl = h5f.id.get_access_plist()

    # Set the cache parameters
    #   1) The number of elements in the meta data cache
    #   2) The maximum number of elements in the raw data chunk cache
    #   3) The total size of the raw data chunk cache in bytes
    #   4) The preemption policy (0.0 means fully read/write through,
    #      and 1.0 means fully read/write back)
    # We set the cache size to 55.125 MB (in bytes)
    fapl.set_cache(0, 1000 * 32, int(64 * 1024 * 1024), 0.0)

    return h5f

class h5DatasetSlow(Dataset):
    def __init__(self, data_paths, labels_paths, params_paths, oracles_paths):
        self.data_paths = data_paths
        self.labels_paths = labels_paths
        self.params_paths = params_paths
        self.oracles_paths = oracles_paths if len(oracles_paths) and oracles_paths[0] else []
        self.lengths = [len(h5py.File(data_path, 'r')['data']) for data_path in data_paths]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        # Determine which file this index belongs to
        file_index = np.searchsorted(self.cumulative_lengths, index, side='right') - 1
        # Convert the global index to a local index within this file
        local_index = index - self.cumulative_lengths[file_index]

        # Now open the corresponding files and read the data, labels, params, oracles for this index
        with h5py.File(self.data_paths[file_index], 'r') as f:
            data = torch.from_numpy(f['data'][local_index]).float()

        with h5py.File(self.labels_paths[file_index], 'r') as f:
            labels = torch.from_numpy(f['data'][local_index]).float()

        with h5py.File(self.params_paths[file_index], 'r') as f:
            params = f['data'][local_index].astype(str)

        if len(self.oracles_paths):
            with h5py.File(self.oracles_paths[file_index], 'r') as f:
                oracles = torch.from_numpy(f['data'][local_index]).float()
        else:
            oracles = torch.from_numpy(np.asarray([])).float()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        labels = labels.to(device)
        oracles = oracles.to(device)
        return data, labels, params, oracles
class h5Dataset(Dataset):
    def __init__(self, data_paths, labels_paths, params_paths, oracles_paths):
        self.data_files = [get_h5py_file(path) for path in data_paths]
        self.labels_files = [get_h5py_file(path) for path in labels_paths]
        self.params_files = [get_h5py_file(path) for path in params_paths]
        self.oracles_files = [get_h5py_file(path) for path in oracles_paths] if len(oracles_paths) and oracles_paths[0] else []

        # Compute total number of samples and create an index mapping
        self.total_samples = 0
        self.index_mapping = []
        for i, data_file in enumerate(self.data_files):
            num_samples = data_file['data'].shape[0]
            self.index_mapping.extend([(i, j) for j in range(num_samples)])
            self.total_samples += num_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        file_index, local_index = self.index_mapping[index]

        data = self.data_files[file_index]['data'][local_index]
        labels = self.labels_files[file_index]['data'][local_index]
        params = self.params_files[file_index]['data'][local_index].astype(str)
        oracles = self.oracles_files[file_index]['data'][local_index] if self.oracles_files else np.asarray([])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.from_numpy(data).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)
        oracles = torch.from_numpy(oracles).float().to(device)

        return data, labels, params, oracles

    def __del__(self):
        for file in self.data_files:
            file.close()
        for file in self.labels_files:
            file.close()
        for file in self.params_files:
            file.close()
        for file in self.oracles_files:
            file.close()

class DiskLoadingDataset(Dataset):
    def __init__(self, data_files, label_files, param_files, oracle_files=None):
        self.data_files = data_files
        self.label_files = label_files
        self.param_files = param_files
        self.oracle_files = oracle_files

        # Calculate the cumulative size of the previous datasets
        self.cumulative_sizes = self._calculate_cumulative_sizes()

    def _calculate_cumulative_sizes(self):
        cumulative_sizes = []
        total_size = 0
        for data_file in self.data_files:
            with np.load(data_file, mmap_mode='r') as data:
                total_size += data['arr_0'].shape[0]
            cumulative_sizes.append(total_size)
        return cumulative_sizes

    def _find_file_index(self, idx):
        # Binary search can be used here for efficiency if the dataset is large
        file_index = 0
        while idx >= self.cumulative_sizes[file_index]:
            file_index += 1
        return file_index

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        file_index = self._find_file_index(idx)

        # Adjust idx to be relative to the file it is in
        if file_index != 0:
            idx -= self.cumulative_sizes[file_index - 1]

        # Load data from the correct file
        with np.load(self.data_files[file_index], mmap_mode='r') as data:
            data_point = torch.from_numpy(data['arr_0'][idx]).float()

        with np.load(self.label_files[file_index], mmap_mode='r') as label:
            label_point = torch.from_numpy(label['arr_0'][idx]).float()

        with np.load(self.param_files[file_index], mmap_mode='r') as param:
            param_point = param['arr_0'][idx]

        if self.oracle_files:
            with np.load(self.oracle_files[file_index], mmap_mode='r') as oracle:
                oracle_point = torch.from_numpy(oracle['arr_0'][idx]).float()
        else:
            oracle_point = torch.tensor([])

        return data_point, label_point, param_point, oracle_point

class CustomDataset(Dataset):
    def __init__(self, data, labels, params, oracles, metrics=None):
        self.data = data
        self.labels = torch.from_numpy(labels.astype(np.int8))
        self.params = params
        self.oracles = torch.from_numpy(oracles.astype(np.int8))
        self.metrics = metrics

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.from_numpy(pickle.loads(self.data[index])).float().to(device)
        labels = self.labels[index].float().to(device)
        oracles = self.oracles[index].float().to(device)

        metrics = self.metrics[index] if len(self.metrics) else None
        return data, labels, self.params[index], oracles, metrics


class SimpleMultiDataset(Dataset):
    def __init__(self, data_list, label_list, param_list, oracle_list=None, metrics=None, act_list=None):
        all_data = []
        for data_arr in data_list:
            for d in data_arr:
                all_data.append(torch.from_numpy(pickle.loads(d)).float())
        self.data = torch.stack(all_data)
        self.labels = torch.from_numpy(np.concatenate(label_list)).float()
        all_params = []
        for params in param_list:
            for p in params:
                if isinstance(p, str):
                    byte_list = [ord(c) for c in p]
                    padded_byte_list = byte_list + [0] * (12 - len(byte_list))
                    all_params.append(torch.tensor(padded_byte_list, dtype=torch.int))
                else:
                    all_params.append(torch.from_numpy(np.array(p)))
        self.params = torch.stack(all_params)
        self.oracles = torch.from_numpy(np.concatenate(oracle_list)).float() if oracle_list else None
        self.metrics = metrics

        if act_list:
            self.act_list = {}
            all_keys = set().union(*[d.keys() for d in act_list])
            for key in all_keys:
                arrays = [d[key] for d in act_list if key in d]
                self.act_list[key] = torch.from_numpy(np.concatenate(arrays)).float()
        else:
            self.act_list = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (
            self.data[idx],
            self.labels[idx],
            self.params[idx],
            self.oracles[idx] if self.oracles is not None else torch.tensor([]),
            {k: v[idx] for k, v in self.metrics.items()} if self.metrics else {},
            {k: v[idx] for k, v in self.act_list.items()} if self.act_list else {}
        )
        return sample

class BaseDatasetBig(Dataset):
    def __init__(self, data_list, labels_list, params_list, oracles_list, included_indices, metrics=None):
        self.data_list = data_list
        self.labels_list = labels_list
        self.params_list = params_list
        self.oracles_list = oracles_list
        self.metrics = metrics
        self.included_indices = included_indices

        self.labels_list = []
        for arr in labels_list:
            if arr.ndim > 1:
                arr = np.argmax(arr, axis=-1).astype(np.int64)
            else:
                arr = arr.astype(np.int64)
            self.labels_list.append(arr)

        for i in range(len(data_list)):
            if isinstance(data_list[i][0], (bytes, bytearray)):
                data_list[i] = [pickle.loads(x) for x in data_list[i]]

        self.cumulative_sizes = self._cumulative_sizes()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _cumulative_sizes(self):
        sizes = [len(x) for x in self.data_list]
        return np.cumsum(sizes)

    def _find_list_index(self, global_index):
        # find which list the global index belongs to, and the local index within that list
        list_index = np.searchsorted(self.cumulative_sizes, global_index + 1)
        if list_index > 0:
            local_index = global_index - self.cumulative_sizes[list_index - 1]
        else:
            local_index = global_index
        return list_index, local_index

    def __len__(self):
        return len(self.included_indices)
        #return self.cumulative_sizes[-1]


def custom_collate(batch):

    data, labels, params, oracles, metrics, act_labels_batch = zip(*batch)

    data = default_collate(data)
    labels = default_collate(labels).long()
    oracles = default_collate(oracles)
    params = default_collate(params)
    if metrics and isinstance(metrics[0], dict):
        metric_keys = metrics[0].keys()
        metrics_collated = {key: default_collate([d[key] for d in metrics if key in d]) for key in metric_keys}
    else:
        metrics_collated = {}
    #metrics_collated = {key: default_collate([d[key] for d in metrics]) for key in metric_keys}
    act_labels_keys = act_labels_batch[0].keys()
    act_labels_batch = {key: default_collate([d[key] for d in act_labels_batch]) for key in act_labels_keys}

    return data, labels, params, oracles, metrics_collated, act_labels_batch

class TrainDatasetBig(BaseDatasetBig):
    def __getitem__(self, idx):
        index = self.included_indices[idx]
        list_index, local_index = self._find_list_index(index)
        data = torch.from_numpy(self.data_list[list_index][local_index]).float()
        labels = torch.as_tensor(self.labels_list[list_index][local_index], dtype=torch.long)
        oracles = torch.from_numpy(self.oracles_list[list_index][local_index].astype(np.float32)) if len(
            self.oracles_list) > 1 else torch.tensor([])

        metrics = {key: self.metrics[key][index] for key in self.metrics.keys()} if self.metrics else 0

        params = self.params_list[list_index][local_index]

        if isinstance(params, str):
            byte_list = [ord(c) for c in params]
            padded_byte_list = byte_list + [0] * (12 - len(byte_list))
            params = torch.tensor(padded_byte_list, dtype=torch.int)

        return data, labels, params, oracles, metrics



class EvalDatasetBig(BaseDatasetBig):
    def __init__(self, data_list, labels_list, params_list, oracles_list, included_indices, metrics=None, act_list=None, ):
        super().__init__(data_list, labels_list, params_list, oracles_list, included_indices, metrics)
        self.act_list = act_list
        self.included_indices = included_indices

    def __getitem__(self, idx):
        index = self.included_indices[idx]
        list_index, local_index = self._find_list_index(index)
        data = torch.from_numpy(pickle.loads(self.data_list[list_index][local_index])).float()
        labels = torch.from_numpy(self.labels_list[list_index][local_index].astype(np.int8))
        oracles = torch.from_numpy(self.oracles_list[list_index][local_index].astype(np.float32)) if len(
            self.oracles_list) > 1 else torch.tensor([])
        act_labels_batch = {
            name: torch.from_numpy(self.act_list[list_index][name][local_index].astype(np.float32))
            for name in self.act_list[list_index].keys()}

        metrics = {key: self.metrics[key][index] for key in self.metrics.keys()} if self.metrics else 0

        params = self.params_list[list_index][local_index]

        if isinstance(params, str):
            byte_list = [ord(c) for c in params]
            padded_byte_list = byte_list + [0] * (12 - len(byte_list))
            params = torch.tensor(padded_byte_list, dtype=torch.int)

        '''print(type(data), type(labels), type(params), type(oracles), type(metrics), type(act_labels_batch))
        for key, value in metrics.items():
            print(f"Key: {key}, Type: {type(value)}")
        for key, value in act_labels_batch.items():
            print(f"Key: {key}, Type: {type(value)}")'''

        return data, labels,params, oracles, metrics, act_labels_batch

class cLstmModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0, is_fc=False, lr=0.0, batch_size=0.0,
                 oracle_is_target=False, oracle_early=False):
        super(cLstmModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers,
                       'output_len': output_len, 'pool_kernel_size': pool_kernel_size,
                       'pool_stride': pool_stride, 'channels': channels, 'kernels': kernels,
                       'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool, 'stride1': stride1,
                       'use_conv2': use_conv2, 'kernel_size1': kernel_size1,
                       'kernels2': kernels2, 'kernel_size2': kernel_size2, 'is_fc': is_fc, 'lr':lr, 'batch_size':batch_size,}

        input_size = 7
        self.input_frames = 5

        self.is_fc = is_fc
        self.use_pool = use_pool
        self.use_conv2 = use_conv2
        conv1_output_size = (input_size - kernel_size1 + 2 * padding1) // stride1 + 1
        self.conv1 = nn.Conv2d(channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)

        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride,
                                     padding=0 if pool_kernel_size < conv1_output_size else 1)

        pool1_output_size = (conv1_output_size - pool_kernel_size + 2 * 0 if pool_kernel_size < conv1_output_size else 1) // pool_stride + 1 if use_pool else conv1_output_size

        if use_conv2:
            conv2_output_size = (pool1_output_size - min(kernel_size2, pool1_output_size) + 2 * padding2) // stride1 + 1
            self.conv2 = nn.Conv2d(kernels, kernels2, kernel_size=min(pool1_output_size, kernel_size2), padding=padding2)
            pks = min(pool_kernel_size, conv2_output_size)
            if use_pool:
                self.pool2 = nn.MaxPool2d(kernel_size=pks, stride=pool_stride,
                                          padding=0 if pool_kernel_size < conv2_output_size else 1)
            pool2_output_size = (conv2_output_size - pks + 2 * 0 if pks < conv2_output_size else 1) // pool_stride + 1 if use_pool else conv2_output_size
        else:
            conv2_output_size = conv1_output_size
            pool2_output_size = pool1_output_size

        input_size = kernels2 * pool2_output_size * pool2_output_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_main_output = nn.Linear(hidden_size, int(output_len))

    def forward(self, x, oracle_inputs=None):
        conv_outputs = []
        for t in range(self.input_frames):
            x_t = x[:, t, :, :, :]
            x_t = self.pool(F.relu(self.conv1(x_t))) if self.use_pool else F.relu(self.conv1(x_t))
            if self.use_conv2:
                x_t = self.pool(F.relu(self.conv2(x_t))) if self.use_pool else F.relu(self.conv2(x_t))
            flattened = x_t.view(x.size(0), -1)
            conv_outputs.append(flattened)
        x = torch.stack(conv_outputs, dim=1)
        out, _ = self.rnn(x)
        outputs = self.fc_main_output(out[:, -1, :])
        return outputs


class cRNNModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0, is_fc=False, lr=0.0, batch_size=0.0,
                 oracle_is_target=False, oracle_early=False):
        super(cRNNModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers,
                       'output_len': output_len, 'pool_kernel_size': pool_kernel_size,
                       'pool_stride': pool_stride, 'channels': channels, 'kernels': kernels,
                       'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool, 'stride1': stride1,
                       'use_conv2': use_conv2, 'kernel_size1': kernel_size1,
                       'kernels2': kernels2, 'kernel_size2': kernel_size2, 'is_fc': is_fc, 'lr':lr, 'batch_size':batch_size,}

        self.input_frames = 5
        self.use_pool = use_pool
        self.use_conv2 = use_conv2

        self.conv1 = nn.Conv2d(channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        conv1_output_size = (7 - kernel_size1 + 2 * padding1) // stride1 + 1

        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride,
                                     padding=0 if pool_kernel_size < conv1_output_size else 1)
            pool1_output_size = (conv1_output_size - pool_kernel_size + 2 * 0 if pool_kernel_size < conv1_output_size else 1) // pool_stride + 1
        else:
            pool1_output_size = conv1_output_size

        if use_conv2:
            self.conv2 = nn.Conv2d(kernels, kernels2, kernel_size=min(pool1_output_size, kernel_size2), padding=padding2)
            conv2_output_size = (pool1_output_size - kernel_size2 + 2 * padding2) // stride1 + 1
            pool2_output_size = (conv2_output_size - pool_kernel_size + 2 * 0 if pool_kernel_size < conv2_output_size else 1) // pool_stride + 1 if use_pool else conv2_output_size
        else:
            pool2_output_size = pool1_output_size

        final_output_size = kernels2 * pool2_output_size * pool2_output_size if use_conv2 else kernels * pool1_output_size * pool1_output_size

        self.rnn = nn.RNN(final_output_size, hidden_size, num_layers, batch_first=True)
        self.fc_main_output = nn.Linear(hidden_size, int(output_len))

    def forward(self, x, oracle_inputs=None):
        conv_outputs = []
        for t in range(self.input_frames):
            x_t = x[:, t, :, :, :]
            x_t = self.pool(F.relu(self.conv1(x_t))) if self.use_pool else F.relu(self.conv1(x_t))
            if self.use_conv2:
                x_t = self.pool(F.relu(self.conv2(x_t))) if self.use_pool else F.relu(self.conv2(x_t))
            flattened = x_t.view(x.size(0), -1)
            conv_outputs.append(flattened)

        x = torch.stack(conv_outputs, dim=1)
        out, _ = self.rnn(x)
        outputs = self.fc_main_output(out[:, -1, :])  # Use last RNN output for the fully connected layer
        return outputs

class CNNModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0, is_fc=False, lr=0.0, batch_size=256,
                 oracle_is_target=False, oracle_early=False):
        super(CNNModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers,
                       'output_len': output_len, 'pool_kernel_size': pool_kernel_size,
                       'pool_stride': pool_stride, 'channels': channels, 'kernels': kernels,
                       'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool, 'stride1': stride1,
                       'use_conv2': use_conv2, 'kernel_size1': kernel_size1,
                       'kernels2': kernels2, 'kernel_size2': kernel_size2, 'is_fc': is_fc, 'lr':lr, 'batch_size':batch_size,}

        self.input_frames = 5  # Assuming a fixed number of input frames as in the original model
        self.use_pool = False
        self.use_conv2 = True

        print(self.kwargs)

        self.conv1 = nn.Conv2d(channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        conv1_output_size = (7 - kernel_size1 + 2 * padding1) // stride1 + 1

        if self.use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            pool1_output_size = ((conv1_output_size - pool_kernel_size) // pool_stride + 1)
        else:
            pool1_output_size = conv1_output_size

        if self.use_conv2:
            self.conv2 = nn.Conv2d(kernels, kernels2, kernel_size=kernel_size2, padding=padding2)
            conv2_output_size = ((pool1_output_size - kernel_size2 + 2 * padding2) // stride1 + 1)
            final_output_size = kernels2 * conv2_output_size * conv2_output_size * 5
        else:
            final_output_size = kernels * pool1_output_size * pool1_output_size * 5
        self.fc = nn.Linear(final_output_size, output_len)

    def forward(self, x, unused):
        conv_outputs = []
        for t in range(self.input_frames):
            x_t = x[:, t, :, :, :]
            x_t = self.pool(F.relu(self.conv1(x_t))) if self.use_pool else F.relu(self.conv1(x_t))
            if self.use_conv2:
                x_t = self.pool(F.relu(self.conv2(x_t))) if self.use_pool else F.relu(self.conv2(x_t))
            flattened = x_t.view(x.size(0), -1)
            conv_outputs.append(flattened)

        final_conv_output = torch.cat(conv_outputs, dim=1)
        outputs = self.fc(final_conv_output)
        return outputs

class RNNModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0, is_fc=False, lr=0.0, batch_size=256,
                 oracle_is_target=False, oracle_early=False):
        super(RNNModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers,
                       'output_len': output_len, 'pool_kernel_size': pool_kernel_size,
                       'pool_stride': pool_stride, 'channels': channels, 'kernels': kernels,
                       'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool, 'stride1': stride1,
                       'use_conv2': use_conv2, 'kernel_size1': kernel_size1,
                       'kernels2': kernels2, 'kernel_size2': kernel_size2, 'oracle_len': oracle_len, 'is_fc': is_fc,
                       'lr': lr, 'batch_size': batch_size, 'oracle_is_target': oracle_is_target, 'oracle_early': oracle_early}

        input_size = 7
        self.input_frames = 5

        self.oracle_early = oracle_early
        self.oracle_is_target = oracle_is_target
        self.is_fc = is_fc
        self.use_pool = use_pool
        self.use_conv2 = use_conv2
        conv1_output_size = (input_size - kernel_size1 + 2 * padding1) // stride1 + 1
        self.conv1 = nn.Conv2d(channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)

        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride,
                                     padding=0 if pool_kernel_size < conv1_output_size else 1)

        pool1_output_size = (conv1_output_size - pool_kernel_size + 2 * 0 if pool_kernel_size < conv1_output_size else 1) // pool_stride + 1 if use_pool else conv1_output_size


        if use_conv2:
            conv2_output_size = (pool1_output_size - min(kernel_size2, pool1_output_size) + 2 * padding2) // stride1 + 1
            self.conv2 = nn.Conv2d(kernels, kernels2, kernel_size=min(pool1_output_size, kernel_size2),
                                   padding=padding2)
            pks = min(pool_kernel_size, conv2_output_size)
            if use_pool:
                self.pool2 = nn.MaxPool2d(kernel_size=pks, stride=pool_stride,
                                          padding=0 if pool_kernel_size < conv2_output_size else 1)
            pool2_output_size = (conv2_output_size - pks + 2 * 0 if pks < conv2_output_size else 1) // pool_stride + 1 if use_pool else conv2_output_size
        else:
            conv2_output_size = conv1_output_size
            pool2_output_size = pool1_output_size

        input_size = kernels2 * pool2_output_size * pool2_output_size
        if self.oracle_early:
            input_size += oracle_len

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_main_output = nn.Linear(hidden_size + oracle_len if not self.oracle_early and not oracle_is_target else hidden_size,
                            int(output_len) if not oracle_is_target else int(output_len))

        if oracle_is_target:
            self.fc_oracle_output = nn.Linear(hidden_size, oracle_len // self.input_frames)

    def forward(self, x, oracle_inputs):
        conv_outputs = []
        for t in range(self.input_frames):
            x_t = x[:, t, :, :, :]

            x_t = self.pool(F.relu(self.conv1(x_t))) if self.use_pool else F.relu(self.conv1(x_t))

            if self.use_conv2:
                x_t = self.pool(F.relu(self.conv2(x_t))) if self.use_pool else F.relu(self.conv2(x_t))

            flattened = x_t.view(x.size(0), -1)
            if self.oracle_early:
                flattened = torch.cat((flattened, oracle_inputs), dim=1)

            conv_outputs.append(flattened)
        x = torch.stack(conv_outputs, dim=1)

        out, _ = self.rnn(x)
        if self.oracle_is_target:
            oracle_outputs = self.fc_oracle_output(out)
            oracle_outputs = oracle_outputs.view(oracle_outputs.size(0), -1)
            newout = out[:, -1, :]
            outputs = torch.cat((self.fc_main_output(newout), oracle_outputs), dim=1)
        elif not self.oracle_early:
            newout = out[:, -1, :]
            outputs = self.fc_main_output(torch.cat((newout, oracle_inputs), dim=1))
        else:
            outputs = self.fc_main_output(out[:, -1, :])
        return outputs


class FeedForwardModel(nn.Module):
    def __init__(self, hidden_size, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0,
                 use_pool=True, use_conv2=False, oracle_len=0, num_layers=1, oracle_layer=0):
        super(FeedForwardModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers,
                       'output_len': output_len, 'pool_kernel_size': pool_kernel_size,
                       'pool_stride': pool_stride, 'channels': channels, 'kernels': kernels,
                       'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool, 'stride1': stride1,
                       'use_conv2': use_conv2, 'kernel_size1': kernel_size1,
                       'kernels2': kernels2, 'kernel_size2': kernel_size2, 'oracle_len': oracle_len}

        self.use_pool = use_pool
        self.use_conv2 = use_conv2
        self.hidden_size = hidden_size
        self.oracle_layer = oracle_layer
        input_size = 7

        # Since we're stacking all the time steps as extra channels:
        # Each time step will add 'channels' to the input.
        time_steps = 9
        self.channels = channels * time_steps
        print('init ff', )

        self.conv1 = nn.Conv2d(self.channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        conv1_output_size = (input_size - kernel_size1 + 2 * padding1) // stride1 + 1
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride,
                                     padding=0 if pool_kernel_size < conv1_output_size else 1)
        # self.output_len = output_len

        pool1_output_size = (conv1_output_size - pool_kernel_size + 2 * 0 if pool_kernel_size < conv1_output_size else 1) // pool_stride + 1 if use_pool else conv1_output_size

        if use_conv2:
            conv2_output_size = (pool1_output_size - min(kernel_size2, pool1_output_size) + 2 * padding2) // stride1 + 1
            self.conv2 = nn.Conv2d(kernels, kernels2, kernel_size=min(pool1_output_size, kernel_size2),
                                   padding=padding2)
            pks = min(pool_kernel_size, conv2_output_size)
            if use_pool:
                self.pool2 = nn.MaxPool2d(kernel_size=pks, stride=pool_stride,
                                          padding=0 if pool_kernel_size < conv2_output_size else 1)
            pool2_output_size = (
                                    conv2_output_size - pks + 2 * 0 if pks < conv2_output_size else 1) // pool_stride + 1 if use_pool else conv2_output_size
        else:
            conv2_output_size = conv1_output_size
            pool2_output_size = pool1_output_size

        flat_size = kernels2 * pool2_output_size * pool2_output_size
        if self.oracle_layer != 0:
            flat_size += oracle_len
        self.fc = nn.Linear(flat_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size + oracle_len if self.oracle_layer == 0 else hidden_size, output_len)

    def forward(self, x, oracle_inputs):
        # Stack the time steps across the channel dimension
        x = x.view(x.size(0), self.channels, x.size(3), x.size(4))

        # Conv and pooling operations
        x = F.relu(self.conv1(x))
        if self.use_pool:
            x = self.pool(x)
        if self.use_conv2:
            x = F.relu(self.conv2(x))
            if self.use_pool:
                x = self.pool(x)

        # Flatten and pass through the FC layers
        x = x.view(x.size(0), -1)
        if self.oracle_layer == 0:
            x = F.relu(self.fc(x))
            x = self.output_layer(torch.cat((x, oracle_inputs), dim=-1))
        else:
            x = F.relu(self.fc(torch.cat((x, oracle_inputs), dim=-1)))
            x = self.output_layer(x)

        return x

class fCNN(nn.Module):
    # this module folds timesteps into channels, as opposed to running conv kernels on each individually
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0, is_fc=False, lr=0.0, batch_size=256,
                 oracle_is_target=False, oracle_early=False):
        super(fCNN, self).__init__()
        self.kwargs = {
            'hidden_size': hidden_size, 'num_layers': num_layers, 'output_len': output_len,
            'channels': channels, 'kernels': kernels, 'kernels2': kernels2, 'kernel_size1': kernel_size1,
            'kernel_size2': kernel_size2, 'stride1': stride1, 'pool_kernel_size': pool_kernel_size,
            'pool_stride': pool_stride, 'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool,
            'use_conv2': use_conv2, 'oracle_len': oracle_len, 'is_fc': is_fc, 'lr': lr, 'batch_size': batch_size,
            'oracle_is_target': oracle_is_target, 'oracle_early': oracle_early
        }

        self.conv1 = nn.Conv2d(channels * 5, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.fc = nn.Linear(kernels * 7 * 7, output_len)  # Adjust the sizing based on input dimensions post-pooling

    def forward(self, x, unused):
        batch_size, timesteps, channels, height, width = x.size()
        x = x.view(batch_size, timesteps * channels, height, width)
        x = self.conv1(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class tCNN(nn.Module):
    # this module folds timesteps into channels, as opposed to running conv kernels on each individually
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0, is_fc=False, lr=0.0, batch_size=256,
                 oracle_is_target=False, oracle_early=False):
        super(tCNN, self).__init__()
        self.kwargs = {
            'hidden_size': hidden_size, 'num_layers': num_layers, 'output_len': output_len,
            'channels': channels, 'kernels': kernels, 'kernels2': kernels2, 'kernel_size1': kernel_size1,
            'kernel_size2': kernel_size2, 'stride1': stride1, 'pool_kernel_size': pool_kernel_size,
            'pool_stride': pool_stride, 'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool,
            'use_conv2': use_conv2, 'oracle_len': oracle_len, 'is_fc': is_fc, 'lr': lr, 'batch_size': batch_size,
            'oracle_is_target': oracle_is_target, 'oracle_early': oracle_early
        }

        self.conv1 = nn.Conv2d(channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.fc = nn.Linear(kernels * 7 * 7 * 5, output_len)  # Adjust the sizing based on input dimensions post-pooling

    def forward(self, x, unused):
        outputs = []
        batch_size, timesteps, channels, height, width = x.size()
        for t in range(timesteps):
            x_t = x[:, t, :, :, :]
            x_t = self.conv1(x_t)
            x_t = F.relu(x_t)
            x_t = x_t.view(x_t.size(0), -1)
            outputs.append(x_t)

        outputs = torch.cat(outputs, dim=1)
        x = self.fc(outputs)
        return x

class sRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0, is_fc=False, lr=0.0, batch_size=0.0,
                 oracle_is_target=False, oracle_early=False):
        super(sRNN, self).__init__()
        self.kwargs = {
            'hidden_size': hidden_size, 'num_layers': num_layers, 'output_len': output_len,
            'channels': channels, 'kernels': kernels, 'kernels2': kernels2, 'kernel_size1': kernel_size1,
            'kernel_size2': kernel_size2, 'stride1': stride1, 'pool_kernel_size': pool_kernel_size,
            'pool_stride': pool_stride, 'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool,
            'use_conv2': use_conv2, 'oracle_len': oracle_len, 'is_fc': is_fc, 'lr': lr, 'batch_size': batch_size,
            'oracle_is_target': oracle_is_target, 'oracle_early': oracle_early
        }

        self.conv1 = nn.Conv2d(channels, kernels, kernel_size1, stride=stride1, padding=padding1)
        if use_pool:
            self.pool = nn.MaxPool2d(pool_kernel_size, stride=pool_stride)
        conv_output_size = (7 + 2 * padding1 - kernel_size1) // stride1 + 1
        if use_pool:
            conv_output_size = (conv_output_size - pool_kernel_size) // pool_stride + 1
        conv_output_size = (kernels, conv_output_size, conv_output_size)
        rnn_input_size = conv_output_size[0] * conv_output_size[1] * conv_output_size[2]

        self.rnn = nn.RNN(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x, unused):
        batch_size, timesteps, channels, height, width = x.shape
        c_out = x.view(batch_size * timesteps, channels, height, width)
        c_out = F.relu(self.conv1(c_out))
        if hasattr(self, 'pool'):
            c_out = self.pool(c_out)
        c_out = c_out.view(batch_size, timesteps, -1)
        out, _ = self.rnn(c_out)
        out = self.fc(out[:, -1, :])
        return out


class sMLP(nn.Module):
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0, is_fc=False, lr=0.0, batch_size=0.0,
                 oracle_is_target=False, oracle_early=False):
        super(sMLP, self).__init__()
        self.kwargs = {
            'hidden_size': hidden_size, 'num_layers': num_layers, 'output_len': output_len,
            'channels': channels, 'kernels': kernels, 'kernels2': kernels2, 'kernel_size1': kernel_size1,
            'kernel_size2': kernel_size2, 'stride1': stride1, 'pool_kernel_size': pool_kernel_size,
            'pool_stride': pool_stride, 'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool,
            'use_conv2': use_conv2, 'oracle_len': oracle_len, 'is_fc': is_fc, 'lr': lr, 'batch_size': batch_size,
            'oracle_is_target': oracle_is_target, 'oracle_early': oracle_early
        }

        input_size = 7 * 7 * channels * 5
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_len)

    def forward(self, x, unused):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
