
import pickle

import h5py
import sys
import os

import pandas as pd

from .utils.evaluation import get_relative_direction

sys.path.append(os.getcwd())

from .objects import *
from .agents import GridAgentInterface
from .pz_envs import env_from_config
from .pz_envs.scenario_configs import ScenarioConfigs
# import src.pz_envs
from torch.utils.data import Dataset
import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch


def one_hot(size, data):
    return np.eye(size)[data]


def gen_data(labels=[], path='supervised', pref_type='', role_type='', record_extra_data=False, prior_metrics=[]):
    # labels = ['loc', 'exist', 'vision', 'b-loc', 'b-exist', 'target']
    posterior_metrics = ['selection', 'selectedBig', 'selectedSmall', 'selectedNeither',
                         'selectedPrevBig', 'selectedPrevSmall', 'selectedPrevNeither',
                         'selectedSame', ]

    env_config = {
        "env_class": "MiniStandoffEnv",
        "max_steps": 25,
        "respawn": True,
        "ghost_mode": False,
        "reward_decay": False,
        "width": 9,
        "height": 9,
    }

    player_interface_config = {
        "view_size": 7,
        "view_offset": 0,
        "view_tile_size": 1,
        "observation_style": "rich",
        "see_through_walls": False,
        "color": "yellow",
        "view_type": 0,
        "move_type": 0
    }
    puppet_interface_config = {
        "view_size": 5,
        "view_offset": 3,
        "view_tile_size": 48,
        "observation_style": "rich",
        "see_through_walls": False,
        "color": "red",
        # "move_type": 1,
        # "view_type": 1,
    }

    frames = 9
    all_path_infos = pd.DataFrame()
    suffix = pref_type + role_type


    for configName in ScenarioConfigs.stages:
        configs = ScenarioConfigs().standoff
        events = ScenarioConfigs.stages[configName]['events']
        params = configs[ScenarioConfigs.stages[configName]['params']]

        _subject_is_dominant = [False] if role_type == '' else [True] if role_type == 'D' else [True, False]
        _subject_valence = [1] if pref_type == '' else [2] if pref_type == 'd' else [1, 2]

        data_name = f'{configName}'
        informedness = data_name[3:-1]
        print('data name', data_name)
        data_obs = []
        data_labels = {}
        all_labels = list(set(labels + prior_metrics + posterior_metrics))
        for label in all_labels:
            data_labels[label] = []
        data_params = []

        for subject_is_dominant in _subject_is_dominant:
            for subject_valence in _subject_valence:
                params['subject_is_dominant'] = subject_is_dominant
                params['sub_valence'] = subject_valence

                if isinstance(params["num_agents"], list):
                    params["num_agents"] = params["num_agents"][0]
                if isinstance(params["num_puppets"], list):
                    params["num_puppets"] = params["num_puppets"][0]

                env_config['config_name'] = configName
                env_config['agents'] = [GridAgentInterface(**player_interface_config) for _ in
                                        range(params['num_agents'])]
                env_config['puppets'] = [GridAgentInterface(**puppet_interface_config) for _ in
                                         range(params['num_puppets'])]

                difficulty = 3
                env_config['opponent_visible_decs'] = (difficulty < 1)
                env_config['persistent_treat_images'] = (difficulty < 2)
                env_config['subject_visible_decs'] = (difficulty < 3)
                env_config['gaze_highlighting'] = (difficulty < 3)
                env_config['persistent_gaze_highlighting'] = (difficulty < 2)

                env = env_from_config(env_config)
                env.record_oracle_labels = True
                env.record_info = True  # used for correctSelection right now

                env.target_param_group_count = 20
                env.param_groups = [ {'eLists': {n: events[n]},
                                      'params': params,
                                      'perms': {n: ScenarioConfigs.all_event_permutations[n]},
                                      'delays': {n: ScenarioConfigs.all_event_delays[n]}
                                      }
                                     for n in events ]
                print('first param group', env.param_groups[0])

                prev_param_group = -1
                tq = tqdm.tqdm(range(len(env.param_groups)))

                total_groups = len(env.param_groups)

                env.deterministic = True
                print('total_groups', total_groups)

                while True:
                    env.deterministic_seed = env.current_param_group_pos

                    obs = env.reset()
                    if env.current_param_group != prev_param_group:
                        eName = env.current_event_list_name
                        tq.update(1)
                    prev_param_group = env.current_param_group
                    this_ob = np.zeros((frames, *obs['p_0'].shape))
                    pos = 0

                    while pos < frames:
                        next_obs, _, _, info = env.step({'p_0': 2})
                        this_ob[pos, :, :, :] = next_obs['p_0']
                        if pos == frames - 1 or env.has_released:
                            data_obs.append(serialize_data(this_ob.astype(np.uint8)))
                            data_params.append(eName)
                            for label in [x for x in all_labels if x not in posterior_metrics]:
                                if label == "correctSelection" or label == 'incorrectSelection':
                                    data = one_hot(5, info['p_0'][label])
                                elif label == "informedness":
                                    data = informedness
                                elif label == "opponents":
                                    data = params["num_puppets"]
                                else:
                                    data = info['p_0'][label]
                                data_labels[label].append(copy.copy(data))
                            break

                        pos += 1

                    if record_extra_data:
                        all_paths = env.get_all_paths(env.grid.volatile, env.instance_from_name['p_0'].pos)

                        for k, path in enumerate(all_paths):

                            _env = copy.deepcopy(env)
                            a = _env.instance_from_name['p_0']
                            while True:
                                _, _, done, info = _env.step({'p_0': get_relative_direction(a, path)})
                                if done['p_0']:
                                    all_path_infos = all_path_infos.append(info['p_0'], ignore_index=True)
                                    break
                        del _env, a

                    if env.current_param_group == total_groups - 1 and env.current_param_group_pos == env.target_param_group_count - 1:
                        # normally the while loop won't break because reset uses a modulus
                        break

        print('len obs', data_name, params["num_puppets"], len(data_obs))
        this_path = os.path.join(path, data_name)
        os.makedirs(this_path, exist_ok=True)

        np.savez_compressed(os.path.join(this_path, 'obs'), np.array(data_obs))
        np.savez_compressed(os.path.join(this_path,  'params'), np.array(data_params))
        for label in all_labels:
            if len(data_labels[label]) > 0:
                np.savez_compressed(os.path.join(this_path, 'label-' + label), np.array(data_labels[label]))

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
class CustomDatasetBig(Dataset):
    def __init__(self, data_list, labels_list, params_list, oracles_list, metrics=None):
        self.data_list = data_list
        self.labels_list = labels_list
        self.params_list = params_list
        self.oracles_list = oracles_list
        self.metrics = metrics

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
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):

        list_index, local_index = self._find_list_index(index)

        data = torch.from_numpy(pickle.loads(self.data_list[list_index][local_index])).float().to(self.device)
        labels = torch.from_numpy(self.labels_list[list_index][local_index].astype(np.int8)).to(self.device)
        oracles = torch.from_numpy(self.oracles_list[list_index][local_index].astype(np.int8)).to(self.device) if len(self.oracles_list) > 1 else torch.tensor([]).to(self.device)

        metrics = {key: self.metrics[key][index] for key in self.metrics.keys()} if self.metrics else 0

        return data, labels, self.params_list[list_index][local_index], oracles, metrics

class RNNModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_len, channels, kernels=8, kernels2=8, kernel_size1=3,
                 kernel_size2=3, stride1=1, pool_kernel_size=2, pool_stride=2, padding1=0, padding2=0, use_pool=True,
                 use_conv2=False, oracle_len=0):
        super(RNNModel, self).__init__()
        self.kwargs = {'hidden_size': hidden_size, 'num_layers': num_layers,
                       'output_len': output_len, 'pool_kernel_size': pool_kernel_size,
                       'pool_stride': pool_stride, 'channels': channels, 'kernels': kernels,
                       'padding1': padding1, 'padding2': padding2, 'use_pool': use_pool, 'stride1': stride1,
                       'use_conv2': use_conv2, 'kernel_size1': kernel_size1,
                       'kernels2': kernels2, 'kernel_size2': kernel_size2, 'oracle_len': oracle_len}

        input_size = 7

        self.use_pool = use_pool
        self.use_conv2 = use_conv2
        conv1_output_size = (input_size - kernel_size1 + 2 * padding1) // stride1 + 1
        self.conv1 = nn.Conv2d(channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)

        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride,
                                     padding=0 if pool_kernel_size < conv1_output_size else 1)
        # self.output_len = output_len

        pool1_output_size = (
                                conv1_output_size - pool_kernel_size + 2 * 0 if pool_kernel_size < conv1_output_size else 1) // pool_stride + 1 if use_pool else conv1_output_size


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

        # input_size = 16 * pool2_output_size * pool2_output_size
        input_size = kernels2 * pool2_output_size * pool2_output_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size + oracle_len, int(output_len))

    def forward(self, x, oracle_inputs):
        conv_outputs = []
        for t in range(9):
            x_t = x[:, t, :, :, :]

            x_t = self.pool(F.relu(self.conv1(x_t))) if self.use_pool else F.relu(self.conv1(x_t))

            if self.use_conv2:
                x_t = self.pool(F.relu(self.conv2(x_t))) if self.use_pool else F.relu(self.conv2(x_t))
            conv_outputs.append(x_t.view(x.size(0), -1))
        x = torch.stack(conv_outputs, dim=1)

        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Use only the last timestep's output
        outputs = self.fc(torch.cat((out, oracle_inputs), dim=-1))
        return outputs
