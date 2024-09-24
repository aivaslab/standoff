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



def gen_data(labels=[], path='supervised', pref_type='', role_type='', record_extra_data=False, prior_metrics=[], conf=None):
    '''
    For all relevant variants, iterates through all possible permutations of environment reset configs and simulates
    up until the release event.
    Records and saves observations, as well as data including labels and metrics.
    '''
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

    frames = 4
    all_path_infos = pd.DataFrame()
    suffix = pref_type + role_type

    onehot_labels = ['correct-loc', 'incorrect-loc']
    extra_labels = ['opponents', 'last-vision-span', 'id'] #we get these manually outside the env

    tq = tqdm.tqdm(range(sum(len(conf.stages[cc]['events']) for cc in conf.stages)))
    # tqdm not currently working with subject dominant and subject valence lists

    unique_id = 0 # this is the id of each datapoint

    for configName in conf.stages:
        configs = conf.standoff
        events = conf.stages[configName]['events']
        params = configs[conf.stages[configName]['params']]

        _subject_is_dominant = [False] if role_type == '' else [True] if role_type == 'D' else [True, False]
        _subject_valence = [1] if pref_type == '' else [2] if pref_type == 'd' else [1, 2]

        data_name = f'{configName}'
        informedness = data_name[3:-1]
        mapping = {'T': 2, 'F': 1, 'N': 0, 't': 2, 'f': 1, 'n': 0, '0': 0, '1': 1}
        # we just get opponents directly from num_puppets later
        informedness = [mapping[char] for char in informedness]

        #print('data name', data_name)
        data_obs = []
        data_labels = {}
        all_labels = list(set(labels + prior_metrics + list(posterior_metrics) + extra_labels))
        for label in all_labels:
            data_labels[label] = []
        data_params = []
        posterior_metrics = set(posterior_metrics)

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
                env_config['conf'] = conf

                env = env_from_config(env_config)
                env.record_oracle_labels = True
                env.record_info = True  # used for correct-loc right now

                env.target_param_group_count = 20
                env.param_groups = [ {'eLists': {n: events[n]},
                                      'params': params,
                                      'perms': {n: conf.all_event_permutations[n]},
                                      'delays': {n: conf.all_event_delays[n]}
                                      }
                                     for n in events ]
                #print('first param group', env.param_groups[0])

                prev_param_group = -1

                total_groups = len(env.param_groups)

                env.deterministic = True
                #print('total_groups', total_groups)
                check_labels = [x for x in all_labels if x not in posterior_metrics and x not in extra_labels and x not in onehot_labels]

                counts = {'total': 0, 'big': 0, 'small': 0}

                while True:
                    env.deterministic_seed = env.current_param_group_pos

                    obs = env.reset()
                    if env.current_param_group != prev_param_group:
                        eName = env.current_event_list_name
                        tq.update(1)
                    prev_param_group = env.current_param_group
                    this_ob = np.zeros((1 + frames, *obs['p_0'].shape))
                    pos = 0


                    temp_labels = {label: [] for label in check_labels}
                    one_labels = {label: [] for label in onehot_labels}

                    while pos <= frames:
                        this_ob[pos, :, :, :] = obs['p_0']
                        obs, _, _, info = env.step({'p_0': 2})

                        for label in check_labels:
                            temp_labels[label].append(info['p_0'][label])
                        for label in onehot_labels:
                            data = one_hot(5, info['p_0'][label])
                            one_labels[label].append(data)


                        if pos == frames or env.has_released:
                            data_obs.append(serialize_data(this_ob.astype(np.uint8)))
                            data_params.append(eName)
                            for label in check_labels:
                                data_labels[label].append(np.stack(temp_labels[label]))
                            for label in onehot_labels:
                                data_labels[label].append(np.stack(one_labels[label]))

                            #data_labels['informedness'].append(informedness)
                            #if informedness != info['p_0']['informedness'] and params["num_puppets"] > 0:
                            #    print('true inf:', informedness, 'step inf:', info['p_0']['informedness'], info['p_0']['loc'], info['p_0']['b-loc'], eName, 'v',
                            #          temp_labels['vision'], 'bait', temp_labels['bait-treat'], 'swap', temp_labels['swap-treat'])
                            data_labels['opponents'].append(params["num_puppets"])
                            data_labels['id'].append(unique_id)
                            unique_id += 1
                            #print(informedness, params["num_puppets"], info['p_0']['shouldGetBig'], info['p_0']['shouldGetSmall'])

                            identify_mismatches(info, env, informedness, params, data_name, configName, eName, info['p_0']['loc'], info['p_0']['b-loc'], counts)

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

                print('regime', data_name, 'counts', counts)

        #print('len obs', data_name, params["num_puppets"], len(data_obs))
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


class BaseDatasetBig(Dataset):
    def __init__(self, data_list, labels_list, params_list, oracles_list, included_indices, metrics=None):
        self.data_list = data_list
        self.labels_list = labels_list
        self.params_list = params_list
        self.oracles_list = oracles_list
        self.metrics = metrics
        self.included_indices = included_indices

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
    labels = default_collate(labels)
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
        data = torch.from_numpy(pickle.loads(self.data_list[list_index][local_index])).float()
        labels = torch.from_numpy(self.labels_list[list_index][local_index].astype(np.int8))
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
        self.use_pool = use_pool
        self.use_conv2 = use_conv2

        self.conv1 = nn.Conv2d(channels, kernels, kernel_size=kernel_size1, padding=padding1, stride=stride1)
        conv1_output_size = (7 - kernel_size1 + 2 * padding1) // stride1 + 1

        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            pool1_output_size = ((conv1_output_size - pool_kernel_size) // pool_stride + 1)
        else:
            pool1_output_size = conv1_output_size

        if use_conv2:
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
