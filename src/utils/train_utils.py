import os

import pandas as pd
import torch as th
import shutil

from sb3_contrib import TRPO, RecurrentPPO
from ..models.custom_cnn import CustomCNN
from typing import Callable


def init_policy(model_class, width, hidden_size, conv_mult=1, frames=1, net_arch=None, name=''):
    if model_class == RecurrentPPO:
        policy = 'MultiInputLstmPolicy'
        policy_kwargs = {'features_extractor_class': CustomCNN,
                            'net_arch': net_arch,
                            'lstm_hidden_size': hidden_size,
                            'features_extractor_kwargs':
                                {'features_dim': width, 'conv_mult': conv_mult, 'frames': 1},
                        }
    else:
        policy = 'MlpPolicy'
        if name == 'mlp':
            policy_kwargs = {'activation_fn': th.nn.ReLU, 'net_arch': [width, dict(pi=[width], vf=[width])]}
        else:
            policy_kwargs = {'features_extractor_class': CustomCNN,
                                'activation_fn': th.nn.ReLU,
                                'net_arch': net_arch,
                                'features_extractor_kwargs':
                                    {'features_dim': width, 'conv_mult': conv_mult, 'frames': frames},
                            }
    return policy, policy_kwargs



def init_dirs(savePath, experimentName, continuing=False, overwrite=False):
    """
    :param savePath: path to save the experiment
    :param experimentName: name of the experiment
    :param continuing: if continuing an experiment
    """
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    global_log_path = os.path.join(savePath, 'globalLogs')
    if not os.path.exists(global_log_path):
        os.mkdir(global_log_path)
    global_log_path = os.path.join(savePath, 'globalLogs', "global_log.csv")
    try:
        global_logs = pd.read_csv(global_log_path, index_col=None)
    except:
        print('no global log, creating new one...')
        global_logs = pd.DataFrame(
            columns=['name', 'timesteps', 'long_name', 'train_conf', 'agent_type', 'agent_policy'])
    count = global_logs[global_logs.name == experimentName].shape[0]
    name = experimentName + str(count) if not continuing and not overwrite else experimentName
    savePath2 = os.path.join(savePath, name)
    continuing_ret = False
    
    if not overwrite:
        if not os.path.exists(savePath2):
            if continuing:
                print('savepath does not exist at', savePath2)
            else:
                print('creating savepath at', savePath2)
            os.mkdir(savePath2)
        else:
            if continuing:
                print('continuing training at', savePath2)
                continuing_ret = True
            else:
                print('savepath already exists at', savePath2)
    else:
        if os.path.exists(savePath2):
            shutil.rmtree(savePath2)
        os.mkdir(savePath2)
    return global_log_path, global_logs, savePath2, name, continuing_ret


def start_global_logs(global_logs, short_name, name, configName, model_class, policy, global_log_path):
    log_line = len(global_logs)
    global_logs.loc[log_line + 1] = {'name': short_name,
                                     'timesteps': 0,
                                     'long_name': name,
                                     'train_conf': configName,
                                     'agent_type': str(model_class),
                                     'agent_policy': str(policy)}
    global_logs.to_csv(global_log_path, index=False)
    return log_line
    
def load_last_checkpoint_model(path, model_class):
    full_path = os.path.join(path, 'checkpoints')
    best_model = None
    best_length = 0
    paths = os.scandir(full_path)
    rep_folders = [pathx for pathx in paths if pathx.is_dir() and pathx.name.startswith('rep_')]
    if rep_folders:
        for rep_folder in rep_folders:
            for checkpoint_path in os.scandir(rep_folder.path):
                if int(checkpoint_path.path[
                       checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")]) > best_length:
                    best_length = int(
                        checkpoint_path.path[
                        checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")])
                    best_model = model_class.load(checkpoint_path.path)
    else:
        for checkpoint_path in os.scandir(full_path):
            if int(checkpoint_path.path[
                   checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")]) > best_length:
                best_length = int(
                    checkpoint_path.path[checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")])
                best_model = model_class.load(checkpoint_path.path)

    return best_model, best_length


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
