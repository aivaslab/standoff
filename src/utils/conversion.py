
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, VecVideoRecorder, VecTransposeImage, \
    VecNormalize
import os
import gym
import json
from stable_baselines3 import TD3, PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
from argparse import Namespace

class_dict = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO, 'RecurrentPPO': RecurrentPPO}


def get_json_params(path, args):
    if args is None:
        args = Namespace()
    with open(path) as json_data:
        data = json.load(json_data)
        model_class = data['model_class']
        if model_class in class_dict.keys():
            model_class = class_dict[model_class]
        args.size = data['size']
        args.style = data['style']
        args.frames = data['frames']
        args.threads = data['threads']
        args.difficulty = data['difficulty']
        configName = data['configName']
        args.vecNormalize = data['vecNormalize'] if 'vecNormalize' in data.keys() else True
        args.norm_rewards = data['norm_rewards'] if 'norm_rewards' in data.keys() else True
        args.normalize_images = data['normalize_images'] if 'normalize_images' in data.keys() else True
        args.shared_lstm = data['shared_lstm'] if 'shared_lstm' in data.keys() else False
        args.use_supervised_models = data['use_supervised_models'] if 'use_supervised_models' in data.keys() else False
        args.supervised_data_source = data['supervised_data_source'] if 'supervised_data_source' in data.keys() else 'random-2500'
        args.supervised_model_label = data['supervised_model_label'] if 'supervised_model_label' in data.keys() else 'loc'
        args.supervised_model_path = data['supervised_model_path'] if 'supervised_model_path' in data.keys() else 'supervised'
        args.ground_truth_modules = data['ground_truth_modules'] if 'ground_truth_modules' in data.keys() else False
    return model_class, configName, args

def calculate_informedness(loc, b_loc):
    informedness_result = []

    for item in [[1, 0], [0, 1]]:  # For 'big' and 'small'

        if item in loc and item in b_loc:
            if loc.index(item) == b_loc.index(item):
                informedness_result += [2] if item == [1, 0] else [2]
            else:
                informedness_result += [1] if item == [1, 0] else [1]
        else:
            informedness_result += [0] if item == [1, 0] else [0]

    return informedness_result

def make_env_comp(env_name, frames=1, vecNormalize=False, norm_rewards=False, size=32, style='rich', monitor_path='dir',
                  rank=-1, threads=1, load_path=None, reduce_color=False, skip_vecNorm=False, sl_module=None, label_dim=0):
    env = gym.make(env_name)
    #print(env.__dict__.keys()) #yields env, _action_space, _observation_space, _reward_range, _metadata, _has_reset
    env = env.env #this line feels silly?
    if sl_module is not None:
        env.supervised_model = sl_module
        env.label_dim = label_dim
    env.record_info = rank != 0
    channels = env.channels
    #num_cpus = min(threads, os.cpu_count())
    num_cpus = threads

    if reduce_color:
        env = ss.color_reduction_v0(env, 'B')
    if style != 'rich' and False:
        env = ss.resize_v0(env, x_size=size, y_size=size)
    if reduce_color:
        env = ss.reshape_v0(env, (size, size, 1))
    env = ss.reshape_v0(env, (channels, size, size))
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, threads, num_cpus=num_cpus, base_class='stable_baselines3')

    # env = VecTransposeImage(env) # todo: double check evals are correct for rich obs
    if frames > 1:
        env = VecFrameStack(env, n_stack=frames, channels_order='first')

    if rank == 0:
        print('monitor path:', os.path.join(monitor_path, f'{env_name}-{rank}'))
        env = VecMonitor(env, os.path.join(monitor_path,
                                           f'{env_name}-{rank}'))  # should get info keywords here for train monitoring, eg accuracy
    if not skip_vecNorm:
        if load_path:
            env = VecNormalize.load(load_path, env)
            env.training = False
            env.norm_reward = norm_rewards
        else:
            env = VecNormalize(env, norm_obs=vecNormalize, norm_reward=norm_rewards, clip_obs=1000., clip_reward=1000.,
                               training=rank == 0)

    env.rank = rank
    return env
