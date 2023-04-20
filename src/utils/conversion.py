import random

# from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import supersuit as ss
from pettingzoo.utils import wrappers
from ..agents import GridAgentInterface
from ..pz_envs import env_from_config
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, VecVideoRecorder, VecTransposeImage, \
    VecNormalize
# from ..utils.vec_normalize import VecNormalizeMultiAgent
from ..pz_envs.scenario_configs import ScenarioConfigs
import os
import gym
import json
from stable_baselines3 import TD3, PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO

class_dict = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO, 'RecurrentPPO': RecurrentPPO}


def get_json_params(path):
    with open(path) as json_data:
        data = json.load(json_data)
        model_class = data['model_class']
        if model_class in class_dict.keys():
            model_class = class_dict[model_class]
        size = data['size']
        style = data['style']
        frames = data['frames']
        threads = data['threads']
        difficulty = data['difficulty']
        configName = data['configName']
        vecNormalize = data['vecNormalize'] if 'vecNormalize' in data.keys() else True
    return model_class, size, style, frames, vecNormalize, difficulty, threads, configName


def make_env_comp(env_name, frames=1, vecNormalize=False, size=32, style='rich', monitor_path='dir',
                  rank=-1, threads=1, load_path=None):
    env = gym.make(env_name)

    env = wrap_env_full(env.env, memory=frames, size=size,
                        vecNormalize=vecNormalize,
                        style=style, monitor_path=monitor_path, rank=rank, channels=env.channels, threads=threads,
                        load_path=load_path)
    if rank == 0:
        print('monitor path:', os.path.join(monitor_path, f'{env_name}-{rank}'))
        env = VecMonitor(env, os.path.join(monitor_path,
                                           f'{env_name}-{rank}'))  # should get info keywords here for train monitoring, eg accuracy
    env.rank = rank
    return env


def wrap_env_full(env, reduce_color=False, memory=1, size=32, vecMonitor=False,
                  configName=None, threads=1, rank=0, vecNormalize=False, style='rich', monitor_path=None,
                  channels=5, num_cpus=1, load_path=None):
    num_cpus = threads  # silly line
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
    if memory > 1:
        env = VecFrameStack(env, n_stack=memory, channels_order='first')

    if vecMonitor:
        if monitor_path != "":
            env = VecMonitor(env, filename=os.path.join(monitor_path, f"{configName}-{rank}"),
                             info_keywords=env.info_keywords)
        else:
            env = VecMonitor(env, filename=f"{configName}-{rank}", info_keywords=env.info_keywords)
    if load_path:
        env = VecNormalize.load(load_path, env)
        print('loaded vec normalize', load_path)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=vecNormalize, norm_reward=False, clip_obs=100., clip_reward=100., training=rank == 0)

    return env
