import random

# from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import supersuit as ss
from pettingzoo.utils import wrappers
from ..agents import GridAgentInterface
from ..pz_envs import env_from_config
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, VecVideoRecorder, VecTransposeImage
from ..utils.vec_normalize import VecNormalizeMultiAgent
from ..pz_envs.scenario_configs import ScenarioConfigs
import os
import gym

'''def group_dataframe(cur_df, groupby_list):
    ret_df = cur_df.groupby(groupby_list, as_index=False).agg(agg_dict(cur_df))
    # cols are tuples either [(name, mean) (name, std)] or (name, first)
    grouped_df.columns = grouped_df.columns.map("_".join).str.replace('_first', '')
    grouped_df = grouped_df.sort_values('model_ep')
    return grouped_df

    grouped_df = group_dataframe(df, ['model_ep', 'configName'])
    grouped_df_noname = df_small.groupby(['model_ep'], as_index=False).agg(agg_dict(df_small)).sort_values('model_ep')
    grouped_df_noname.columns = (grouped_df_noname.columns - ignored_columns).map("_".join)
    grouped_df_small = df_small.groupby(['model_ep', 'configName'], as_index=False).agg(agg_dict(df_small)).sort_values('model_ep')
    grouped_df_small.columns = (grouped_df_small.columns - ignored_columns).map("_".join)'''


def make_env_comp(env_name, frames=1, vecNormalize=False, size=32, style='rich', monitor_path='dir',
                  rank=-1):
    # if model_class != RecurrentPPO and frames > 1:
    #    env = gym.make(env_name)
    #    env = wrap_env_full(env.env, memory=frames, size=size, saveVids=saveVids,
    #                        path=video_path, vecNormalize=vecNormalize, recordEvery=1, style='rich')
    # else:
    env = gym.make(env_name)

    env = wrap_env_full(env.env, memory=frames, size=size,
                        vecNormalize=vecNormalize,
                        style=style, monitor_path=monitor_path, rank=rank)
    print('monitor path:', os.path.join(monitor_path, f'{env_name}-{rank}'))
    env = VecMonitor(env, os.path.join(monitor_path, f'{env_name}-{rank}')) # should get info keywords here for train monitoring, eg accuracy
    env.rank = rank
    return env


def make_env(envClass, player_config, configName=None, memory=1, threads=1, reduce_color=False, size=64,
             reward_decay=False, ghost_mode=True, max_steps=50, saveVids=False, path="", recordEvery=1e4,
             vecMonitor=True,
             rank=0, vecNormalize=False):
    env_config = {
        "env_class": envClass,
        "max_steps": max_steps,
        "respawn": True,
        "ghost_mode": ghost_mode,
        "reward_decay": reward_decay,
        "width": 9 if envClass == "para_TutorialEnv" else 19,
        "height": 9 if envClass == "para_TutorialEnv" else 19,
        "memory": memory,
        "step_reward": -0.1,
        "configName": configName,
    }
    configs = ScenarioConfigs().standoff
    reset_configs = {**configs["defaults"], **configs[configName]}

    if isinstance(reset_configs["num_agents"], list):
        reset_configs["num_agents"] = reset_configs["num_agents"][0]
    if isinstance(reset_configs["num_puppets"], list):
        reset_configs["num_puppets"] = reset_configs["num_puppets"][0]

    env_config['agents'] = [GridAgentInterface(**player_config) for _ in range(reset_configs['num_agents'])]
    env_config['puppets'] = [GridAgentInterface(**player_config) for _ in range(reset_configs['num_puppets'])]
    # env_config['num_agents'] = reset_configs['num_agents']
    # env_config['num_puppets'] = reset_configs['num_puppets']
    env_config['config_name'] = configName

    env = env_from_config(env_config)
    env.agent_view_size = player_config["view_size"] * player_config["view_tile_size"]

    configName = random.choice(list(env.configs.keys())) if configName is None else configName
    env.hard_reset(env.configs[configName])
    info_keywords = env.info_keywords

    if reduce_color:
        env = ss.color_reduction_v0(env, 'B')
    env = ss.resize_v0(env, x_size=size, y_size=size)
    if reduce_color:
        env = ss.reshape_v0(env, (size, size, 1))
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, threads, num_cpus=1, base_class='stable_baselines3')
    # num_cpus=1 changed from 2 to avoid csv issues. does it affect speed?
    env = VecTransposeImage(env)
    if memory > 1:
        env = VecFrameStack(env, n_stack=memory, channels_order='first')
        # consider StackedObservations

    if vecMonitor:
        if path != "":
            env = VecMonitor(env, filename=os.path.join(path, f"{configName}-{rank}"), info_keywords=info_keywords)
        else:
            env = VecMonitor(env, filename=f"{configName}-{rank}", info_keywords=info_keywords)
    if rank == 0 and vecNormalize:
        # must be after VecMonitor for monitor to show unnormed rewards
        env = VecNormalizeMultiAgent(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    return env


def wrap_env_full(env, reduce_color=False, memory=1, size=32, vecMonitor=False,
                  configName=None, threads=1, rank=0, vecNormalize=False, style='rich', monitor_path=None):
    if reduce_color:
        env = ss.color_reduction_v0(env, 'B')
    if style != 'rich' and False:
        env = ss.resize_v0(env, x_size=size, y_size=size)
    if reduce_color:
        env = ss.reshape_v0(env, (size, size, 1))
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v1(env, threads, num_cpus=1, base_class='stable_baselines3')
    # num_cpus=1 changed from 2 to avoid csv issues. does it affect speed?

    if style != 'rich':
        # this line might be causing issues anyway, should check
        env = VecTransposeImage(env)
    if memory > 1:
        env = VecFrameStack(env, n_stack=memory, channels_order='first')
        # consider StackedObservations

    if vecMonitor:
        if monitor_path != "":
            env = VecMonitor(env, filename=os.path.join(monitor_path, f"{configName}-{rank}"),
                             info_keywords=env.info_keywords)
        else:
            env = VecMonitor(env, filename=f"{configName}-{rank}", info_keywords=env.info_keywords)
    if rank == 0 and vecNormalize:
        # must be after VecMonitor for monitor to show unnormed rewards
        env = VecNormalizeMultiAgent(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    return env


def wrap_env(para_env, **kwargs):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env(para_env, **kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(para_env, **kwargs):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = para_env(**kwargs)
    env = parallel_to_aec(env)
    return env


def pz2sb3(env, num_cpus=2):
    '''
    takes a wrapped env 
    returns sb3 compatible via supersuit concatenated set of vector environments
    '''
    envInstance = aec_to_parallel(env)
    env2 = ss.black_death_v3(ss.pad_action_space_v0(ss.pad_observations_v0(envInstance)))
    env2 = ss.pettingzoo_env_to_vec_env_v1(env2)
    env2 = ss.concat_vec_envs_v1(env2, num_cpus, base_class='stable_baselines3')
    env2.black_death = True
    return env2
