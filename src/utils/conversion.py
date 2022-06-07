from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import supersuit as ss
from pettingzoo.utils import wrappers
from ..agents import GridAgentInterface
from marlgrid.marlgrid.pz_envs import env_from_config
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, VecVideoRecorder, VecTransposeImage
import os

def make_env(envClass, player_config, configName=None, memory=1, threads=1, reduce_color=False, size=64,
    reward_decay=False, ghost_mode=True, max_steps=50, saveVids=False, path="", recordEvery=1e4):

    player_interface_config = player_config
    agents = [GridAgentInterface(**player_config) for _ in range(1)]

    env_config =  {
        "env_class": envClass,
        "max_steps": max_steps,
        "respawn": True,
        "ghost_mode": ghost_mode,
        "reward_decay": reward_decay,
        "width": 9 if envClass == "para_TutorialEnv" else 19,
        "height": 9 if envClass == "para_TutorialEnv" else 19,
        "agents": agents,
        "memory": memory,
        "step_reward": -0.1,
    }

    env = env_from_config(env_config)
    env.agent_view_size = player_interface_config["view_size"]*player_interface_config["view_tile_size"]

    configName = random.choice(list(env.configs.keys())) if configName == None else configName
    env.hard_reset(env.configs[configName])

    #train on multiple configs how?

    if reduce_color:
        env = ss.color_reduction_v0(env, 'B')
    env = ss.resize_v0(env, x_size=size, y_size=size)
    if reduce_color:
        env = ss.reshape_v0(env, (size, size, 1))
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, threads, num_cpus=2, base_class='stable_baselines3')
    env = VecTransposeImage(env)
    if memory > 1:
        env = VecFrameStack(env, n_stack=memory)
        #consider StackedObservations
    if saveVids:
        env = VecVideoRecorder(env, path, record_video_trigger=lambda x: x % recordEvery == 0, video_length=50, name_prefix=configName)
    if path != "":
        env = VecMonitor(env, filename=os.path.join(path, "timesteps"))
    else:
        env = VecMonitor(env)
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
