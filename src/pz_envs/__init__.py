from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

from .tutorial import *
from .standoff import *
from .doorkey import *
from .empty import *

from ..agents import GridAgentInterface
from gym.envs.registration import register as gym_register
from .scenario_configs import ScenarioConfigs

import sys
import inspect
import random

this_module = sys.modules[__name__]
registered_envs = []


def register_standoff_env(
        env_name,
        env_class,
        config_name,
        difficulty,
        reward_decay=True,
        observation_style='rich',
        observation_density=1,
        view_tile_size=1,
        view_size=15,
        view_offset=4,
):
    configs = ScenarioConfigs().standoff

    env_config = {
        "max_steps": 50,
        "respawn": True,
        "reward_decay": reward_decay,
        "width": 19,
        "height": 19,
        "step_reward": -0.1,
        "config_name": config_name,
    }


    player_config = {
        "view_size": view_size,
        "view_offset": view_offset,
        "view_tile_size": view_tile_size,
        "observation_style": observation_style,
        "observation_density": observation_density,
        "see_through_walls": False,
        "color": "prestige",
        "view_type": 0,
        "move_type": 0
    }

    reset_configs = {**configs["defaults"], **configs[config_name]}

    if isinstance(reset_configs["num_agents"], list):
        reset_configs["num_agents"] = reset_configs["num_agents"][0]
    if isinstance(reset_configs["num_puppets"], list):
        reset_configs["num_puppets"] = reset_configs["num_puppets"][0]

    env_config['config_name'] = config_name
    env_config['agents'] = [GridAgentInterface(**player_config) for _ in range(reset_configs['num_agents'])]
    env_config['puppets'] = [GridAgentInterface(**player_config) for _ in range(reset_configs['num_puppets'])]

    # env_config['num_agents'] = reset_configs['num_agents']
    # env_config['num_puppets'] = reset_configs['num_puppets']

    class RegEnv(env_class):
        def __new__(cls):
            env = super(env_class, RegEnv).__new__(env_class)
            env.__init__(
                opponent_visible_decs=(difficulty < 1),
                persistent_treat_images=(difficulty < 2),
                subject_visible_decs=(difficulty < 3),
                gaze_highlighting=(difficulty < 3),
                persistent_gaze_highlighting=(difficulty < 2),
                **env_config)

            return env

    env_class_name = f"env_{len(registered_envs)}"
    setattr(this_module, env_class_name, RegEnv)
    registered_envs.append(env_name)
    gym_register(env_name, entry_point=f"standoff.src.pz_envs:{env_class_name}")


def env_from_config(env_config, randomize_seed=True):
    possible_envs = {k: v for k, v in globals().items() if inspect.isclass(v) and issubclass(v, para_MultiGridEnv)}

    env_class = possible_envs[env_config['env_class']]

    env_kwargs = {k: v for k, v in env_config.items() if k != 'env_class'}
    if randomize_seed:
        env_kwargs['seed'] = env_kwargs.get('seed', 0) + random.randint(0, 1337 * 1337)

    return env_class(**env_kwargs)


for observation_style in 'rich', 'image':
    for view_size in [13, 15, 17, 19]:
        for difficulty in range(3):
            for stage in range(3):
                for config in ScenarioConfigs.stageNames[stage+1]:
                    configName = difficulty if stage < 2 else config.replace(" ", "") + "-" + str(difficulty)
                    register_standoff_env(
                        "Standoff-S{0}-{1}-{2}-{3}-v0".format(stage+1, configName, view_size, observation_style),
                        StandoffEnv,
                        config,
                        difficulty,
                        observation_style=observation_style,
                        observation_density=1,
                        view_tile_size=1,
                        view_size=view_size,
                        view_offset=4,
                    )
