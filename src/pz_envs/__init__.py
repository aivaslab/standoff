from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

from .standoff import *
from .ministandoff import *
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
        _difficulty,
        reward_decay=False,
        _observation_style='rich',
        #observation_density=1,
        view_tile_size=1,
        _view_size=15,
        view_offset=4,
        use_label=False,
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
        "supervised_model": None if not use_label else 1,
    }


    player_config = {
        "view_size": _view_size,
        "view_offset": view_offset,
        "view_tile_size": view_tile_size,
        "observation_style": _observation_style,
        "see_through_walls": False,
        "color": "prestige",
        "view_type": 0,
        "move_type": 0
    }

    reset_configs = {**configs["defaults"], **configs[config_name]}


    env_config['config_name'] = config_name
    env_config['agents'] = [GridAgentInterface(**player_config) for _ in range(reset_configs['num_agents'])]
    env_config['puppets'] = [GridAgentInterface(**player_config) for _ in 
        range(max(reset_configs['num_puppets']) if isinstance(reset_configs['num_puppets'], list) 
        else reset_configs['num_puppets'])]

    # num_puppets contains lists, not maxima
    #env_config['num_puppets'] = reset_configs['num_puppets']

    class RegEnv(env_class):
        def __new__(cls):
            env = super(env_class, RegEnv).__new__(env_class)
            env.__init__(
                opponent_visible_decs=(_difficulty < 1),
                persistent_treat_images=(_difficulty < 2),
                subject_visible_decs=(_difficulty < 3),
                gaze_highlighting=(_difficulty < 3),
                persistent_gaze_highlighting=(_difficulty < 2),
                **env_config,
                )

            return env

    env_class_name = f"env_{len(registered_envs)}"
    setattr(this_module, env_class_name, RegEnv)
    gym.logger.set_level(gym.logger.ERROR)
    registered_envs.append(env_name)
    gym_register(env_name, entry_point=f"standoff.src.pz_envs:{env_class_name}")


def env_from_config(env_config, randomize_seed=True):
    possible_envs = {k: v for k, v in globals().items() if inspect.isclass(v) and issubclass(v, para_MultiGridEnv)}

    env_class = possible_envs[env_config['env_class']]

    env_kwargs = {k: v for k, v in env_config.items() if k != 'env_class'}
    if randomize_seed:
        env_kwargs['seed'] = env_kwargs.get('seed', 0) + random.randint(0, 1337 * 1337)

    return env_class(**env_kwargs)

if __name__ == '__main__':
    conf = ScenarioConfigs()
    
    for observation_style in ['rich']:
        for use_label in [True, False]:
            for view_size in [7, 17]:
                for difficulty in [3]:
                    for config in conf.standoff.keys():
                        configName = config.replace(" ", "")
                        register_standoff_env(
                            f"Standoff-{configName}-{view_size}-{observation_style}-{difficulty}-v0" if not use_label else
                            f"Standoff-{configName}-{view_size}-{observation_style}-{difficulty}-v1",
                            StandoffEnv if view_size == 17 else MiniStandoffEnv,
                            config,
                            difficulty,
                            _observation_style=observation_style,
                            #observation_density=1,
                            view_tile_size=1,
                            _view_size=view_size,
                            view_offset=3 if view_size > 7 else 0,
                            use_label=use_label,
                        )