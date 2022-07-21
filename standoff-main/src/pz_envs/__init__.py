from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

from .compfeed import *
from .tutorial import *
from .standoff import *
from .contentFB import *
from .doorkey import *
from .empty import *
from .knowguess import *
from .yummyyucky import *
from .sallyanne import *


from ..agents import GridAgentInterface
from gym.envs.registration import register as gym_register


import sys
import inspect
import random

this_module = sys.modules[__name__]
registered_envs = []


def register_marl_env(
    env_name,
    env_class,
    n_agents,
    grid_size,
    view_size,
    view_tile_size=8,
    view_offset=0,
    agent_color=None,
    env_kwargs={},
):
    colors = ["red", "blue", "purple", "orange", "olive", "pink"]
    assert n_agents <= len(colors)

    class RegEnv(env_class):
        def __new__(cls):
            instance = super(env_class, RegEnv).__new__(env_class)
            instance.__init__(
                agents=[
                    GridAgentInterface(
                        color=c if agent_color is None else agent_color,
                        view_size=view_size,
                        view_tile_size=8,
                        view_offset=view_offset,
                        )
                    for c in colors[:n_agents]
                ],
                grid_size=grid_size,
                **env_kwargs,
            )
            return instance

    env_class_name = f"env_{len(registered_envs)}"
    setattr(this_module, env_class_name, RegEnv)
    registered_envs.append(env_name)
    gym_register(env_name, entry_point=f"marlgrid.envs:{env_class_name}")


def env_from_config(env_config, randomize_seed=True):
    possible_envs = {k:v for k,v in globals().items() if inspect.isclass(v) and issubclass(v, para_MultiGridEnv)}
    
    env_class = possible_envs[env_config['env_class']]
    
    env_kwargs = {k:v for k,v in env_config.items() if k != 'env_class'}
    if randomize_seed:
        env_kwargs['seed'] = env_kwargs.get('seed', 0) + random.randint(0, 1337*1337)
    
    return env_class(**env_kwargs)

register_marl_env(
    "MarlGrid-1AgentDoorKeyEnv7x7-v0",
    para_DoorKeyEnv,
    grid_size=7,
    view_size=7,
    n_agents=1,
)

register_marl_env(
    "MarlGrid-1AgentSallyAnneEnv15x15-v0",
    para_SallyAnneEnv,
    grid_size=15,
    view_size=7,
    n_agents=1,
)

register_marl_env(
    "MarlGrid-1AgentTutorialEnv9x9-v0",
    para_TutorialEnv,
    grid_size=9,
    view_size=7,
    n_agents=1,
)
