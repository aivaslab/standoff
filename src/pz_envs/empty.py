from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

def EmptyMultiGrid(**kwargs):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_EmptyMultiGrid(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_EmptyMultiGrid(para_MultiGridEnv):
    mission = "get to the green square"
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "doorkey"}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(color="green", reward=1), width - 2, height - 2)


        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)