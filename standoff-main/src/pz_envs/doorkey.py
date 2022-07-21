from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

class para_DoorKeyEnv(para_MultiGridEnv):
    """
    Environment with a door and key, sparse reward.
    Similar to DoorKeyEnv in 
        https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/doorkey.py
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {'render_modes': ['human', 'rgb_array'], "name": "doorkey"}

    def _rand_int(self, x, y):
        return randrange(x, y)

    def _gen_grid(self, width, height):
        # Create an empty grid
        #print('runnning')
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(color="green", reward=100), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 1)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door(color="yellow", state=3), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        self.agent_spawn_kwargs = {'size': (splitIdx, 5)}
        self.place_agents(**self.agent_spawn_kwargs)
