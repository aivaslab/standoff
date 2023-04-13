from ..base_AEC import para_MultiGridEnv, MultiGrid
from ..objects import Wall, Door, Key, Goal, Lava


class EmptyMultiGrid(para_MultiGridEnv):
    mission = "get to the green square"
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "doorkey"}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(color="green", reward=1), width - 2, height - 2)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)
