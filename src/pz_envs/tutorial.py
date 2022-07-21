from ..base_AEC import *
from ..objects import *
from random import randrange
import random
from ..pz_envs.scenario_configs import ScenarioConfigs


class para_TutorialEnv(para_MultiGridEnv):
    """
    Environment sparse reward.
    Currently designed for 9x9 envs.
    """

    mission = "get to the goal"
    metadata = {'render_modes': ['human', 'rgb_array'], "name": "tutorial"}

    path = ''
    allParams = {"type": ["t", "n"], "var": ["a", "b", "c", "d", "e", "f", "g"], "puppets": [0]}
    params = {}
    configs = ScenarioConfigs.tutorial

    def hard_reset(self, params=None):
        self.params = params

    def timer_active(self, name):
        if "hide" in name:
            for x, y in self.box_locs:
                b1 = Box(color="yellow")
                c = self.grid.get(x, y)
                if c:
                    b1.contains = c
                    b1.can_overlap = c.can_overlap
                    b1.get_reward = c.get_reward
                else:
                    b1.can_overlap = lambda: True
                    b1.get_reward = lambda x: self.box_reward
                self.put_obj(b1, x, y)

    def _set_seed(self, seed):
        if seed != -1:
            self.seed_mode = True
            self.curSeed = seed

    def _rand_int(self, x, y):
        return randrange(x, y)

    def _gen_grid(self, width, height, eType="t", eVar="a", puppets=0):

        self.grid = MultiGrid((width, height))

        colors = random.sample(['purple', 'orange', 'yellow', 'blue', 'pink', 'red'], 4)

        # grid and surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if eType == "t":  # tutorial
            if eVar in "bcd":
                for x in range(2, width - 1, 2):
                    for y in range(2, height - 1, 2):
                        if eVar in "cd":
                            self.put_obj(Wall(), x, y)
                        else:
                            num = randrange(0, 10)
                            if num == 0:
                                self.put_obj(Wall(), x, y)
                            elif num == 1:
                                self.put_obj(Door(color=colors[0], state=randrange(1, 4)), x, y)
                            elif num == 2:
                                self.put_obj(Key(color=colors[0], state=randrange(1, 4)), x, y)
            if eVar == "d":
                for i in range(3):
                    num = randrange(0, 3)
                    if num == 0:
                        self.place_obj(Wall(), top=(0, 0), size=(width, height))
                    elif num == 1:
                        self.place_obj(Door(color=colors[0], state=randrange(1, 4)), top=(0, 0), size=(width, height))
                    elif num == 2:
                        self.place_obj(Key(color=colors[0], state=randrange(1, 4)), top=(0, 0), size=(width, height))

            self.box_locs = []
            self.box_locs.append(self.place_obj(Goal(color="green", reward=100), top=(0, 0), size=(width, height)))
            if eVar in "eg":
                # preferences
                self.box_locs.append(
                    self.place_obj(Goal(color="green", reward=50, size=0.5), top=(0, 0), size=(width, height)))
            if eVar in "f":
                self.box_locs.append(
                    self.place_obj(Goal(color="green", reward=1, size=0.01), top=(0, 0), size=(width, height)))
            if eVar in "fg":
                # memory
                self.timers = {}
                self.add_timer("hide", random.randint(2, 5))
                pass

        elif eType == "n":  # memory navigation

            print("n")
            self.grid.wall_rect(0, 1, width - 1, height - 2)

            goals = random.sample([0, 1], 2)

            self.put_obj(Lava(), 4, 4)
            self.put_obj(Lava(), 5, 4)

            for x in range(2, 6):
                self.put_obj(Lava(), x, 5)
                self.put_obj(Lava(), x, 3)

            if eVar in 'a':  # no visible goal
                pass
            if eVar in 'bcd':
                self.put_obj(Goal(reward=100, color='green'), 6, 4)
            if eVar in 'efgh':  # offset goal
                self.put_obj(Goal(reward=100, color='green'), 6, 3 + 2 * goals[0])
            if eVar in 'cd':  # one path blocked by lava
                self.put_obj(Lava(), 6, 3 + 2 * goals[0])
                self.put_obj(Lava(), 6, 2 + 4 * goals[0])
            if eVar in 'd':  # alt path blocked by lava
                self.put_obj(Lava(), 6, 3 + 2 * goals[1])
                self.put_obj(Lava(), 6, 2 + 4 * goals[1])
            if eVar in 'g':  # alt offset goal (smaller)
                self.put_obj(Goal(reward=50, color='green', size=0.5), 6, 3 + 2 * goals[1])
            if eVar in 'fh':  # lava btw offsets
                self.put_obj(Lava(), 6, 5)
            print("n2")

            self.agent_spawn_kwargs = {'top': (3, 4), 'size': (1, 1)}

        # self.place_agents(**self.agent_spawn_kwargs)
