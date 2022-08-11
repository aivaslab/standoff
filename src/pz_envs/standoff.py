from ..base_AEC import para_MultiGridEnv, MultiGrid
from ..objects import Wall, Goal, Curtain, Block, GlassBlock, Box
import random
from ..puppets import pathfind
import copy
from ..pz_envs.scenario_configs import ScenarioConfigs


class StandoffEnv(para_MultiGridEnv):
    mission = "get the best food before your opponent"
    metadata = {'render_modes': ['human', 'rgb_array'], "name": "standoffEnv"}
    configs = ScenarioConfigs.standoff

    def __init__(
            self,
            agents=None,
            puppets=None,
            grid_size=None,
            width=11,
            height=11,
            max_steps=100,
            memory=1,
            colorMemory=False,
            reward_decay=False,
            seed=1337,
            respawn=False,
            ghost_mode=True,
            step_reward=0,
            done_reward=-10,
            agent_spawn_kwargs=None,
            num_puppets=1,
            num_agents=1,
            configName=""
    ):
        super().__init__(agents, puppets, grid_size, width, height, max_steps, memory, colorMemory, reward_decay, seed,
                         respawn, ghost_mode, step_reward, done_reward, agent_spawn_kwargs)
        if agent_spawn_kwargs is None:
            agent_spawn_kwargs = {'top': (0, 0), 'size': (2, self.width)}
        self.agent_spawn_kwargs = agent_spawn_kwargs
        if agents is None:
            agents = []
        if puppets is None:
            puppets = []
        self.params = None
        self.configName = configName

    def hard_reset(self, params=None):
        """
        Reset the environment params.
        """
        defaults = ScenarioConfigs.standoff["defaults"]

        if params is None:
            params = {}
        newParams = copy.copy(params)
        for k in defaults.keys():
            if k in params.keys():
                if isinstance(params[k], list):
                    newParams[k] = random.choice(params[k])
            else:
                if isinstance(defaults[k], list):
                    newParams[k] = random.choice(defaults[k])
                else:
                    newParams[k] = defaults[k]
        self.params = newParams

        # special since max_puppets is weird
        # maybe delete this
        self.possible_puppets = ["player_" + str(x + len(self.agents)) for x in range(self.params["num_puppets"])]

    def reset_vision(self):
        """
        Reset the vision of the agents.
        """
        boxes = self.params["boxes"]
        for agent in self.agents_and_puppets():
            self.agent_goal[agent] = random.choice(range(boxes))
            self.best_reward[agent] = -100
            for box in range(boxes):
                self.last_seen_reward[agent + str(box)] = -100
                if agent + str(box) not in self.can_see.keys():
                    self.can_see[agent + str(box)] = True  # default to not hidden until it is

    def _gen_grid(self, width, height,
                  adversarial=True,
                  hidden=True,
                  rational=True,
                  sharedRewards=False,
                  boxes=5,
                  sub_valence=1,
                  dom_valence=1,
                  num_puppets=1,
                  followDistance=1,
                  lavaHeight=2,
                  baits=1,
                  baitSize=2,
                  informed='informed',
                  swapType='swap',
                  visibility='curtains',
                  cause='blocks',
                  lava='lava',
                  firstBig=True,
                  num_agents=1,
                  ):

        self.hard_reset(self.configs[self.configName])
        startRoom = 2
        atrium = 2

        if swapType == "replace" and boxes <= 2:
            swapType = "swap"

        self.box_reward = 1
        self.food_locs = list(range(boxes))
        random.shuffle(self.food_locs)
        self.release = [[] for _ in range(4)]
        releaseGap = boxes * 2 + atrium
        self.width = boxes * 2 + 3
        self.height = lavaHeight + startRoom * 2 + atrium * 2 + 2
        self.grid = MultiGrid((self.width, self.height))
        self.grid.wall_rect(1, 1, self.width - 2, self.height - 2)

        self.agent_spawn_pos = {}
        self.agent_door_pos = {}
        for k, agent in enumerate(self.agents_and_puppets()):
            h = 1 if agent == "player_0" else self.height - 2
            d = 1 if agent == "player_0" else 3
            xx = 2 * random.choice(range(boxes)) + 2
            self.agent_spawn_pos[agent] = (xx, h, d)
            self.agent_door_pos[agent] = (xx, h + (1 if agent == "player_0" else -1))
            a = self.instance_from_name[agent]
            a.valence = sub_valence if agent == "player_0" else dom_valence
            if k > num_puppets:
                a.spawn_delay = 1000
                a.active = False

        for j in range(self.width):
            self.put_obj(Wall(), j, startRoom + atrium)
            self.put_obj(Wall(), j, startRoom)
            self.put_obj(Wall(), j, self.height - startRoom - atrium - 1)
            self.put_obj(Wall(), j, self.height - startRoom - 1)

        for j in range(2, self.width - 2):
            if visibility == "curtains":
                for i in range(startRoom + 1, startRoom + atrium):
                    self.put_obj(Curtain(color='red'), j, i)
                for i in range(self.height - startRoom - atrium - 1 + 1, self.height - startRoom - 1):
                    self.put_obj(Curtain(color='red'), j, i)

        self.grid.wall_rect(0, 0, self.width, self.height)

        for box in range(boxes + 1):
            if box < boxes:
                self.put_obj(Wall(), box * 2 + 1, startRoom - 1)
                self.put_obj(Block(init_state=0, color="blue"), box * 2 + 2, startRoom)
                self.put_obj(Block(init_state=0, color="blue"), box * 2 + 2, self.height - startRoom - 1)

                self.release[0] += [(box * 2 + 2, startRoom)]
                self.release[1] += [(box * 2 + 2, self.height - startRoom - 1)]

                self.put_obj(Wall(), box * 2 + 1, self.height - 2)
                self.put_obj(Block(init_state=0, color="blue"), box * 2 + 2, startRoom + atrium)
                self.put_obj(Block(init_state=0, color="blue"), box * 2 + 2, self.height - startRoom - atrium - 1)

                self.release[2] += [(box * 2 + 2, startRoom + atrium)]
                self.release[3] += [(box * 2 + 2, self.height - startRoom - atrium - 1)]

            for j in range(lavaHeight):
                x = box * 2 + 1
                y = j + startRoom + atrium + 1
                self.put_obj(GlassBlock(color="cyan", init_state=1), x, y)

        self.agent_goal, self.last_seen_reward, self.can_see, self.best_reward = {}, {}, {}, {}
        self.reset_vision()
        # init timers

        self.timers = {}
        curTime = 1
        self.add_timer("init", 1)
        for bait in range(0, baits * baitSize, baitSize):
            baitLength = 7
            informed2 = informed
            if informed == "half1":
                informed2 = "informed" if bait == 0 else "uninformed"
            elif informed == "half2":
                informed2 = "informed" if bait == 1 else "uninformed"

            if informed2 == "informed":
                # no hiding
                swapTime = random.randint(1, baitLength - 1)
            elif informed2 == "uninformed":
                # swap during blind
                swapTime = random.randint(1, baitLength - 2)
                blindStart = random.randint(0, swapTime)
                blindStop = random.randint(swapTime, baitLength)
                self.add_timer("blind player_1", curTime + blindStart)
                self.add_timer("reveal player_1", curTime + blindStop)
            elif informed2 == "fake":
                # swap/hide before or after blind
                if random.choice([True, False]):
                    swapTime = random.randint(1, baitLength)
                    blindStart = random.randint(0, swapTime - 2)
                    blindStop = random.randint(blindStart, swapTime - 1)
                else:
                    swapTime = random.randint(0, baitLength - 3)
                    blindStart = swapTime + random.randint(swapTime, baitLength - 1)
                    blindStop = swapTime + random.randint(blindStart, baitLength)

                assert blindStart < blindStop
                assert blindStop < baitLength

                self.add_timer("blind player_1", curTime + blindStart)
                self.add_timer("reveal player_1", curTime + blindStop)
            else:
                swapTime = 1000  # no swap
            if bait < 2:
                if baitSize == 2:
                    self.add_timer("place12", curTime + swapTime)
                elif baitSize == 1:
                    if firstBig == bait:
                        self.add_timer("place1", curTime + swapTime)
                    else:
                        self.add_timer("place2", curTime + swapTime)
            else:
                st = swapType
                if "remove" in st:
                    st = st + random.choice(["1", "2"])
                self.add_timer(st, curTime + swapTime)
            if hidden:
                if bait + baitSize < 2:
                    if firstBig == bait:
                        self.add_timer("hide1", curTime + swapTime + 1)
                    else:
                        self.add_timer("hide2", curTime + swapTime + 1)
                if bait + baitSize > baits - 1:
                    self.add_timer("hideall", curTime + swapTime + 1)
            curTime += baitLength

        if followDistance < 0:
            subRelease = 0
            domRelease = -followDistance
        else:
            subRelease = followDistance
            domRelease = 0
        self.add_timer("release_0", curTime + 1 + domRelease)
        self.add_timer("release_1", curTime + 1 + subRelease)
        self.add_timer("release_2", curTime + 1 + releaseGap + domRelease)
        self.add_timer("release_3", curTime + 1 + releaseGap + subRelease)

    def timer_active(self, name):
        # todo: make all these events more sensibly written, not dependent on food starting locs
        boxes = self.params["boxes"]
        firstBig = self.params["firstBig"]
        followDistance = self.params["followDistance"]
        y = self.height // 2 - followDistance
        if "release" in name:
            release_no = int(name[-1])
            for xx, yy in self.release[release_no]:
                self.del_obj(xx, yy)
        if "place" in name or "hide" in name or "remove" in name:
            for box in range(boxes):
                x = box * 2 + 2
                if "place" in name:
                    if box == self.food_locs[not firstBig] and "1" in name:
                        self.put_obj(Goal(reward=100, size=1.0, color='green'), x, y)
                    if box == self.food_locs[firstBig] and "2" in name:
                        self.put_obj(Goal(reward=25, size=0.5, color='green'), x, y)

                elif "hide" in name:
                    if "all" in name or (box == self.food_locs[not firstBig] and "1" in name) or (
                            box == self.food_locs[firstBig] and "2" in name):
                        b1 = Box(color="yellow")
                        c = self.grid.get(x, y)
                        if c:
                            b1.contains = c
                            b1.can_overlap = c.can_overlap
                            b1.get_reward = c.get_reward
                        else:
                            b1.can_overlap = lambda: True
                            b1.get_reward = lambda x: self.box_reward
                            # todo: why does one of these have arg? overlap is property?
                        self.put_obj(b1, x, y)

                elif "remove" in name:
                    if box == self.food_locs[0] and "1" in name:
                        self.del_obj(x, y)
                    elif box == self.food_locs[1] and "2" in name:
                        self.del_obj(x, y)
        if name == "replace":
            # swap big food with a no food tile
            # currently only does big food, should it do small?
            for box in range(boxes):
                x = box * 2 + 2
                y = self.height // 2
                if box == self.food_locs[2]:
                    self.put_obj(Goal(reward=100, size=1.0, color='green'), x, y)
                elif box == self.food_locs[not firstBig]:
                    self.del_obj(x, y)
        if name == "move":
            # both foods are moved to new locations
            for box in range(boxes):
                x = box * 2 + 2
                if box == self.food_locs[2]:
                    self.put_obj(Goal(reward=100, size=1.0, color='green'), x, y)
                if box == self.food_locs[3]:
                    self.put_obj(Goal(reward=25, size=0.5, color='green'), x, y)
                elif box == self.food_locs[not firstBig] or box == self.food_locs[firstBig]:
                    self.del_obj(x, y)
        if name == "swap":
            for box in range(boxes):
                x = box * 2 + 2
                if box == self.food_locs[firstBig]:
                    self.put_obj(Goal(reward=100, size=1.0, color='green'), x, y)
                elif box == self.food_locs[not firstBig]:
                    self.put_obj(Goal(reward=25, size=0.5, color='green'), x, y)

        if "blind" in name or "reveal" in name:
            splitName = name.split()
            b = self.grid.get(*self.agent_door_pos[splitName[1]])

            if "blind" in name:
                b.state = 1
                b.see_behind = lambda: False
            if "reveal" in name:
                b.state = 0
                b.see_behind = lambda: True
            # record whether each agent can see each food
            agent = self.instance_from_name[splitName[1]]
            for box in range(boxes):
                self.can_see[splitName[1] + str(box)] = False if "blind" in name else True

        # whenever food updates, remember locations
        if name in ["init", "swap", "replace", "reveal"] or "remove" in name or "place" in name or "release" in name:

            for box in range(boxes):
                x = box * 2 + 2
                for agent in self.agents_and_puppets():
                    if self.can_see[agent + str(box)]:
                        tile = self.grid.get(x, y)
                        if hasattr(tile, "reward") and hasattr(tile, "size"):
                            # size used to distinguish treats from boxes
                            self.last_seen_reward[agent + str(box)] = tile.reward if isinstance(tile.reward, int) else 0
                            # print('rew update', agent, box, tile.reward)
                        elif not self.grid.get(x, y) and self.last_seen_reward[agent + str(box)] != 0:
                            # print('0ing', box)
                            self.last_seen_reward[agent + str(box)] = 0

            new_target = False
            target_agent = None
            for box in range(boxes):
                for agent in self.agents_and_puppets():
                    reward = self.last_seen_reward[agent + str(box)]
                    if (self.agent_goal[agent] != box) and (reward >= self.best_reward[agent]):
                        self.agent_goal[agent] = box
                        self.best_reward[agent] = reward
                        new_target = True
                        target_agent = agent
            if new_target and target_agent != "player_0":
                a = self.instance_from_name[target_agent]
                if a.active:
                    x = self.agent_goal[target_agent] * 2 + 2
                    path = pathfind(self.grid.overlapping, a.pos, (x, y))
                    self.infos[target_agent]["path"] = path
