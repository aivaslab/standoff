import numpy as np

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
    info_keywords = ('minibatch', 'timestep',
                     'shouldAvoidBig', 'shouldAvoidSmall', 'correctSelection', 'selection',
                    'selectedBig', 'selectedSmall', 'selectedNeither',
                    'selectedPrevBig', 'selectedPrevSmall', 'selectedPrevNeither', 'incorrectSelection')

    def __init__(
            self,
            agents=None,
            puppets=None,
            grid_size=None,
            width=11,
            height=11,
            max_steps=100,
            reward_decay=False,
            seed=1337,
            respawn=False,
            ghost_mode=True,
            step_reward=0,
            done_reward=-10,
            agent_spawn_kwargs=None,
            config_name="error",
            subject_visible_decs=False,
            opponent_visible_decs=False,
            persistent_treat_images=False,
            gaze_highlighting=False,
            persistent_gaze_highlighting=False,
    ):
        super().__init__(agents,
                         puppets,
                         grid_size,
                         width,
                         height,
                         max_steps,
                         reward_decay,
                         seed,
                         respawn,
                         ghost_mode,
                         step_reward,
                         done_reward,
                         agent_spawn_kwargs,
                         num_agents=len(agents),
                         num_puppets=len(puppets),
                         config_name=config_name,
                         subject_visible_decs=subject_visible_decs,
                         opponent_visible_decs=opponent_visible_decs,
                         persistent_treat_images=persistent_treat_images,
                         gaze_highlighting=gaze_highlighting,
                         persistent_gaze_highlighting=persistent_gaze_highlighting,)
        self.new_target = None
        if agent_spawn_kwargs is None:
            agent_spawn_kwargs = {'top': (0, 0), 'size': (2, self.width)}
        self.agent_spawn_kwargs = agent_spawn_kwargs
        if agents is None:
            agents = []
        if puppets is None:
            puppets = []
        self.params = None
        self.configName = config_name
        self.minibatch = 0
        self.deterministic = False #used for generating deterministic baiting events for ground-truth evaluation
        self.deterministic_seed = 0 #cycles baiting locations, changes by one each time it is used
        self.cached_paths = {}

        # difficulty settings
        self.subject_visible_decs = subject_visible_decs
        self.opponent_visible_decs = opponent_visible_decs
        self.persistent_treat_images = persistent_treat_images
        self.gaze_highlighting = gaze_highlighting
        self.persistent_gaze_highlighting = persistent_gaze_highlighting
        self.only_highlight_treats = True

        self.fill_spawn_holes = False
        self.random_subject_spawn = False


    def hard_reset(self, params=None):
        """
        Reset the environment params.
        """
        defaults = ScenarioConfigs.standoff["defaults"]

        if params is None:
            params = {}
        newParams = copy.deepcopy(params)
        for k in defaults.keys():
            if k in params.keys():
                if isinstance(params[k], list):
                    newParams[k] = params[k][self.deterministic_seed % len(params[k])] if self.deterministic else random.choice(params[k])
            else:
                if isinstance(defaults[k], list):
                    newParams[k] = defaults[k][self.deterministic_seed % len(defaults[k])] if self.deterministic else random.choice(defaults[k])
                else:
                    newParams[k] = defaults[k]
        self.params = copy.deepcopy(newParams)

        # special since max_puppets is weird
        # maybe delete this
        self.possible_puppets = [f"player_{x + len(self.agents)}" for x in range(self.params["num_puppets"])]

    def reset_vision(self):
        """
        Reset the vision of the agents.
        """
        self.agent_goal, self.last_seen_reward, self.can_see, self.best_reward = {}, {}, {}, {}
        boxes = self.params["boxes"]
        for agent in self.agents_and_puppets():
            self.agent_goal[agent] = self.deterministic_seed % boxes if self.deterministic else random.choice(range(boxes))
            self.best_reward[agent] = -100
            for box in range(boxes):
                self.last_seen_reward[agent + str(box)] = -100
                if agent + str(box) not in self.can_see.keys():
                    self.can_see[agent + str(box)] = True  # default to not hidden until it is

    def get_deterministic_seed(self):
        #self.deterministic_seed += 1
        return self.deterministic_seed

    def _gen_grid(self,
                  sub_valence=1,
                  dom_valence=1,
                  num_puppets=1,
                  subject_is_dominant=False,
                  lava_height=2,
                  events=[],
                  hidden=False,
                  share_rewards=False,
                  boxes=5,
                  ):

        startRoom = 2
        atrium = 2
        self.boxes = boxes

        self.hidden = hidden
        self.subject_is_dominant = subject_is_dominant
        self.box_reward = 1
        self.food_locs = list(range(boxes))
        if not self.deterministic:
            random.shuffle(self.food_locs)
        else:
            #rotate food locs list by deterministic seed
            self.food_locs = self.food_locs[self.deterministic_seed % len(self.food_locs):] + self.food_locs[:self.deterministic_seed % len(self.food_locs)]
        self.released_tiles = [[] for _ in range(4)]
        release_gap = boxes * 2 + atrium - 1
        self.width = boxes * 2 + 3
        self.height = lava_height + startRoom * 2 + atrium * 2 + 2
        self.grid = MultiGrid((self.width, self.height))
        self.grid.wall_rect(1, 1, self.width - 2, self.height - 2)
        self.small_food_locations = []
        self.big_food_locations = []

        self.objs_to_hide = []

        self.agent_spawn_pos = {}
        self.agent_door_pos = {}

        all_door_poses = []

        self.smallReward = int(100/(self.boxes-2))
        for k, agent in enumerate(self.agents_and_puppets()):
            h = 1 if agent == "player_0" else self.height - 2
            d = 1 if agent == "player_0" else 3
            if self.random_subject_spawn:
                bb = self.deterministic_seed % self.boxes if self.deterministic else random.choice(range(boxes))
                xx = 2 * bb + 2
            else:
                xx = 2 * self.boxes//2 + 1
            self.agent_spawn_pos[agent] = (xx, h, d)
            self.agent_door_pos[agent] = (xx, h + (1 if agent == "player_0" else -1))
            all_door_poses.append(self.agent_door_pos[agent])
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
            if not self.subject_visible_decs:
                for i in range(startRoom + 1, startRoom + atrium):
                    self.put_obj(Curtain(color='red'), j, i)
            if not self.opponent_visible_decs:
                for i in range(self.height - startRoom - atrium - 1 + 1, self.height - startRoom - 1):
                    self.put_obj(Curtain(color='red'), j, i)

        self.grid.wall_rect(0, 0, self.width, self.height)

        for box in range(boxes + 1):
            if box < boxes:
                self.put_obj(Wall(), box * 2 + 1, startRoom - 1)

                # initial door release, only where door is not in all_door_poses
                if (box * 2 + 2, startRoom) in all_door_poses:
                    self.put_obj(Block(init_state=0, color="blue"), box * 2 + 2, startRoom)
                else:
                    self.put_obj(Wall(), box * 2 + 2, startRoom)

                # same as above, for opponent
                if (box * 2 + 2, self.height - startRoom - 1) in all_door_poses:
                    self.put_obj(Block(init_state=0, color="blue"), box * 2 + 2, self.height - startRoom - 1)
                else:
                    self.put_obj(Wall(), box * 2 + 2, self.height - startRoom - 1)

                if (box * 2 + 2, startRoom) in all_door_poses:
                    self.released_tiles[0] += [(box * 2 + 2, startRoom)]
                if (box * 2 + 2, self.height - startRoom - 1) in all_door_poses:
                    self.released_tiles[1] += [(box * 2 + 2, self.height - startRoom - 1)]

                # secondary door release
                self.put_obj(Wall(), box * 2 + 1, self.height - 2)
                self.put_obj(Block(init_state=0, color="blue"), box * 2 + 2, startRoom + atrium)
                self.put_obj(Block(init_state=0, color="blue"), box * 2 + 2, self.height - startRoom - atrium - 1)

                self.released_tiles[2] += [(box * 2 + 2, startRoom + atrium)]
                self.released_tiles[3] += [(box * 2 + 2, self.height - startRoom - atrium - 1)]

            for j in range(lava_height):
                x = box * 2 + 1
                y = j + startRoom + atrium + 1
                self.put_obj(GlassBlock(color="cyan", init_state=1), x, y)

        self.reset_vision()

        ## Bucket location allocation for timers
        empty_buckets = [i for i in range(boxes)]
        event_args = [None for _ in range(len(events))]
        bait_args = [self.smallReward, 100]
        for k, event in enumerate(events):
            type = event[0]
            for x in range(len(event)):
                if event[x] == "empty":
                    event[x] = empty_buckets.pop(random.randrange(len(empty_buckets)) if not self.deterministic else self.get_deterministic_seed() % len(empty_buckets))
                elif event[x] == "else":
                    available_spots = [i for i in range(boxes) if i != event[x - 1]]
                    event[x] = available_spots.pop(random.randrange(len(available_spots)) if not self.deterministic else self.get_deterministic_seed() % len(available_spots))
                elif isinstance(event[x], int):
                    event[x] = events[event[x]][1]  # get first location

            if type == "bait":
                event_args[k] = bait_args.pop(random.randrange(len(bait_args)) if not self.deterministic else self.get_deterministic_seed() % len(bait_args))
            elif type == "remove":
                empty_buckets.append(event[1])
            elif type == "obscure" or type == "reveal":
                # hardcoded, will not work for multiple conspecifics
                event_args[k] = "player_1"
            # could also add functionality for moving empty buckets which are swapped, but that is not used in any tasks

        # add timers for events
        self.timers = {}
        curTime = 1
        self.add_timer(["init"], 1)
        for k, event in enumerate(events):
            self.add_timer(event, curTime, arg=event_args[k])
            curTime += 1
        if subject_is_dominant:
            sub_release = 1
            dom_release = 0
        else:
            sub_release = 0
            dom_release = 1
        self.add_timer(["release"], curTime + dom_release, arg=0)
        self.add_timer(["release"], curTime + sub_release, arg=1)
        self.add_timer(["release"], curTime + release_gap + dom_release, arg=2)
        self.add_timer(["release"], curTime + release_gap + sub_release, arg=3)

    def append_food_locs(self, obj, loc):
        if hasattr(obj, "reward") and obj.reward == 100:
            if len(self.big_food_locations) == 0 or (self.big_food_locations[-1] != loc):
                self.big_food_locations.append(loc)
        elif hasattr(obj, "reward") and obj.reward == self.smallReward:
            if len(self.small_food_locations) == 0 or (self.small_food_locations[-1] != loc):
                self.small_food_locations.append(loc)

    def get_all_paths(self, maze, position, offset=1):
        maze = np.array(maze).astype(int)
        if not self.deterministic:
            return None
        y = self.height//2 + offset
        paths = []
        for box in range(self.boxes):
            x = box * 2 + 2
            path = pathfind(maze, position[0:2], (x, y), self.cached_paths)
            paths += [path, ]
        return paths

    def timer_active(self, event, arg=None):
        name = event[0]
        arg = arg
        y = self.height // 2
        if self.hidden and len(self.objs_to_hide) > 0:
            for obj in self.objs_to_hide:
                pos = obj.pos
                self.put_obj(
                    Box("orange", contains=obj, reward=obj.reward, show_contains=self.persistent_treat_images),
                    pos[0], pos[1],
                    update_vis=False) # do not change gaze highlight if puppet saw this bait
        if name == 'init':
            for box in range(self.boxes):
                x = box * 2 + 2
                self.put_obj(Box(color="orange"), x, y)
            self.infos['player_0']['shouldAvoidBig'] = False
            self.infos['player_0']['shouldAvoidSmall'] = False
            self.infos['player_0']['correctSelection'] = -1
            self.infos['player_0']['incorrectSelection'] = -1
            self.infos['player_0']['minibatch'] = self.minibatch
            self.infos['player_0']['timestep'] = self.total_step_count
        elif name == 'bait':
            x = event[1] * 2 + 2
            obj = Goal(reward=arg, size=arg * 0.01, color='green', hide=self.hidden)
            self.put_obj(obj, x, y)
            self.objs_to_hide.append(obj)
        elif name == "remove":
            x = event[1] * 2 + 2
            self.del_obj(x, y)
        elif name == "obscure" or name == "reveal":
            b = self.grid.get(*self.agent_door_pos[arg])
            if name == "obscure":
                b.state = 1
                b.see_behind = lambda: False
            elif name == "reveal":
                b.state = 0
                b.see_behind = lambda: True
            for box in range(self.boxes):
                self.can_see[arg + str(box)] = False if "obscure" in name else True
        elif name == "swap":
            b1 = self.grid.get(event[1] * 2 + 2, y)
            b2 = self.grid.get(event[2] * 2 + 2, y)
            r1 = b1.reward if hasattr(b1, "reward") else 0
            r2 = b2.reward if hasattr(b2, "reward") else 0
            obj1 = Goal(reward=r2, size=r2 * 0.01, color='green', hide=self.hidden)
            obj2 = Goal(reward=r1, size=r1 * 0.01, color='green', hide=self.hidden)
            self.put_obj(obj1, event[1] * 2 + 2, y)
            self.put_obj(obj2, event[2] * 2 + 2, y)
            self.objs_to_hide.append(obj1)
            self.objs_to_hide.append(obj2)
        elif name == "release":
            for x, y in self.released_tiles[arg]:
                self.del_obj(x, y)

        # track where the big and small foods have been
        for box in range(self.boxes):
            x = box * 2 + 2
            obj = self.grid.get(x, y)
            self.append_food_locs(obj, box)

        # oracle food location memory for puppet ai
        if name == "bait" or name == "swap" or name == "remove" or (self.hidden is True and name == "reveal"):
            for box in range(self.boxes):
                x = box * 2 + 2
                for agent in self.agents_and_puppets():
                    if self.can_see[agent + str(box)]:
                        tile = self.grid.get(x, y)
                        #if hasattr(tile, "reward") and hasattr(tile, "size"):
                        if tile is not None and tile.type == "Goal":
                            # size used to distinguish treats from boxes
                            self.last_seen_reward[agent + str(box)] = tile.reward if isinstance(tile.reward, int) else 0
                            #print('rew update', agent, box, tile.reward)
                        elif not self.grid.get(x, y) and self.last_seen_reward[agent + str(box)] != 0:
                            self.last_seen_reward[agent + str(box)] = 0

            self.new_target = False
            for box in range(self.boxes):
                for agent in self.puppets:
                    reward = self.last_seen_reward[agent + str(box)]
                    if (self.agent_goal[agent] != box) and (reward >= self.best_reward[agent]):
                        self.agent_goal[agent] = box
                        self.best_reward[agent] = reward
                        self.new_target = True
                        self.target_agent = agent
        if self.new_target:
            self.new_target = False
            a = self.instance_from_name[self.target_agent]
            if a.active:
                x = self.agent_goal[self.target_agent] * 2 + 2
                path = pathfind(self.grid.overlapping_pathing, a.pos, (x, y), self.cached_paths)
                self.infos[self.target_agent]["path"] = path
                tile = self.grid.get(x, y)
                if hasattr(tile, "reward") and tile.reward == 100:
                    self.infos['player_0']['shouldAvoidBig'] = not self.subject_is_dominant
                    self.infos['player_0']['shouldAvoidSmall'] = False
                else:
                    self.infos['player_0']['shouldAvoidSmall'] = not self.subject_is_dominant
                    self.infos['player_0']['shouldAvoidBig'] = False

        if len(self.big_food_locations) > 0 and len(self.small_food_locations) > 0:
            if 'shouldAvoidBig' in self.infos['player_0'].keys() and self.infos['player_0']['shouldAvoidBig']:
                self.infos['player_0']['correctSelection'] = self.small_food_locations[-1]
                self.infos['player_0']['incorrectSelection'] = self.big_food_locations[-1]
            else:
                self.infos['player_0']['correctSelection'] = self.big_food_locations[-1]
                self.infos['player_0']['incorrectSelection'] = self.small_food_locations[-1]
