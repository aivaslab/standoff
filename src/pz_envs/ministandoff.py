import numpy as np

from ..base_AEC import para_MultiGridEnv, MultiGrid
from ..objects import Goal, Curtain, Block, Box
import random
from ..puppets import pathfind
import copy
from ..pz_envs.scenario_configs import ScenarioConfigs


def get_rewards(obj):
    if hasattr(obj, "contains") and obj.contains is not None:
        obj = obj.contains
    r = obj.reward if hasattr(obj, "reward") else 0
    sr = obj.sub_obs_reward if hasattr(obj, "sub_obs_reward") else 0
    return r, sr

def index_permutations(permutations, seed):
    result = [0] * len(permutations)
    for i in range(len(permutations)-1, -1, -1):
        seed, r = divmod(seed, permutations[i])
        result[i] = r
    return result

class MiniStandoffEnv(para_MultiGridEnv):
    mission = "get the best food before your opponent"
    metadata = {'render_modes': ['human', 'rgb_array'], "name": "miniStandoffEnv"}
    info_keywords = ('minibatch', 'timestep',
                     'shouldAvoidBig', 'shouldAvoidSmall', 'correct-loc', 'selection',
                     'selectedBig', 'selectedSmall', 'selectedNeither',
                     'selectedPrevBig', 'selectedPrevSmall', 'selectedPrevNeither', 'incorrect-loc',
                     'selectedSame', 'firstBaitReward', 'eventVisibility')

    def __init__(
            self,
            agents=None,
            puppets=None,
            grid_size=None,
            width=8,
            height=8,
            max_steps=100,
            reward_decay=False,
            seed=1337,
            respawn=False,
            ghost_mode=True,
            step_reward=-0.1,
            done_without_box_reward=-3,
            agent_spawn_kwargs=None,
            config_name="error",
            subject_visible_decs=False,
            opponent_visible_decs=False,
            persistent_treat_images=False,
            gaze_highlighting=False,
            persistent_gaze_highlighting=False,
            supervised_model=None,
            use_separate_reward_layers=True,
            conf=None,
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
                         done_without_box_reward,
                         agent_spawn_kwargs,
                         num_agents=len(agents),
                         num_puppets=len(puppets),
                         config_name=config_name,
                         subject_visible_decs=subject_visible_decs,
                         opponent_visible_decs=opponent_visible_decs,
                         persistent_treat_images=persistent_treat_images,
                         gaze_highlighting=gaze_highlighting,
                         persistent_gaze_highlighting=persistent_gaze_highlighting,
                         supervised_model=supervised_model,
                         use_separate_reward_layers=use_separate_reward_layers)
        self.treat_swapped = None
        self.treat_baited = None
        self.bait_loc = None
        self.swap_loc = None
        self.stop_on_release = False
        self.new_target = None
        if conf == None:
            print('found conf none')
            self.conf = ScenarioConfigs()
        else:
            self.conf = conf
        #configs = conf.standoff
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
        self.deterministic = False  # used for generating deterministic baiting events for ground-truth evaluation
        self.deterministic_seed = 0  # cycles baiting locations, changes by one each time it is used
        self.cached_paths = {}

        # difficulty settings
        self.subject_visible_decs = subject_visible_decs
        self.opponent_visible_decs = opponent_visible_decs
        self.persistent_treat_images = persistent_treat_images
        self.gaze_highlighting = gaze_highlighting
        self.persistent_gaze_highlighting = persistent_gaze_highlighting
        self.only_highlight_treats = True

        self.fill_spawn_holes = False
        self.random_subject_spawn = False # turned off for AAAI
        self.odd_spawns = True
        self.random_odd_spawns = True  # overrides self.odd_spawns when true
        self.record_oracle_labels = False

        self.supervised_model = supervised_model  # used for generating special supervised labels
        self.last_supervised_labels = None
        self.has_released = False
        self.was_ever_misinformed = {}

        self.param_groups = [
            {'eLists': self.conf.all_event_lists, 'params': self.conf.standoff['defaults'], 'perms': self.conf.all_event_permutations, 'delays': self.conf.all_event_delays},
        ]

    def reset_vision(self):
        """
        Reset the vision of the agents.
        """
        self.agent_goal, self.last_seen_reward, self.can_see, self.best_reward = {}, {}, {}, {}
        boxes = self.params["boxes"]
        self.saw_last_update = [True for _ in range(boxes)]

        self.treat_was_wrong = {agent: {treat_type: False for treat_type in ['big', 'small']} for agent in self.agents_and_puppets() + ['i']}

        # we add 'i' as an imaginary agent
        for agent in self.agents_and_puppets() + ['i']:
            self.agent_goal[agent] = 2 #self.deterministic_seed % boxes if self.deterministic else random.choice(range(boxes))
            self.best_reward[agent] = -100
            for box in range(boxes):
                self.last_seen_reward[agent + str(box)] = -1000
                if agent + str(box) not in self.can_see.keys():
                    self.can_see[agent + str(box)] = True  # default to not hidden until it is

    def get_deterministic_seed(self):
        # self.deterministic_seed += 1
        return self.deterministic_seed

    def _gen_grid(self,
                  sub_valence=1,
                  dom_valence=1,
                  num_puppets=1,
                  subject_is_dominant=False,
                  events=[],
                  hidden=False,
                  boxes=5,
                  perms=[],
                  delays=[]
                  ):

        #print(f"current_param_group_pos: {self.current_param_group_pos}")
        #print(f"delays: {delays}")
        #print(f"Final events after processing: {events}")

        startRoom = 1
        self.boxes = boxes
        if self.use_box_colors:
            self.box_color_order = list(range(5))
            random.shuffle(self.box_color_order)

        self.hidden = hidden
        self.subject_is_dominant = subject_is_dominant
        self.box_reward = 1
        self.released_tiles = [[] for _ in range(4)]
        self.curtain_tiles = []
        release_gap = 5
        self.width = 8
        self.height = 8
        self.grid = MultiGrid((self.width, self.height))
        self.small_food_locations = []
        self.big_food_locations = []

        self.objs_to_hide = []

        self.agent_spawn_pos = {}
        self.agent_door_pos = {}

        self.has_baited = False
        self.visible_event_list = []
        self.currently_visible = True

        all_door_poses = []

        self.bigReward = 100
        self.smallReward = int(self.bigReward / (self.boxes - 2))
        self.sub_valence = sub_valence

        self.end_at_frame = -1

        for k, agent in enumerate(self.agents_and_puppets()):
            h = 1 if agent == "p_0" else self.height - 1  # todo: make this work with dominance properly
            d = 1 if agent == "p_0" else 3
            if self.random_subject_spawn:
                bb = 1 + self.deterministic_seed % (self.boxes - 2) if self.deterministic else 1 + random.choice(
                    range(boxes - 2))
                xx = bb + 1
            else:
                xx = self.boxes // 2 + 1
            self.agent_spawn_pos[agent] = (xx, h, d)
            self.agent_door_pos[agent] = (xx, h + (1 if agent == "p_0" else -1))
            all_door_poses.append(self.agent_door_pos[agent])
            a = self.instance_from_name[agent]
            a.valence = sub_valence if agent == "p_0" else dom_valence
            if k > num_puppets:
                a.spawn_delay = 1000
                a.active = False

        self.grid.wall_rect(0, 0, self.width - 1, self.height)

        for box in range(boxes):
            # self.put_obj(Wall(), box + 1, startRoom - 1)
            xx_spawn = box + 1

            # initial door release, only where door is not in all_door_poses
            self.put_obj(Block(init_state=0, color="blue"), xx_spawn, startRoom + 1)
            self.released_tiles[1] += [(xx_spawn, startRoom + 1)]
            self.curtain_tiles += [(xx_spawn, startRoom + 0)]

            if (xx_spawn, startRoom + 1) not in all_door_poses:
                pass
            else:
                self.put_obj(Block(init_state=0, color="blue"), xx_spawn, startRoom + 1)
                self.released_tiles[1] += [(xx_spawn, startRoom + 1)]

            # same as above, for opponent
            self.put_obj(Block(init_state=0, color="blue"), xx_spawn, self.height - startRoom - 2)
            self.released_tiles[1] += [(xx_spawn, self.height - startRoom - 2)]
            self.curtain_tiles += [(xx_spawn, self.height - startRoom - 1)]
            if (xx_spawn, self.height - startRoom - 1) not in all_door_poses:
                # self.put_obj(Wall(), xx_spawn, self.height - startRoom - 1)
                pass
            else:
                self.put_obj(Block(init_state=0, color="blue"), xx_spawn, self.height - startRoom - 1)
                self.released_tiles[0] += [(xx_spawn, self.height - startRoom - 1)]

        self.reset_vision()

        self.box_updated_this_timestep = [False for _ in range(boxes)]
        ## Bucket location allocation for timers
        empty_buckets = [i for i in range(boxes)]
        event_args = [None for _ in range(len(events))]
        bait_args = [self.smallReward, self.bigReward]

        baited = False
        obscured = False
        counter = 1
        instantiated_perms = index_permutations(perms, (self.current_param_group_pos // len(self.delays)))
        self.infos['p_0']['perm'] = str(instantiated_perms)
        delay = list(delays)
        self.infos['p_0']['delay'] = delay + [0] * (4 - len(delay))
        for b in range(2):
            # store the bait/swap locations
            self.infos['p_0'][f'p-b-{b}'] = -1
            self.infos['p_0'][f'p-s-{b}'] = -1

        baits_so_far = 0
        swaps_so_far = 0
        for k, event in enumerate(events):
            event_type = event[0]

            if event_type == "random":
                event_list = ["b"]
                if baited:
                    event_list += ["sw"]
                if obscured:
                    event_list += ["re"]
                else:
                    event_list += ["ob"]
                event_type = random.choice(event_list)
                if event_type == "b":
                    event = ["b", random.randrange(boxes)]
                    baited = True
                    event_args[k] = random.choice(bait_args)
                elif event_type == "ob":
                    event = ["ob", "p_1"]
                    obscured = True
                    event_args[k] = event[1]
                elif event_type == "re":
                    event = ["re", "p_1"]
                    obscured = False
                    event_args[k] = event[1]
                elif event_type == "sw":
                    event = ["sw", random.randrange(boxes), random.randrange(boxes)]
                    event_args[k] = event[1:]
                events[k] = event

            else:
                was_empty = False
                for x in range(len(event)):
                    #print(event, empty_buckets)

                    if event[x] == "e":
                        was_empty = True
                        counter *= len(empty_buckets)
                        # used for both empty baits and swap locations
                        #print('popping', empty_buckets, k, instantiated_perms[k])
                        event[x] = float(empty_buckets.pop(random.randrange(
                            len(empty_buckets)) if not self.deterministic else instantiated_perms[k]))
                        if event_type == "b":
                            self.infos['p_0'][f'p-b-{baits_so_far}'] = event[x]
                        elif event_type == "sw":
                            self.infos['p_0'][f'p-s-{swaps_so_far}'] = event[x] # gets the popped value
                            if event[x] == 5:
                                print('5 found e', event)

                        # if we are swapping to an empty bucket, and the prev bucket was not empty, make it empty
                        if event[0] == 'sw' and x == 2:
                            if int(event[1]) not in empty_buckets:
                                #empty_buckets.append(int(event[1]))
                                # this line commented out on sept 30 to prevent coincidental 2nd swap to 1st loc
                                pass

                    elif event[x] == "else":
                        available_spots = [i for i in range(boxes) if i != event[x - 1]]
                        event[x] = available_spots.pop(random.randrange(
                            len(available_spots)) if not self.deterministic else instantiated_perms[k])

                        # if we are swapping to an empty bucket, and the prev bucket was not empty, make it empty
                        if event[0] == 'sw' and x == 2 and int(event[1]) not in empty_buckets and int(event[2]) in empty_buckets:
                            empty_buckets.append(int(event[1]))
                            empty_buckets.remove(int(event[2]))
                    elif isinstance(event[x], int): # integers are used for indices, and floats for locations
                        if event_type == "b" and x == 2:
                            # delayed 2nd bait: we bait at the swap "from" location
                            self.infos['p_0'][f'p-b-{baits_so_far}'] = events[event[x]][1]
                            event[x] = events[event[x]][1]
                        if event[0] != 'b':
                            #print(x, event[x], self.current_event_list_name, events, self.event_lists[self.current_event_list_name])
                            temp_event = events[event[x]][2]  # get a location from an index for first swap index number
                            #print(event[x])
                            if x == 2:
                                # Special case: If we are swapping to a previous swap index, make sure we don't reuse one bait index
                                if events[event[x]][0] == 'sw':
                                    if event[1] == events[event[x]][2]:
                                        temp_event = events[event[x]][1]
                            event[x] = float(temp_event)
                            if x == 2:
                                self.infos['p_0'][f'p-s-{swaps_so_far}'] = event[x]

                    #print('result', event, empty_buckets)

                if event_type == "b":
                    event_args[k] = bait_args[int(event[1])]  # get the size of the treat
                    # remove empty bucket for cases where event[x] wasn't e
                    if not was_empty:
                        #print(event, empty_buckets, instantiated_perms, k, instantiated_perms[k], len(empty_buckets))
                        #self.infos['p_0'][f'p-b-{baits_so_far}'] = instantiated_perms[k]
                        empty_buckets.pop(instantiated_perms[k])
                    baits_so_far += 1
                elif event_type == "rem":
                    empty_buckets.append(event[1])
                elif event_type == "ob" or event_type == "re":
                    # hardcoded, will not work for multiple conspecifics
                    event_args[k] = "p_1"
                elif event_type == 'sw':
                    #self.infos['p_0'][f'p-s-{swaps_so_far}'] = instantiated_perms[k]
                    swaps_so_far += 1

                # could also add functionality for moving empty buckets which are swapped, but that is not used in any tasks

        # add timers for events
        self.timers = {}
        curTime = 0
        self.add_timer(["init"], 0)
        delay_position = 0
        for k, event in enumerate(events):
            self.add_timer(event, curTime, arg=event_args[k])
            if event[0] not in ['b', 'sw']:
                #curTime += 1 # obscure/reveal events no longer take time
                pass
            else:
                curTime += delays[delay_position]
                delay_position += 1
        self.add_timer(["rel"], curTime, arg=0)
        self.add_timer(["rel"], curTime + release_gap + 1, arg=1)

        # returns the final timer, the release of the sub, could be made dynamic so just max of timers
        return curTime + release_gap

    def append_food_locs(self, obj, loc):
        if hasattr(obj, "reward") and obj.reward == self.bigReward:
            if (len(self.big_food_locations) == 0) or (self.big_food_locations[-1] != loc):
                self.big_food_locations.append(loc)
        elif hasattr(obj, "reward") and obj.reward == self.smallReward:
            if (len(self.small_food_locations) == 0) or (self.small_food_locations[-1] != loc):
                self.small_food_locations.append(loc)
        elif hasattr(obj, "contains") and hasattr(obj.contains, "reward"):
            if obj.contains.reward == self.bigReward:
                if (len(self.big_food_locations) == 0) or (self.big_food_locations[-1] != loc):
                    self.big_food_locations.append(loc)
            elif obj.contains.reward == self.smallReward:
                if (len(self.small_food_locations) == 0) or (self.small_food_locations[-1] != loc):
                    self.small_food_locations.append(loc)

    def get_all_paths(self, maze, position, offset=0):
        maze = np.array(maze).astype(int)

        _y = self.height // 2 + offset
        if self.subject_is_dominant:
            _y -= 1
        paths = []
        for box in range(self.boxes):
            x = box + 1

            pgrid = copy.copy(maze)
            for _x in range(self.width):
                if _x != x:
                    pgrid[_x, _y] = True
                    pgrid[_x, _y + 1] = True

            path = pathfind(pgrid, tuple(position[0:2]), (x, _y), self.cached_paths)
            paths += [path, ]
        return paths

    def timer_active(self, event, arg=None):
        name = event[0]
        arg = arg
        y = self.height // 2
        if self.subject_is_dominant:
            y -= 1

        
        if name == 'init':
            for box in range(self.boxes):
                x = box + 1
                if self.use_box_colors:
                    self.put_obj(Box(color=self.color_list[self.box_color_order[box]], state=self.box_color_order[box]),
                                 x, y)
                else:
                    self.put_obj(Box(color="orange"), x, y)
            self.vision_span_start = 0
            if self.record_info:
                self.infos['p_0']['eName'] = self.current_event_list_name
                self.infos['p_0']['shouldAvoidBig'] = False
                self.infos['p_0']['shouldAvoidSmall'] = False
                self.infos['p_0']['shouldGetBig'] = False
                self.infos['p_0']['shouldGetSmall'] = False
                self.infos['p_0']['correct-loc'] = -1
                self.infos['p_0']['incorrect-loc'] = -1
                self.infos['p_0']['minibatch'] = self.minibatch
                self.infos['p_0']['timestep'] = self.total_step_count
                self.infos['p_0']['gettier_big'] = False
                self.infos['p_0']['gettier_small'] = False
        elif name == 'b':
            x = event[2] + 1
            loc = int(event[2])
            if self.sub_valence == 1:
                sub_obs_reward = arg
            else:
                sub_obs_reward = 100 if arg == self.smallReward else 33
            obj = Goal(reward=arg, size=arg * 0.01, color='green', hide=False, sub_obs_reward=sub_obs_reward)
            if not self.has_baited:
                self.has_baited = True
                if self.record_info:
                    self.infos['p_0']['firstBaitReward'] = arg
            self.put_obj(obj, int(x), y)
            self.objs_to_hide.append(obj)
            self.box_updated_this_timestep[loc] = True
            self.bait_loc = [int(box == loc) for box in range(self.boxes)]
            self.treat_box[int(self.box_locations[loc])] = \
                [1, 0] if arg == self.bigReward else \
                [0, 1] if arg == self.smallReward else \
                [0, 0]

            if self.currently_visible:
                self.i_b_treat_box[int(self.i_b_box_locations[loc])] = \
                    [1, 0] if arg == self.bigReward else \
                    [0, 1] if arg == self.smallReward else \
                    [0, 0]
                if self.params['num_puppets'] > 0:
                    self.b_treat_box[int(self.b_box_locations[loc])] = \
                        [1, 0] if arg == self.bigReward else \
                        [0, 1] if arg == self.smallReward else \
                        [0, 0]

            self.treat_baited = [arg == self.smallReward, arg == self.bigReward]
        elif name == "rem":
            x = event[1] + 1
            tile = self.grid.get(x, y)
            if tile is not None:
                if hasattr(tile, "reward") and tile.reward == self.bigReward and self.currently_visible:
                    self.big_food_locations.append(-1)
                elif hasattr(tile, "reward") and tile.reward == self.smallReward:
                    self.small_food_locations.append(-1)
                    # this is a special case of removedUninformed1 where there is no correct solution.
            self.del_obj(x, y)
            self.box_updated_this_timestep[int(event[1])] = True
        elif name == "ob" or name == "re":
            #b = self.grid.get(*self.agent_door_pos[arg]) if arg in self.agent_door_pos.keys() else self.grid.get(3, self.height-3) # default for obscuring when no opponent
            # above line was used for hiding all doors
            for box in range(self.boxes):
                b = self.grid.get(box + 1, self.height - 3)
                if name == "ob":
                    b.state = 1
                    b.see_behind = lambda: False
                    self.currently_visible = False
                    self.vision_span_start = 1000 # the last vision span starts never
                elif name == "re":
                    b.state = 0
                    b.see_behind = lambda: True
                    self.currently_visible = True
                    self.vision_span_start = self.step_count
            for box in range(self.boxes):
                self.can_see[arg + str(box)] = False if "ob" in name else True
                self.can_see['i' + str(box)] = False if "ob" in name else True

                x = box + 1
                tile = self.grid.get(x, y)
                if tile is not None and tile.type == "Goal":
                    # mark for hiding at the next timer step
                    self.objs_to_hide.append(tile)
                    self.box_updated_this_timestep[box] = True
        elif name == "sw":
            #print(event[1], event[2], 'should be floats!')
            e1 = int(event[1])
            e2 = int(event[2])
            b1 = self.grid.get(e1 + 1, y)
            b2 = self.grid.get(e2 + 1, y)
            self.box_updated_this_timestep[e1] = True
            self.box_updated_this_timestep[e2] = True

            self.swap_loc = [int(x in [e1, e2]) for x in range(self.boxes)]

            self.box_locations[e1], self.box_locations[e2] = self.box_locations[e2], self.box_locations[e1]
            if self.currently_visible:
                self.i_b_box_locations[e1], self.i_b_box_locations[e2] = self.i_b_box_locations[e2], self.i_b_box_locations[e1]
                if self.params['num_puppets'] > 0:
                    self.b_box_locations[e1], self.b_box_locations[e2] = self.b_box_locations[e2], self.b_box_locations[e1]

            if self.use_box_colors:
                self.put_obj(b2, e1 + 1, y)
                self.put_obj(b1, e2 + 1, y)
                temp = self.box_color_order[e1]
                self.box_color_order[e1] = self.box_color_order[e2]
                self.box_color_order[e2] = temp
            else:
                # November 24 changed to check for containers and delete old objects correctly instead of just removing reference

                r1, sr1 = get_rewards(b1)
                r2, sr2 = get_rewards(b2)

                self.del_obj(e1 + 1, y)
                self.del_obj(e2 + 1, y)

                obj1 = Goal(reward=r2, size=r2 * 0.01, color='green', hide=False, sub_obs_reward=sr2)
                obj2 = Goal(reward=r1, size=r1 * 0.01, color='green', hide=False, sub_obs_reward=sr1)
                self.treat_swapped = [self.smallReward in [r1, r2], self.bigReward in [r1, r2]]
                self.put_obj(obj1, e1 + 1, y)
                self.put_obj(obj2, e2 + 1, y)
                self.objs_to_hide.append(obj1)
                self.objs_to_hide.append(obj2)
        elif name == "rel":
            self.infos['p_0']['last-vision-span'] = [int(x >= self.vision_span_start) for x in range(5)]

            if self.record_info:
                self.infos['p_0']['eventVisibility'] = ''.join(
                    ['1' if x else '0' for x in self.visible_event_list])
                #print(self.infos['p_0']['eventVisibility'], self.infos['p_0']['last-vision-span'])
            for x, _y in self.released_tiles[arg]:
                self.del_obj(x, _y)
            if arg == 0:
                for x, _y in self.curtain_tiles:
                    self.put_obj(Curtain(color='red'), x, _y, update_vis=False)
            if self.stop_on_release is True:
                self.dones['p_0'] = True
            self.has_released = True

        self.visible_event_list.append(self.currently_visible)

        for loc in range(self.boxes):
            x = loc + 1
            obj = self.grid.get(x, y)
            self.append_food_locs(obj, loc)  # appends to self.big_food_locations and self.small_food_locations

        # oracle food location memory for puppet ai
        target_agents = []
        if name == "b" or name == "sw" or name == "rem" or (self.hidden is True and name == "re"):
            for key, value in self.last_seen_reward.items():
                self.last_seen_reward[key] = value - 1 # we discount older rewards to prefer new updates
            did_swap = False
            for box in range(self.boxes):
                x = box + 1
                for agent in self.agents_and_puppets() + ['i']:
                    if self.can_see[agent + str(box)] and self.currently_visible:
                        tile = self.grid.get(x, y)
                        # if hasattr(tile, "reward") and hasattr(tile, "size"):
                        if tile is not None and tile.type == "Goal":
                            if name == "sw":
                                if not did_swap:
                                    did_swap = True
                                    b1 = agent + str(int(event[1]))
                                    b2 = agent + str(int(event[2]))
                                    temp = self.last_seen_reward[b1]
                                    self.last_seen_reward[b1] = self.last_seen_reward[b2]
                                    self.last_seen_reward[b2] = temp

                            if hasattr(tile, 'get_reward'):
                                self.last_seen_reward[agent + str(box)] = tile.get_reward()


            self.new_target = False
            for box in range(self.boxes):
                for agent in self.puppets + ['i']:
                    reward = self.last_seen_reward[agent + str(box)]
                    if reward >= self.best_reward[agent]:
                        self.agent_goal[agent] = box
                        self.best_reward[agent] = reward
                        self.new_target = True
                        target_agents.append(agent)
                        #print('new target', self.best_reward[agent], self.agent_goal[agent], self.last_seen_reward)
        if self.new_target:
            self.new_target = False
            for target_agent in target_agents:
                if target_agent != 'i':
                    a = self.instance_from_name[target_agent]
                    if a.active:
                        x = self.agent_goal[target_agent] + 1
                        pgrid = self.grid.volatile
                        for _x in range(self.width):
                            if _x != x:
                                pgrid[_x, y] = True
                                pgrid[_x, y + 1] = True
                        path = pathfind(pgrid, a.pos, (x, y), self.cached_paths)
                        # print('path', path)
                        self.infos[target_agent]["path"] = path
                # tile = self.grid.get(x, y)
                # we cannot track shouldAvoidBig etc here because the treat location might change

            #print(self.step_count, self.infos['p_0']["exist"], self.infos['p_0']["b-exist"], self.infos['p_0']["target"], self.infos['p_0']["vision"], self.infos['p_0']["loc"], self.infos['p_0']["b-loc"])

        

        if self.record_info:
            # if agent's goal of player_1 matches big treat location, then shouldAvoidBig is True

            info = self.infos['p_0']
            if len(self.puppets):
                info['puppet_goal'] = self.agent_goal[self.puppets[-1]]
                if len(self.big_food_locations) > 0 and info['puppet_goal'] == self.big_food_locations[-1]:
                    info['shouldAvoidBig'] = not self.subject_is_dominant
                    info['shouldGetBig'] = False
                    info['shouldGetSmall'] = True
                    info['shouldAvoidSmall'] = False
                elif len(self.small_food_locations) > 0 and info['puppet_goal'] == self.small_food_locations[-1]:
                    info['shouldAvoidSmall'] = not self.subject_is_dominant
                    info['shouldAvoidBig'] = False
                    info['shouldGetBig'] = True
                    info['shouldGetSmall'] = False
                else:
                    info['shouldAvoidBig'] = False
                    info['shouldAvoidSmall'] = False
                    info['shouldGetBig'] = True
                    info['shouldGetSmall'] = False

            if len(self.big_food_locations) > 0 and len(self.small_food_locations) > 0:
                if 'shouldAvoidBig' in self.infos['p_0'].keys() and self.infos['p_0']['shouldAvoidBig']:
                    info['correct-loc'] = self.small_food_locations[-1]
                    info['incorrect-loc'] = self.big_food_locations[-1]
                    info['shouldGetSmall'] = True
                    info['shouldGetBig'] = False
                else:
                    info['correct-loc'] = self.big_food_locations[-1]
                    info['incorrect-loc'] = self.small_food_locations[-1]
                    info['shouldGetBig'] = True
                    info['shouldGetSmall'] = False
            elif len(self.small_food_locations) > 0:
                if not self.infos['p_0']['shouldAvoidSmall']:
                    info['correct-loc'] = self.small_food_locations[-1]
                    info['shouldGetSmall'] = True
                    info['shouldGetBig'] = False
                    info['incorrect-loc'] = -1
                else:
                    assert self.step_count < 4 # this shouldn't happen ever
                    info['correct-loc'] = -1
                    info['shouldGetBig'] = True
                    info['shouldGetSmall'] = False
                    info['incorrect-loc'] = self.small_food_locations[-1]
            elif len(self.big_food_locations) > 0:
                if not self.infos['p_0']['shouldAvoidBig']:
                    info['correct-loc'] = self.big_food_locations[-1]
                    info['shouldGetBig'] = True
                    info['shouldGetSmall'] = False
                    info['incorrect-loc'] = -1
                else:
                    assert self.step_count < 4 # this shouldn't happen ever
                    #print('error: step count was', self.step_count)
                    info['correct-loc'] = -1
                    info['shouldGetSmall'] = True
                    info['shouldGetBig'] = False
                    info['incorrect-loc'] = self.big_food_locations[-1]

            if info['correct-loc'] == -1 and self.step_count > 0:
                #print('was -1')
                #print(self.infos['p_0'], self.step_count)
                #print(self.big_food_locations, self.small_food_locations)
                pass

            if len(self.puppets) == 0:
                if info['shouldGetBig'] == False and self.step_count == 4:
                    print('found issue', info)
