# Multi-agent gridworld.
# Based on MiniGrid: https://github.com/maximecb/gym-minigrid.
from __future__ import annotations

import copy

import gym
import numpy as np
import functools
import random
# import traceback

from .objects import Wall, Goal, Lava, GridAgent, COLORS
from gym_minigrid.rendering import downsample
from gym.utils.seeding import np_random
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from .agents import occlude_mask
# import hashlib
import xxhash

from src.rendering import SimpleImageViewer
from .utils.conversion import calculate_informedness

# from gym.envs.classic_control.rendering import SimpleImageViewer

TILE_PIXELS = 9
NUM_ITERS = 100
NONE = 4

def distribute_weights_to_second_largest(arr):
    unique_values = np.unique(arr)

    if len(unique_values) > 1:
        second_largest = np.sort(unique_values)[-2]
        weights = (arr == second_largest).astype(float)
    else:
        weights = np.ones_like(arr, dtype=float)
        weights[2] = 0  # Set the weight at index 2 to zero

    total_weight = np.sum(weights)
    if total_weight > 0:
        weights /= total_weight  # Normalize weights to sum to 1

    return weights


class ObjectRegistry:
    """
    This class contains dicts that map objects to numeric keys and vise versa.
    Used so that grid worlds can represent objects using numerical arrays rather
        than lists of lists of generic objects.
    """

    def __init__(self, objs=None, max_num_objects=1000):
        if objs is None:
            objs = []
        self.key_to_obj_map = {}
        self.obj_to_key_map = {}
        self.max_num_objects = max_num_objects
        for obj in objs:
            self.add_object(obj)

    def get_next_key(self):
        for k in range(self.max_num_objects):
            if k not in self.key_to_obj_map:
                break
        else:
            raise ValueError("Object registry full.")
        return k

    def __len__(self):
        return len(self.key_to_obj_map)

    def add_object(self, obj):
        new_key = self.get_next_key()
        self.key_to_obj_map[new_key] = obj
        self.obj_to_key_map[obj] = new_key
        return new_key

    def contains_object(self, obj):
        return obj in self.obj_to_key_map

    def contains_key(self, key):
        return key in self.key_to_obj_map

    def get_key(self, obj):
        if obj in self.obj_to_key_map:
            return self.obj_to_key_map[obj]
        else:
            return self.add_object(obj)

    # 5/4/2020 This gets called A LOT. Replaced calls to this function with direct dict gets
    #           in an attempt to speed things up. Probably didn't make a big difference.
    def obj_of_key(self, key):
        return self.key_to_obj_map[key]


def rotate_grid(grid, rot_k):
    """
    This function basically replicates np.rot90 (with the correct args for rotating images).
    But it's faster.
    """
    rot_k = rot_k % 4
    if rot_k == 3:
        return np.moveaxis(grid[:, ::-1], 0, 1)
    elif rot_k == 1:
        return np.moveaxis(grid[::-1, :], 0, 1)
    elif rot_k == 2:
        return grid[::-1, ::-1]
    else:
        return grid

def convert_to_scalar(loc_list):
    pattern_to_scalar = {
        (1, 1): 3,
        (1, 0): 2,
        (0, 1): 1,
        (0, 0): 0
    }
    scalar_list = [pattern_to_scalar[tuple(loc)] for loc in loc_list]
    return scalar_list


class MultiGrid:
    tile_cache = {}

    def __init__(self, shape=(11, 11), obj_reg=None, orientation=0):
        self.orientation = orientation
        if isinstance(shape, tuple):
            self.width, self.height = shape
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8)  # w,h
        elif isinstance(shape, np.ndarray):
            self.width, self.height = shape.shape
            self.grid = shape
        else:
            raise ValueError("Must create grid from shape tuple or array.")

        if self.width < 3 or self.height < 3:
            raise ValueError("Grid needs width, height >= 3")

        self.obj_reg = ObjectRegistry(objs=[None]) if obj_reg is None else obj_reg

    @property
    def opacity(self):
        transparent_fun = np.vectorize(lambda k: (
            self.obj_reg.key_to_obj_map[k].see_behind() if hasattr(self.obj_reg.key_to_obj_map[k],
                                                                   'see_behind') else True))
        return ~transparent_fun(self.grid)

    @property
    def overlapping(self):
        overlap_fun = np.vectorize(lambda k: (
            self.obj_reg.key_to_obj_map[k].can_overlap() if hasattr(self.obj_reg.key_to_obj_map[k],
                                                                    'can_overlap') else True))
        return ~overlap_fun(self.grid)

    @property
    def all_treats(self):
        treats = np.vectorize(lambda k: (
            self.obj_reg.key_to_obj_map[k].type == 'Goal' if hasattr(self.obj_reg.key_to_obj_map[k],
                                                                     'type') else False))
        return treats(self.grid)

    @property
    def all_rewards(self):
        rews = np.vectorize(lambda k: (
            self.obj_reg.key_to_obj_map[k].reward() if hasattr(self.obj_reg.key_to_obj_map[k],
                                                               'reward') else 0))
        return rews(self.grid)

    @property
    def all_boxes(self):
        boxes = np.vectorize(lambda k: (
            self.obj_reg.key_to_obj_map[k].type == 'Box' if hasattr(self.obj_reg.key_to_obj_map[k],
                                                                    'type') else False))
        return boxes(self.grid)

    @property
    def volatile(self):
        # refers to things which shall be removed from the grid after some time
        overlap_fun = np.vectorize(lambda k: (
            self.obj_reg.key_to_obj_map[k].volatile() if hasattr(self.obj_reg.key_to_obj_map[k],
                                                                 'volatile') else True))
        return ~overlap_fun(self.grid)

    def __getitem__(self, *args, **kwargs):
        return self.__class__(
            np.ndarray.__getitem__(self.grid, *args, **kwargs),
            obj_reg=self.obj_reg,
            orientation=self.orientation,
        )

    def rotate_left(self, k=1):
        return self.__class__(
            rotate_grid(self.grid, rot_k=k),  # np.rot90(self.grid, k=k),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - k) % 4,
        )

    def slice(self, topX, topY, width, height, rot_k=0):
        """
        Get a subset of the grid
        """
        sub_grid = self.__class__(
            (width, height),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - rot_k) % 4,
        )
        x_min = max(0, topX)
        x_max = min(topX + width, self.width)
        y_min = max(0, topY)
        y_max = min(topY + height, self.height)

        x_offset = x_min - topX
        y_offset = y_min - topY
        sub_grid.grid[
        x_offset: x_max - x_min + x_offset, y_offset: y_max - y_min + y_offset
        ] = self.grid[x_min:x_max, y_min:y_max]

        sub_grid.grid = rotate_grid(sub_grid.grid, rot_k)

        sub_grid.width, sub_grid.height = sub_grid.grid.shape

        return sub_grid

    def set(self, i, j, obj, update_vis_mask=None):
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        if update_vis_mask is not None:
            update_vis_mask[i, j] = False
        self.grid[i, j] = self.obj_reg.get_key(obj)

    def get(self, i: object, j: object) -> object:
        if 0 <= i < self.width:
            if 0 <= j < self.height:
                return self.obj_reg.key_to_obj_map[self.grid[i, j]]
        return -1

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type(), update_vis_mask=None)

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type(), update_vis_mask=None)

    def wall_rect(self, x, y, w, h, obj_type=Wall):
        self.horz_wall(x, y, w, obj_type=obj_type)
        self.horz_wall(x, y + h - 1, w, obj_type=obj_type)
        self.vert_wall(x, y, h, obj_type=obj_type)
        self.vert_wall(x + w - 1, y, h, obj_type=obj_type)

    def __str__(self):
        render = (
            lambda x: "  "
            if x is None or not hasattr(x, "str_render")
            else x.str_render(dir=self.orientation)
        )
        hstars = "*" * (2 * self.width + 2)
        return (
                hstars
                + "\n"
                + "\n".join(
            "*" + "".join(render(self.get(i, j)) for i in range(self.width)) + "*"
            for j in range(self.height)
        )
                + "\n"
                + hstars
        )

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype="uint8")  # was bool

        array = np.zeros((self.width, self.height, 3), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        array[i, j, :] = 0
                    else:
                        array[i, j, :] = v.encode()
        return array

    @classmethod
    def decode(cls, array):
        """width, height, channels = array.shape
        assert channels == 3
        vis_mask[i, j] = np.ones(shape=(width, height), dtype="uint8")  # was bool
        grid = cls((width, height))"""
        raise NotImplementedError
        pass

    @classmethod
    def cache_render_fun(cls, key, f, *args, **kwargs):
        if key not in cls.tile_cache:
            cls.tile_cache[key] = f(*args, **kwargs)
        return np.copy(cls.tile_cache[key])

    @classmethod
    def cache_render_obj(cls, obj, tile_size, subdivs):
        if obj is None:
            return cls.cache_render_fun((tile_size, None), cls.empty_tile, tile_size, subdivs)
        else:
            if obj.type == 'Agent' and obj.carrying is not None:
                img = cls.cache_render_fun(
                    (tile_size, obj.__class__.__name__ + obj.carrying.__class__.__name__, *obj.encode()),
                    cls.render_object, obj, tile_size, subdivs
                )
            else:
                if obj.type == 'Box' and obj.show_contains is True and obj.contains is not None:
                    img = cls.cache_render_fun(
                        (tile_size, obj.__class__.__name__ + str(obj.contains.reward), *obj.encode()),
                        cls.render_object, obj, tile_size, subdivs
                    )
                else:
                    img = cls.cache_render_fun(
                        (tile_size, obj.__class__.__name__ + str(obj.size), *obj.encode()),
                        cls.render_object, obj, tile_size, subdivs
                    )
            if hasattr(obj, 'render_post'):
                return obj.render_post(img)
            else:
                return img

    @classmethod
    def empty_tile(cls, tile_size, subdivs):
        alpha = max(0, min(20, tile_size - 10))
        img = np.full((tile_size, tile_size, 3), alpha, dtype=np.uint8)
        img[1:, :-1] = 0
        return img

    @classmethod
    def render_object(cls, obj, tile_size, subdivs):
        img = np.zeros((tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)
        obj.render(img)
        # if 'Agent' not in obj.type and len(obj.agents) > 0:
        #     obj.agents[0].render(img)
        return downsample(img, subdivs).astype(np.uint8)

    @classmethod
    def blend_tiles(cls, img1, img2):
        '''
        This function renders one "tile" on top of another. Kinda janky, works surprisingly well.
        Assumes img2 is a downscaled monochromatic with a black (0,0,0) background.
        '''
        alpha = img2.sum(2, keepdims=True)
        max_alpha = alpha.max()
        if max_alpha == 0:
            return img1
        return (
                ((img1 * (max_alpha - alpha)) + (img2 * alpha)
                 ) / max_alpha
        ).astype(img1.dtype)

    @classmethod
    def render_tile(cls, obj,
                    tile_size: int = TILE_PIXELS,
                    subdivs: int = 3,
                    top_agent=None):

        if obj is None:
            img = cls.cache_render_obj(obj, tile_size, subdivs)
        else:
            if ('Agent' in obj.type) and (top_agent in obj.agents):
                # If the tile is a stack of agents that includes the top agent, then just render the top agent.
                img = cls.cache_render_obj(top_agent, tile_size, subdivs)
            else:
                # Otherwise, render (+ downsize) the item in the tile.
                img = cls.cache_render_obj(obj, tile_size, subdivs)
                # If the base obj isn't an agent but has agents on top, render an agent and blend it in.
                if len(obj.agents) > 0 and 'Agent' not in obj.type:
                    if top_agent in obj.agents:
                        img_agent = cls.cache_render_obj(top_agent, tile_size, subdivs)
                    else:
                        img_agent = cls.cache_render_obj(obj.agents[0], tile_size, subdivs)
                    img = cls.blend_tiles(img_agent, img)

            # Render the tile border if any of the corners are black.
            # Removed for speed
            '''if (img[([0, 0, -1, -1], [0, -1, 0, -1])] == 0).all(axis=-1).any():
                img = img + cls.cache_render_fun((tile_size, None), cls.empty_tile, tile_size, subdivs)'''
        return img

    def render(self, tile_size, highlight_mask=None, visible_mask=None, top_agent=None, gaze_highlight_mask=None):
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)  # [..., None] + COLORS['shadow']
        img[:] = COLORS['shadow']

        for j in range(0, self.height):
            for i in range(0, self.width):
                if visible_mask is not None and not visible_mask[i, j]:
                    continue
                obj = self.get(i, j)

                tile_img = MultiGrid.render_tile(
                    obj,
                    tile_size=tile_size,
                    top_agent=top_agent,
                    subdivs=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size

                img[ymin:ymax, xmin:xmax, :] = rotate_grid(tile_img, self.orientation)

                c = COLORS['red']
                if gaze_highlight_mask is not None and gaze_highlight_mask[i, j]:
                    img[ymin:ymax, xmin, :] = c
                    img[ymin:ymax, xmax - 1, :] = c
                    img[ymin, xmin:xmax, :] = c
                    img[ymax - 1, xmin:xmax, :] = c

        if highlight_mask is not None:
            hm = np.kron(highlight_mask.T, np.full((tile_size, tile_size), 255, dtype=np.uint8)
                         )[..., None]  # arcane magic.
            img = np.right_shift(img.astype(np.uint16) * 8 + hm * 2, 3).clip(0, 255).astype(np.uint8)

        img = img.astype(np.uint8)
        return img


# noinspection PyTypeChecker
class para_MultiGridEnv(ParallelEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {'render.modes': ['human'], "name": "multigrid_alpha"}

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
            step_reward=0.01,
            done_without_box_reward=-10,
            agent_spawn_kwargs=None,
            num_agents=1,
            num_puppets=0,
            config_name='null',
            subject_visible_decs=False,
            opponent_visible_decs=False,
            persistent_treat_images=False,
            gaze_highlighting=False,
            persistent_gaze_highlighting=False,
            observation_style='rich',
            dense_obs=True,
            supervised_model=None,
            use_separate_reward_layers=False,
    ):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """

        self.end_at_frame = None
        self.agent_goal = None
        self.boxes = None
        self.saw_last_update = None
        self.currently_visible = None
        self.last_seen_reward = None
        self.smallReward = None
        self.bigReward = None
        self.box_updated_this_timestep = None
        self.record_oracle_labels = False
        self.supervised_model = supervised_model
        self.past_observations = None
        self.record_info = False  # set this to true during evaluation
        self.max_steps_real = None
        self.only_highlight_treats = False
        self.gaze_highlighting = gaze_highlighting
        self.persistent_gaze_highlighting = persistent_gaze_highlighting
        self.prev_puppet_mask = None  # used for persistent gaze

        self.agent_spawn_pos = None
        self.params = None
        if agents is None:
            agents = []
        if agent_spawn_kwargs is None:
            agent_spawn_kwargs = {}
        self.rewards = None
        self.np_random = None
        if grid_size is not None:
            assert width is None and height is None
            width, height = grid_size, grid_size

        self.respawn = respawn

        self.config_name = config_name

        self.window = None

        self.timers = {}

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reward_decay = reward_decay
        self.step_reward = step_reward  # 0
        self.done_without_box_reward = done_without_box_reward  # -10
        self.distance_from_boxes_reward = -0.5  # increase done penalty based on distance to goals
        self.penalize_same_selection = -5.0  # penalty for selecting the same box as the opponent
        self.seed(seed=seed)
        self.agent_spawn_kwargs = agent_spawn_kwargs
        self.ghost_mode = ghost_mode
        self.agent_view_size = agents[0].view_size * agents[0].view_tile_size  # 34 magic number fix?

        self.agents = agents
        self.puppets = puppets
        self.grid = MultiGrid(shape=(width, height))  # added this, not sure where grid comes from in og

        self.possible_agents = [f"p_{r}" for r in range(num_agents)]
        if num_puppets > 0:
            self.possible_puppets = [f"p_{r + num_agents}" for r in range(num_puppets)]
        else:
            self.possible_puppets = []

        self.action_spaces = {agent: Discrete(4) for agent in self.possible_agents}
        # 7 for additional things

        self.env_done = False
        self.step_count = 0

        self.minibatch = 0
        self.total_step_count = 0

        self.observation_style = observation_style
        self.dense_obs = dense_obs
        self.channels = 3
        self.use_box_colors = False
        self.use_separate_reward_layers = use_separate_reward_layers

        if self.use_box_colors:
            self.color_list = list(COLORS)

        if self.observation_style == 'rich':
            self.rich_observation_layers = [
                lambda k, mapping: (((mapping[k].type == 'Agent') + mapping[k].valence) if hasattr(mapping[k], 'valence') else False),
                lambda k, mapping: (mapping[k].type == 'Box' * mapping[k].state if hasattr(mapping[k], 'type') else False) if self.use_box_colors
                else (mapping[k].type == 'Box' if hasattr(mapping[k], 'type') else False),
            ]
            if self.use_separate_reward_layers:
                # big and small refer to the numbers shown here, which are always what the opponent sees
                # if sub_valence is not 1, the observation changes, but not the object reward states
                # so sub_observed_reward shows the post-valence change reward
                self.rich_observation_layers.extend([
                    lambda k, mapping: (mapping[k].get_sub_obs_reward() == 100 if hasattr(mapping[k], 'get_sub_obs_reward') and mapping[k].contains is None else 0),
                    lambda k, mapping: (mapping[k].get_sub_obs_reward() == 33 if hasattr(mapping[k], 'get_sub_obs_reward') and mapping[k].contains is None else 0),
                ])
            else:
                self.rich_observation_layers.extend([
                    lambda k, mapping: (mapping[k].get_reward() if hasattr(mapping[k], 'get_reward') and mapping[k].contains is None else 0)
                ])

            # handle walls, etc
            if self.dense_obs is False:
                self.rich_observation_layers.extend([
                    lambda k, mapping: (mapping[k].can_overlap() if hasattr(mapping[k], 'can_overlap') else 1),
                    lambda k, mapping: (mapping[k].volatile() if hasattr(mapping[k], 'volatile') else 0),
                    lambda k, mapping: (mapping[k].see_behind() if hasattr(mapping[k], 'see_behind') else 1),
                ])
            else:
                self.rich_observation_layers.append(
                    lambda k, mapping: (
                            4 * (mapping[k].can_overlap() if hasattr(mapping[k], 'can_overlap') else 1) +
                            2 * (mapping[k].see_behind() if hasattr(mapping[k], 'see_behind') else 1) +
                            1 * (mapping[k].volatile() if hasattr(mapping[k], 'volatile') else 0)
                    ),
                )
            if self.gaze_highlighting:
                self.rich_observation_layers.append('gaze')
            self.channels = len(self.rich_observation_layers) + (1 if self.supervised_model is not None else 0)

        self.observation_spaces = {agent: Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, self.channels),
            dtype='uint8'
        ) for agent in self.possible_agents}

        # self.action_space = self.action_spaces[self.possible_agents[0]]
        # self.observation_space = self.observation_spaces[self.possible_agents[0]]
        # cannot define these because it makes uncallable

        self.agent_instances = [agent for agent in agents]
        self.puppet_instances = [puppet for puppet in puppets]
        self.instance_from_name = {name: agent for name, agent in
                                   zip(self.possible_agents + self.possible_puppets, agents + puppets)}
        self.loadingPickle = False
        self.allRooms = []

        self.last_supervised_labels = None
        self.supervised_label_dict = {}
        self.has_released = False

        self.current_param_group = 0
        self.current_param_group_pos = 0
        self.target_param_group_count = -1
        self.current_event_list = None
        self.current_event_list_name = None
        self.param_groups = []
        self.event_lists = None
        self.has_reset = False

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def check_agent_position_integrity(self, title=''):
        '''
        This function checks whether each agent is present in the grid in exactly one place.
        This is particularly helpful for validating the world state when ghost_mode=False and
        agents can stack, since the logic for moving them around gets a bit messy.
        Prints a message and drops into pdb if there's an inconsistency.
        '''
        agent_locs = [[] for _ in range(len(self.agents + self.puppets))]
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                x = self.grid.get(i, j)
                for k, agent in enumerate(self.agents + self.puppets):
                    if x == agent:
                        agent_locs[k].append(('top', (i, j)))
                    if hasattr(x, 'agents') and agent in x.agents:
                        agent_locs[k].append(('stacked', (i, j)))
        if not all([len(x) == 1 for x in agent_locs]):
            print(f"{title} > Failed integrity test!")
            for a, al in zip(self.agents, agent_locs):
                print(" > ", a.color, '-', al)
            import pdb
            pdb.set_trace()

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def agents_and_puppets(self):
        """
        For legacy agent functions to also work with puppets
        """
        return self.agents + self.puppets

    def agent_and_puppet_instances(self):
        """
        For legacy agent functions to also work with puppets
        """
        return self.agent_instances + self.puppet_instances[:self.params['num_puppets']]

    def reset(self, override_params=-1):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        if self.target_param_group_count > -1 or not self.has_reset or override_params > -1:
            if (self.current_param_group_pos + 1 >= self.target_param_group_count) or not self.has_reset or override_params > -1:
                if override_params > -1:
                    print('overriding param group')
                    self.current_param_group = override_params
                elif self.has_reset:
                    self.current_param_group = (self.current_param_group + 1) % len(self.param_groups)
                else:
                    self.current_param_group = 0
                self.current_param_group_pos = 0
                self.event_lists = self.param_groups[self.current_param_group]['eLists']
                self.params = self.param_groups[self.current_param_group]['params']
                self.perms = list(self.param_groups[self.current_param_group]['perms'].items())[0][1]
                self.delays = list(self.param_groups[self.current_param_group]['delays'].items())[0][1]

            else:
                self.current_param_group_pos += 1

            self.has_reset = True

            # deal with num_puppets being lower than maximum
            self.possible_puppets = [f"p_{x + len(self.agents)}" for x in range(self.params["num_puppets"])]

        # get event list and perm list from event lists
        if self.params['deterministic']:
            self.current_event_list_name, self.params['events'] = copy.deepcopy(list(self.event_lists.items())[self.current_param_group_pos % len(self.event_lists)])
            self.params['perms'] = self.perms
            self.params['delays'] = self.delays[(self.current_param_group_pos) % len(self.delays)]
            self.target_param_group_count = np.product(self.params['perms'])*len(self.delays)
        else:
            self.current_event_list_name, self.params['events'] = copy.deepcopy(random.choice(list(self.event_lists.items())))


        self.agents = self.possible_agents[:]
        self.puppets = self.possible_puppets[:]
        self.rewards = {agent: 0 for agent in self.agents_and_puppets()}
        self._cumulative_rewards = {agent: 0 for agent in self.agents_and_puppets()}
        self.has_reached_goal = {agent: False for agent in self.agents_and_puppets()}
        self.dones = {agent: False for agent in self.agents_and_puppets()}
        self.infos = {agent: {} for agent in self.agents_and_puppets()}
        if self.record_info:
            for key in self.info_keywords:
                self.infos['p_0'][key] = ''
        self.state = {agent: NONE for agent in self.agents_and_puppets()}
        self.treat_box = [[0, 0] for _ in range(5)]
        self.b_treat_box = [[0, 0] for _ in range(5)]
        self.i_b_treat_box = [[0, 0] for _ in range(5)]

        self.box_locations = [i for i in range(5)]
        self.b_box_locations = [i for i in range(5)]
        self.i_b_box_locations = [i for i in range(5)]
        # we don't generate observations for puppets

        # self.observations = {agent: self.gen_agent_obs(a) for agent, a in zip(self.agents_and_puppets(), self.agent_and_puppet_instances())}

        self.total_step_count += self.step_count
        self.step_count = 0
        self.env_done = False
        self.previously_selected_boxes = []

        for name, agent in zip(self.agents + self.puppets, list(self.agent_and_puppet_instances())):
            agent.agents = []
            agent.name = name
            agent.next_actions = []
            agent.pathDict = {}
            self.instance_from_name[name] = agent
            agent.reset(new_episode=True)

        if self.loadingPickle:
            self.grid = random.choice(self.allRooms)
            last_timer = 40
        else:
            valid_params = ['sub_valence', 'dom_valence', 'num_puppets', 'subject_is_dominant', 'events',
                            'hidden', 'boxes', 'perms', 'delays']
            #print('generating grid with events:', self.deterministic_seed, self.current_event_list_name, self.params['events'])
            last_timer = self._gen_grid(**{x: self.params[x] for x in valid_params})
            # gen_grid also generates timers
            self.prev_puppet_mask = np.zeros((self.grid.height, self.grid.width))
            '''flag = 0
            while flag < 100:
                try:
                    self._gen_grid(self.width, self.height, **self.params)
                    flag = 100
                except Exception as e:
                    flag = flag + 1
                    if flag == 100 or True:
                        print('exception', e)
                        traceback.print_exc()
                    pass'''
        self.max_steps_real = min(self.max_steps, last_timer + 16)

        for k, agent in enumerate(self.agent_and_puppet_instances()):
            if agent.spawn_delay == 0:
                try:
                    self.put_obj(agent, self.agent_spawn_pos[agent.name][0],
                                 self.agent_spawn_pos[agent.name][1])  # x,y,dir
                    agent.dir = self.agent_spawn_pos[agent.name][2]
                except:
                    self.place_obj(agent, **self.agent_spawn_kwargs)

                agent.activate()

        self.prior_observations = {agent: self.gen_agent_obs(a) for agent, a in zip(self.agents, self.agent_instances)}
        self.observations = {agent: self.prior_observations[agent] if self.supervised_model is None else
        np.concatenate((self.prior_observations[agent], np.zeros((1, a.view_size, a.view_size)))) for
                             agent, a in zip(self.agents, self.agent_instances)}
        self.has_released = False

        if self.supervised_model is not None:
            # if supervised model is 0 we will use ground truth
            if isinstance(self.supervised_model, str):
                self.record_oracle_labels = True
            else:
                self.supervised_model.training = False
            self.past_observations = np.zeros((10, *self.prior_observations[self.agents[0]].shape))

        # robservations = {agent: self.observations[agent] for agent in self.agents}
        # rrewards = {agent: self.rewards[agent] for agent in self.agents}
        # rdones = {agent: self.dones[agent] for agent in self.agents}
        # rinfos = {agent: self.infos[agent] for agent in self.agents}

        # print('reset', robservations, rrewards, rdones, rinfos)

        return self.observations

        # return robservations, rrewards, rdones, rinfos

    def add_timer(self, event, time, arg=None):
        if str(time) in self.timers.keys():
            self.timers[str(time)].append((event, arg))
        else:
            self.timers[str(time)] = [(event, arg), ]

    def timer_active(self, event):
        pass

    def puppet_pathing(self, agent):
        a = self.instance_from_name[agent]
        if self.infos[agent] != {}:
            if 'act' in self.infos[agent].keys():
                a.next_actions.append(self.infos[agent]['act'])
            if 'path' in self.infos[agent].keys():
                a.pathDict = self.infos[agent]['path']

        if a.pathDict != {}:
            sname = str(tuple(a.pos))
            if sname in a.pathDict.keys():
                direction = a.pathDict[sname]
            else:
                direction = random.choice([0, 1, 2, 3])
            relative_dir = (a.dir - direction) % 4
            if relative_dir == 3 or relative_dir == 2:
                a.next_actions.append(1)
            elif relative_dir == 1:
                a.next_actions.append(0)
            elif relative_dir == 0:
                a.next_actions.append(2)

    def step(self, actions):

        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''

        # reset box update vector
        self.box_updated_this_timestep = [False for _ in range(self.boxes)]

        self.bait_loc = [False for _ in range(self.boxes)]
        self.swap_loc = [False for _ in range(self.boxes)]
        self.swap_history = []
        self.treat_baited = [False for _ in range(2)]
        self.treat_swapped = [False for _ in range(2)]
        # activate timed events
        if str(self.step_count) in self.timers.keys():
            for event in self.timers[str(self.step_count)]:
                self.timer_active(event[0], event[1])

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            return {}, {}, {}, {}

        for agent_name, agent in zip(self.agents, self.agent_instances):
            if not agent.active and not agent.done and self.step_count >= agent.spawn_delay:
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()
                self._cumulative_rewards[agent] = 0

        # get all puppet actions
        puppet_actions = {}
        for agent in self.puppets:
            a = self.instance_from_name[agent]
            nextAct = 2
            if len(a.next_actions) > 0:
                nextAct = a.next_actions.pop(0)
            else:
                if len(a.pathDict.keys()) > 0:
                    pass
                else:
                    nextAct = 2
            puppet_actions[agent] = nextAct

        actions = dict(actions, **puppet_actions)

        for agent_name in actions:
            action = actions[agent_name]
            agent = self.instance_from_name[agent_name]
            agent.step_reward = 0
            self.rewards[agent_name] = 0

            if agent.active:
                #self.rewards[agent_name] = self.step_reward
                #agent.reward(self.step_reward)

                # get stuff from timers
                # infos[agent_name] = self.infos[agent_name]

                cur_pos = agent.pos[:]
                cur_cell = self.grid.get(*cur_pos)
                fwd_pos = agent.front_pos[:]
                fwd_cell = self.grid.get(*fwd_pos)
                agent_moved = False

                if agent.move_type == 0:
                    # Rotate left
                    if action == agent.actions.left:
                        agent.dir = (agent.dir - 1) % 4

                    # Rotate right
                    elif action == agent.actions.right:
                        agent.dir = (agent.dir + 1) % 4

                if action == agent.actions.forward:
                    # move forward
                    fwd_pos = agent.front_pos[:]
                    fwd_cell = self.grid.get(*fwd_pos)
                elif agent.move_type == 1:
                    # move cardinally
                    if action == agent.actions.left:
                        fwd_pos = agent.left_pos[:]
                    if action == agent.actions.right:
                        fwd_pos = agent.right_pos[:]
                    if action == agent.actions.done:
                        fwd_pos = agent.back_pos[:]
                    fwd_cell = self.grid.get(*fwd_pos)

                if action == agent.actions.forward or (agent.move_type == 1 and (
                        action in [agent.actions.left, agent.actions.right, agent.actions.done])):
                    # Under the follow conditions, the agent can move forward.
                    can_move = fwd_cell != -1 and (fwd_cell is None or fwd_cell.can_overlap())
                    if self.ghost_mode is False and isinstance(fwd_cell, GridAgent):
                        can_move = False

                    if can_move:
                        agent_moved = True
                        # Add agent to new cell
                        if fwd_cell is None:
                            self.grid.set(*fwd_pos, agent, update_vis_mask=self.prev_puppet_mask)
                            agent.pos = fwd_pos
                        elif fwd_cell != -1:
                            fwd_cell.agents.append(agent)
                            agent.pos = fwd_pos

                            # send signal to test next action outputs

                            # if "Test" in str(fwd_cell.__class__):
                            #   self.infos[agent_name]["test"] = fwd_cell.dir

                            # send signal to override next action
                            '''if "Arrow" in str(fwd_cell.__class__):
                                relative_dir = (agent.dir - fwd_cell.dir) % 4
                                if relative_dir == 3:
                                    self.infos[agent_name]["act"] = 0
                                if relative_dir == 1:
                                    self.infos[agent_name]["act"] = 1'''

                        # Remove agent from old cell
                        if cur_cell == agent:
                            self.grid.set(*cur_pos, None, update_vis_mask=self.prev_puppet_mask)
                        elif cur_cell != None:
                            # used to just be else... possibly issue because not spawning agents correctly
                            assert cur_cell.can_overlap()
                            # also this if used to not be here
                            if agent in cur_cell.agents:
                                cur_cell.agents.remove(agent)

                        # Add agent's agents to old cell
                        for left_behind in agent.agents:
                            cur_obj = self.grid.get(*cur_pos)
                            if cur_obj is None:
                                self.grid.set(*cur_pos, left_behind, update_vis_mask=self.prev_puppet_mask)
                            elif cur_obj.can_overlap():
                                cur_obj.agents.append(left_behind)
                            else:  # How was "agent" there in teh first place?
                                raise ValueError("?!?!?!")

                        # After moving, the agent shouldn't contain any other agents.
                        agent.agents = []
                        # test_integrity(f"After moving {agent.color} fellow")

                        # Rewards can be got iff. fwd_cell has a "get_reward" method
                        if hasattr(fwd_cell, 'get_reward'):
                            og_rwd = fwd_cell.get_reward(agent)
                            if agent.valence != 1:
                                temp = og_rwd
                                og_rwd = 100 if temp == self.smallReward else self.smallReward
                            # self.grid.set(*fwd_cell.pos, None) # don't remove box
                            # fwd_cell.set_reward(self.penalize_same_selection)

                            if bool(self.reward_decay):
                                rwd = og_rwd * (1.0 - 0.9 * (self.step_count / self.max_steps_real)) + (self.step_reward * self.step_count)
                            else:
                                rwd = og_rwd + (self.step_reward * self.step_count)

                            if fwd_cell in self.previously_selected_boxes:
                                same_selection = True
                                rwd = self.penalize_same_selection
                            else:
                                same_selection = False
                                self.previously_selected_boxes.append(fwd_cell)

                            # removed, unclear what for
                            # step_rewards[agent_no] += rwd
                            self.rewards[agent_name] += rwd
                            self.has_reached_goal[agent_name] = True

                            if agent_name == 'p_0':
                                self.dones[agent_name] = True
                                agent.done = True

                                # handle infos
                                if self.record_info:
                                    if self.metadata["name"] == "miniStandoffEnv":
                                        box = agent.pos[0] - 1
                                    else:
                                        box = (agent.pos[0] - 2) / 2
                                    self.infos[agent_name]["selection"] = box
                                    self.infos[agent_name]["accuracy"] = (
                                            box == self.infos[agent_name]["correct-loc"] or self.infos[agent_name][
                                        "correct-loc"] == -1)
                                    self.infos[agent_name]["weakAccuracy"] = (
                                            box == self.infos[agent_name]["correct-loc"] or box ==
                                            self.infos[agent_name]["incorrect-loc"])
                                    self.infos[agent_name]["selectedBig"] = (og_rwd > self.bigReward * 0.5)
                                    self.infos[agent_name]["selectedSmall"] = (og_rwd > self.smallReward * 0.5 and og_rwd < self.bigReward * 0.5)
                                    self.infos[agent_name]["selectedNeither"] = (og_rwd < self.smallReward * 0.5)
                                    self.infos[agent_name]["selectedPrevBig"] = (box in self.big_food_locations)
                                    self.infos[agent_name]["selectedPrevSmall"] = (box in self.small_food_locations)
                                    self.infos[agent_name]["selectedPrevNeither"] = not (
                                            box in self.big_food_locations) and not (box in self.small_food_locations)
                                    self.infos[agent_name][
                                        "selectedSame"] = same_selection  # selected same box as a previous agent
                            else:
                                agent.active = False
                            agent.reward(rwd)
                            # agent.step_reward = rwd

                        if isinstance(fwd_cell, (Lava, Goal)):
                            if agent_name == 'p_0':
                                agent.done = True
                                # added below
                                self.dones[agent_name] = True

                # Pick up an object
                elif action == agent.actions.pickup:
                    if fwd_cell and fwd_cell != -1 and fwd_cell.can_pickup():
                        if agent.carrying is None:
                            agent.carrying = fwd_cell
                            agent.carrying.cur_pos = np.array([-1, -1])
                            self.grid.set(*fwd_pos, None, update_vis_mask=self.prev_puppet_mask)
                    else:
                        pass

                # Drop an object
                elif action == agent.actions.drop:
                    if not fwd_cell and agent.carrying:
                        self.grid.set(*fwd_pos, agent.carrying, update_vis_mask=self.prev_puppet_mask)
                        agent.carrying.cur_pos = fwd_pos
                        agent.carrying = None
                    else:
                        pass

                # Toggle/activate an object
                elif action == agent.actions.toggle:
                    if fwd_cell and fwd_cell != -1:
                        wasted = bool(fwd_cell.toggle(agent, fwd_pos))
                        self.prev_puppet_mask[fwd_pos[0], fwd_pos[1]] = False
                    else:
                        pass

                agent.on_step(fwd_cell if agent_moved else None)

        # rewards for all agents are placed in the .rewards dictionary

        self.step_count += 1
        if self.step_count >= self.max_steps_real:
            self.env_done = True

        # observe the current state
        for agent_name, agent in zip(self.agents, self.agent_instances):
            generated_obs = self.gen_agent_obs(agent)
            if self.supervised_model is not None:
                if self.step_count < 10 and not self.has_released:
                    self.past_observations[self.step_count] = generated_obs
                    # hashed = hashlib.sha1(self.past_observations.view(np.uint8)).hexdigest()
                    # let's use a faster hashing function from xxhash:
                    hashed = xxhash.xxh64(self.past_observations.view(np.uint8)).hexdigest()
                    if hashed in self.supervised_label_dict.keys():
                        self.last_supervised_labels = self.supervised_label_dict[hashed]
                    else:
                        # check if supervised model is string
                        if isinstance(self.supervised_model, str):
                            self.last_supervised_labels = np.asarray(self.infos['p_0'][self.supervised_model]).flatten()
                        else:
                            self.last_supervised_labels = self.supervised_model.forward(
                                np.asarray([self.past_observations])).detach().numpy()[0]
                        self.supervised_label_dict[hashed] = self.last_supervised_labels
                label_obs = np.zeros((1, agent.view_size, agent.view_size), dtype="uint8")
                label_obs[0, 0, :self.last_supervised_labels.shape[0]] = self.last_supervised_labels
                self.observations[agent_name] = np.concatenate((generated_obs, label_obs), axis=0)
            else:
                self.observations[agent_name] = generated_obs

            # self.rewards[agent_name] = agent.rew
            if self.env_done:
                self.dones[agent_name] = True
                if not self.has_reached_goal[agent_name]:
                    dr = self.done_without_box_reward
                    if self.distance_from_boxes_reward != 0:
                        done_distance = abs((self.height // 2) - agent.pos[1])
                        # the vertical spaces agent pos differs from goal pos
                        dr += self.distance_from_boxes_reward * done_distance
                    self.rewards[agent_name] += dr
                    agent.reward(dr)

        # Adds .rewards to ._cumulative_rewards
        self._cumulative_rewards = {agent: self._cumulative_rewards[agent] + self.rewards[agent] for agent in
                                    self.agents}

        # self._accumulate_rewards() #not defined

        for agent in self.puppets:
            self.puppet_pathing(agent)

        if self.record_oracle_labels and ((self.step_count <= self.end_at_frame or self.end_at_frame == -1) or self.has_released):
            tolerance = 6
            y = self.height // 2
            target_agent = "p_1"
            one_hot_goal = [0] * self.boxes
            one_hot_goal_imaginary = [0] * self.boxes

            one_hot_goal_box = [0] * self.boxes
            one_hot_goal_imaginary_box = [0] * self.boxes

            if self.params['num_puppets'] > 0:
                one_hot_goal[self.agent_goal[target_agent]] = 1
                one_hot_goal_box[self.box_locations[self.agent_goal[target_agent]]] = 1


            one_hot_goal_imaginary[self.agent_goal['i']] = 1
            one_hot_goal_imaginary_box[self.box_locations[self.agent_goal['i']]] = 1

            info = self.infos['p_0']

            info["target-loc"] = one_hot_goal
            info["i-target-loc"] = list(map(int, one_hot_goal_imaginary))
            info["target-size"] = list(map(int, [info['shouldAvoidSmall'], info['shouldAvoidBig']])) #is this imaginary?
            #info["i-target-size"] = defined below
            info["box-updated"] = list(map(int, self.box_updated_this_timestep))

            info["swap-treat"] = list(map(int, self.treat_swapped))
            info["bait-treat"] = list(map(int, self.treat_baited))

            info["swap-loc"] = list(map(int, self.swap_loc))

            #self.swap_history += [self.box_locations[:]]
            info["box-locations"] = self.box_locations
            info["b-box-locations"] = self.b_box_locations
            info["i-b-box-locations"] = self.i_b_box_locations

            info["bait-loc"] = list(map(int, self.bait_loc))

            info["treat-box"] = self.treat_box
            info["b-treat-box"] = self.b_treat_box
            info["i-b-treat-box"] = self.i_b_treat_box

            info["target-box"] = one_hot_goal_box
            info["i-target-box"] = one_hot_goal_imaginary_box

            # todo: test each of these

            if not self.currently_visible:
                for i in range(self.boxes):
                    if self.box_updated_this_timestep[i]:
                        self.saw_last_update[i] = False

            info["saw-last-update"] = list(map(int, self.saw_last_update))
            info["vision"] = int(self.currently_visible)
            real_boxes = [self.grid.get(box + 1, y) for box in range(self.boxes)]

            real_box_rewards = [box.get_reward() if box is not None and hasattr(box, "get_reward") else 0 for box in real_boxes]
            if self.params['num_puppets'] > 0:
                all_rewards_seen = [self.last_seen_reward[target_agent + str(box)] for box in range(self.boxes)]
            else:
                all_rewards_seen = [0] * self.boxes

            all_rewards_seen_imaginary = [self.last_seen_reward['i' + str(box)] for box in range(self.boxes)]

            info["loc"] = [
                [1, 0] if reward == self.bigReward else
                [0, 1] if reward == self.smallReward else
                [0, 0]
                for reward in real_box_rewards
            ]

            max_small_reward = max([reward for reward in all_rewards_seen if abs(reward - self.smallReward) < tolerance], default=None)
            max_big_reward = max([reward for reward in all_rewards_seen if abs(reward - self.bigReward) < tolerance], default=None)
            max_small_reward_i = max([reward for reward in all_rewards_seen_imaginary if abs(reward - self.smallReward) < tolerance], default=None)
            max_big_reward_i = max([reward for reward in all_rewards_seen_imaginary if abs(reward - self.bigReward) < tolerance], default=None)

            info["b-loc"] = [
                [1, 0] if reward == max_big_reward else
                [0, 1] if reward == max_small_reward else
                [0, 0]
                for reward in all_rewards_seen
            ]
            info["i-b-loc"] = [
                [1, 0] if reward == max_big_reward_i else
                [0, 1] if reward == max_small_reward_i else
                [0, 0]
                for reward in all_rewards_seen_imaginary
            ]

            info["fb"] = any(
                b_loc != loc
                for b_loc, loc in zip(info["b-loc"], info["loc"])
            )
            info["i-fb"] = any(
                b_loc != loc
                for b_loc, loc in zip(info["i-b-loc"], info["loc"])
            )

            info["b-loc-diff"] = [
                [int(l != b) for l, b in zip(loc, b_loc)]
                for loc, b_loc in zip(info["loc"], info["b-loc"])
            ]

            info["i-b-loc-diff"] = [
                [int(l != i_b) for l, i_b in zip(loc, i_b_loc)]
                for loc, i_b_loc in zip(info["loc"], info["i-b-loc"])
            ]

            #info['correct-box'][int(self.box_locations[int(self.infos['p_0']['correct-loc'])])] = 1
            # correct-loc comes from ministandoff step following new target
            info["b-correct-loc"] = distribute_weights_to_second_largest(all_rewards_seen) if self.params['num_puppets'] > 0 else [1 / self.boxes] * self.boxes
            info["i-b-correct-loc"] = distribute_weights_to_second_largest(all_rewards_seen_imaginary)
            info['correct-box'] = [0] * self.boxes
            info['b-correct-box'] = [0] * self.boxes
            info['i-b-correct-box'] = [0] * self.boxes

            # correct-loc is not one-hot, so we handle it differently:
            info['correct-box'][info["box-locations"][info["correct-loc"]]] = 1
            for box in range(self.boxes):
                info['b-correct-box'][box] = info["b-correct-loc"][info["b-box-locations"][box]]
                info['i-b-correct-box'][box] = info["i-b-correct-loc"][info["i-b-box-locations"][box]]

            #info["b-correct-box"] = [info["box-locations"][k] if x == 1 else 0 for k, x in enumerate(info["b-correct-loc"])]
            info["i-b-correct-box"] = [info["box-locations"][k] if x == 1 else 0 for k, x in enumerate(info["i-b-correct-loc"])]

            for name in ["loc", "b-loc", "i-b-loc", "b-loc-diff", "i-b-loc-diff"]:

                #info["scalar-" + name] = convert_to_scalar(info[name]) #broken for the diff ones?
                info["big-" + name] = [x[0] for x in info[name]]
                info["small-" + name] = [x[1] for x in info[name]]
                #info["any-" + name] = [int(x != [0, 0]) for x in info[name]]

            def loc_to_box(loc_vector, box_locations):
                box_info = [0] * len(box_locations)
                for loc, box in enumerate(box_locations):
                    if loc_vector[loc] == 1:
                        box_info[box] = 1
                return box_info
            info["big-box"] = loc_to_box(info["big-loc"], info["box-locations"])
            info["small-box"] = loc_to_box(info["small-loc"], info["box-locations"])
            info["big-b-box"] = loc_to_box(info["big-b-loc"], info["b-box-locations"])
            info["small-b-box"] = loc_to_box(info["small-b-loc"], info["b-box-locations"])
            info["b-box"] = [
                [1, 0] if big == 1 else
                [0, 1] if small == 1 else
                [0, 0]
                for big, small in zip(info["big-b-box"], info["small-b-box"])
            ]

            info["exist"] = [1 if self.bigReward in real_box_rewards else 0, 1 if self.smallReward in real_box_rewards else 0]
            info["b-exist"] = [
                1 if any(abs(reward - self.bigReward) < tolerance for reward in all_rewards_seen) else 0,
                1 if any(abs(reward - self.smallReward) < tolerance for reward in all_rewards_seen) else 0
            ]
            info["i-b-exist"] = [
                1 if any(abs(reward - self.bigReward) < tolerance for reward in all_rewards_seen_imaginary) else 0,
                1 if any(abs(reward - self.smallReward) < tolerance for reward in all_rewards_seen_imaginary) else 0
            ]

            info["b-exist-diff"] = [
                int(l != b) for l, b in zip(info["exist"], info["b-exist"])
            ]

            info["i-b-exist-diff"] = [
                int(l != i_b) for l, i_b in zip(info["exist"], info["i-b-exist"])
            ]

            imaginary_reward = real_box_rewards[self.agent_goal['i']]
            info["i-target-size"] = list(map(int, [imaginary_reward == self.bigReward, imaginary_reward == self.smallReward]))


            if False and self.params['num_puppets'] == 0 and [1, 0] in info["i-b-loc"]:
                print('calc:', info["loc"],'bloc', info["b-loc"], info["i-b-loc"],  'bexi', info["b-exist"], info["i-b-exist"], 'inf', calculate_informedness(info["loc"], info["b-loc"]), all_rewards_seen)

            info["informedness"] = calculate_informedness(info["loc"], info["b-loc"])
            info["i-informedness"] = calculate_informedness(info["loc"], info["i-b-loc"])

        # clear puppets from obs, rewards, dones, infos

        robservations = {agent: self.observations[agent] for agent in self.agents}
        rrewards = {agent: self.rewards[agent] for agent in self.agents}
        rdones = {agent: self.dones[agent] for agent in self.agents}
        rinfos = {agent: self.infos[agent] for agent in self.agents}

        return robservations, rrewards, rdones, rinfos

    def slice_gaze_grid(self, agent, gaze_grid):

        # probably horribly inefficient. making a multigrid to use existing slice function
        gaze_grid = MultiGrid(gaze_grid)
        # gaze_grid.grid = gaze_grid

        if agent.view_type == 0:
            # egocentric view
            topX, topY, botX, botY = agent.get_view_exts()
            grid = gaze_grid.slice(
                topX, topY, agent.view_size, agent.view_size, rot_k=agent.dir + 1
            )
        else:
            grid = gaze_grid.slice(
                0, 0, agent.view_size, agent.view_size, 0
            )
        ret = grid.grid
        del gaze_grid
        return ret

    def gen_obs_grid(self, agent):

        # If the agent is inactive, return an empty grid and a visibility mask that hides everything.
        if not agent.active:
            # below, not sure orientation is correct but as of 6/27/2020 that doesn't matter because
            # agent views are usually square and this grid won't be used for anything.
            grid = MultiGrid((agent.view_size, agent.view_size), orientation=agent.dir + 1)
            vis_mask = np.zeros((agent.view_size, agent.view_size), dtype="uint8")  # was np.bool
            return grid, vis_mask

        if agent.view_type == 0:
            # egocentric view
            topX, topY, botX, botY = agent.get_view_exts()
            grid = self.grid.slice(
                topX, topY, agent.view_size, agent.view_size, rot_k=agent.dir + 1
            )
        else:
            grid = self.grid.slice(
                0, 0, agent.view_size, agent.view_size, 0
            )

        # Process occluders and visibility
        # Note that this incurs some slight performance cost
        vis_mask = agent.process_vis(grid.opacity)

        # Warning about the rest of the function:
        #  Allows masking away objects that the agent isn't supposed to see.
        #  But breaks consistency between the states of the grid objects in the parial views
        #   and the grid objects overall.
        if len(getattr(agent, 'hide_item_types', [])) > 0:
            for i in range(grid.width):
                for j in range(grid.height):
                    item = grid.get(i, j)
                    if (item is not None) and (item is not agent) and (item.type in agent.hide_item_types):
                        if len(item.agents) > 0:
                            grid.set(i, j, item.agents[0])
                        else:
                            grid.set(i, j, None)

        return grid, vis_mask

    def gen_agent_obs(self, agent):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        if self.gaze_highlighting is True:
            # get the puppet's view mask
            puppet_mask = np.zeros((agent.view_size, agent.view_size),
                                   dtype="uint8")  # if we don't find a puppet instance? unclear when this happens
            if self.params['num_puppets'] > 0:
                for puppet in self.puppet_instances[:self.params['num_puppets']]:
                    if puppet != self.instance_from_name["p_0"]:
                        if puppet.pos is not None:
                            puppet_mask = occlude_mask(~self.grid.opacity, puppet.pos)  # only reveals one tile?
                            if self.only_highlight_treats:
                                puppet_mask = np.logical_and(puppet_mask, self.grid.all_treats)

                            if self.persistent_gaze_highlighting is True:
                                self.prev_puppet_mask = np.logical_or(self.prev_puppet_mask, puppet_mask)
                                puppet_mask = self.prev_puppet_mask

                            puppet_mask = self.slice_gaze_grid(agent,
                                                               puppet_mask)  # get relative gaze mask in agent view
            else:
                # otherwise puppet mask is just 0s
                puppet_mask = np.zeros((agent.view_size, agent.view_size), dtype="uint8")
        else:
            puppet_mask = None

        view_grid, vis_mask = self.gen_obs_grid(agent)

        if self.observation_style == 'rich':
            # bypass rendering, just do dense function
            mapping = view_grid.obj_reg.key_to_obj_map

            visibility = vis_mask
            obs = np.zeros((len(self.rich_observation_layers), agent.view_size, agent.view_size))
            for i, layer in enumerate(self.rich_observation_layers):
                if len(mapping) < 2:
                    continue
                if isinstance(layer, str):
                    if layer == 'vis':
                        obs[i, :, :] = visibility
                    elif layer == 'gaze':
                        obs[i, :, :] = puppet_mask
                else:
                    obs[i, :, :] = np.multiply(np.vectorize(layer)(view_grid.grid, mapping), visibility)

            # remove this to speed up training, though it's useful for debugging
            '''
            if np.isnan(obs).any():
                print("Warning: NaN values detected in the observations")
                # To further debug, print relevant variables or arrays here
                print("visibility:", visibility)
                print("layers:", self.rich_observation_layers)
                print("obs", obs)'''
            return obs

        grid_image = view_grid.render(tile_size=agent.view_tile_size, visible_mask=vis_mask, top_agent=agent,
                                      gaze_highlight_mask=puppet_mask)
        if agent.observation_style == 'image':
            return grid_image
        else:
            ret = {'pov': grid_image}
            if agent.observe_rewards:
                ret['reward'] = getattr(agent, 'step_reward', 0)
            if agent.observe_position:
                agent_pos = agent.pos if agent.pos is not None else (0, 0)
                ret['position'] = np.array(agent_pos) / np.array([self.width, self.height],
                                                                 dtype="uint8")  # was np.float
            if agent.observe_orientation:
                agent_dir = agent.dir if agent.dir is not None else 0
                ret['orientation'] = agent_dir
            return ret

    def __str__(self):
        return self.grid.__str__()

    def put_obj(self, obj, i, j, update_vis=True):
        """
        Put an object at a specific position in the grid. Replace anything that is already there.
        """
        self.grid.set(i, j, obj, update_vis_mask=self.prev_puppet_mask if update_vis else None)
        if obj is not None:
            obj.set_position((i, j))
        return True

    def del_obj(self, i, j):
        o = self.grid.get(i, j)
        self.grid.grid[i, j] = 0
        self.prev_puppet_mask[i, j] = False
        del o

    def try_place_obj(self, obj, pos):
        ''' Try to place an object at a certain position in the grid.
        If it is possible, then do so and return True.
        Otherwise do nothing and return False. '''
        # grid_obj: whatever object is already at pos.
        grid_obj = self.grid.get(*pos)

        # If the target position is empty, then the object can always be placed.
        if grid_obj is None:
            self.grid.set(*pos, obj)
            obj.set_position(pos)
            return True

        # Otherwise only agents can be placed, and only if the target position can_overlap.
        if not (grid_obj.can_overlap() and obj.is_agent):
            return False

        # If ghost mode is off and there's already an agent at the target cell, the agent can't
        #   be placed there.
        if (not self.ghost_mode) and (grid_obj.is_agent or (len(grid_obj.agents) > 0)):
            return False

        grid_obj.agents.append(obj)
        obj.set_position(pos)
        return True

    def place_obj(self, obj, top=(0, 0), size=None, reject_fn=None, max_tries=1e5):
        max_tries = int(max(1, min(max_tries, 1e5)))
        top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)
        bottom = (min(top[0] + size[0], self.grid.width), min(top[1] + size[1], self.grid.height))

        # agent_positions = [tuple(agent.pos) if agent.pos is not None else None for agent in self.agents]
        for try_no in range(max_tries):
            pos = self.np_random.randint(top, bottom)
            if (reject_fn is not None) and reject_fn(pos):
                continue
            else:
                if self.try_place_obj(obj, pos):
                    break
        else:
            raise RecursionError("Rejection sampling failed in place_obj.")

        return pos

    def place_agents(self, top=None, size=None, rand_dir=True, max_tries=1000):
        # warnings.warn("Placing agents with the function place_agents is deprecated.")
        pass

    def render(
            self,
            mode="human",
            close=False,
            highlight=True,
            tile_size=TILE_PIXELS,
            show_agent_views=True,
            max_agents_per_col=3,
            agent_col_width_frac=0.3,
            agent_col_padding_px=2,
            pad_grey=100
    ):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == "human" and not self.window:
            self.window = SimpleImageViewer(caption='standoff')
        # Compute which cells are visible to the agent
        highlight_mask = np.full((self.width, self.height), False, dtype=np.bool)
        for agentname, agent in zip(self.agents, self.agent_instances):
            if agent.active:
                xlow, ylow, xhigh, yhigh = agent.get_view_exts()
                dxlow, dylow = max(0, 0 - xlow), max(0, 0 - ylow)
                dxhigh, dyhigh = max(0, xhigh - self.grid.width), max(0, yhigh - self.grid.height)
                if agent.see_through_walls:
                    highlight_mask[xlow + dxlow:xhigh - dxhigh, ylow + dylow:yhigh - dyhigh] = True
                else:
                    a, b = self.gen_obs_grid(agent)
                    highlight_mask[xlow + dxlow:xhigh - dxhigh, ylow + dylow:yhigh - dyhigh] |= (
                        rotate_grid(b, a.orientation)[dxlow:(xhigh - xlow) - dxhigh, dylow:(yhigh - ylow) - dyhigh]
                    )
        # Render the whole grid
        img = self.grid.render(
            tile_size, highlight_mask=highlight_mask if highlight else None
        )
        rescale = lambda X, rescale_factor=2: np.kron(
            X, np.ones((int(rescale_factor), int(rescale_factor), 1))
        )

        if show_agent_views and mode == "human":

            target_partial_width = int(img.shape[0] * agent_col_width_frac - 2 * agent_col_padding_px)
            target_partial_height = (img.shape[1] - 2 * agent_col_padding_px) // max_agents_per_col

            agent_views = [self.gen_agent_obs(agent) for agent in self.agent_and_puppet_instances()]
            agent_views = [view['pov'] if isinstance(view, dict) else view for view in agent_views]
            agent_views = [
                rescale(view, min(target_partial_width / view.shape[0], target_partial_height / view.shape[1])) for view
                in agent_views]
            # import pdb; pdb.set_trace()
            agent_views = [agent_views[pos:pos + max_agents_per_col] for pos in
                           range(0, len(agent_views), max_agents_per_col)]

            f_offset = lambda view: np.array(
                [target_partial_height - view.shape[1], target_partial_width - view.shape[0]]) // 2

            cols = []
            for col_views in agent_views:
                col = np.full((img.shape[0], target_partial_width + 2 * agent_col_padding_px, 3), pad_grey,
                              dtype=np.uint8)
                for k, view in enumerate(col_views):
                    offset = f_offset(view) + agent_col_padding_px
                    offset[0] += k * target_partial_height
                    col[offset[0]:offset[0] + view.shape[0], offset[1]:offset[1] + view.shape[1], :] = view
                cols.append(col)

            img = np.concatenate((img, *cols), axis=1)


        if mode == "human":
            if not self.window.isopen:
                self.window.imshow(img)
                #self.window.window.set_caption("Standoff")
                self.window.isopen = True
            else:
                self.window.imshow(img)

        return img
