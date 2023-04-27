import copy

import numpy as np
import src

from src.rendering import InteractivePlayerWindow
from src.agents import GridAgentInterface
from src.pz_envs import env_from_config
from src.pz_envs.scenario_configs import ScenarioConfigs
from src.utils.conversion import make_env_comp
import pyglet
import gym
import src.pz_envs


class HumanPlayer:
    def __init__(self):
        self.player_window = InteractivePlayerWindow(
            caption="standoff",
            #display=pyglet.canvas.get_display()
        )
        self.episode_count = 0

    def action_step(self, obs):
        return self.player_window.get_action(obs.astype(np.uint8))

    def save_step(self, obs, act, rew, done):
        print(f"   step {self.step_count:<3d}: reward {rew} (episode total {self.cumulative_reward})")
        self.cumulative_reward += rew
        self.step_count += 1

    def start_episode(self):
        self.cumulative_reward = 0
        self.step_count = 0
    
    def end_episode(self):
        print(
            f"Finished episode {self.episode_count} after {self.step_count} steps."
            f"  Episode return was {self.cumulative_reward}."
        )
        self.episode_count += 1


env_config =  {
    "env_class": "StandoffEnv",
    "max_steps": 50,
    "respawn": True,
    "ghost_mode": False,
    "reward_decay": True,
    "width": 9,
    "height": 9
}

player_interface_config = {
    "view_size": 15,
    "view_offset": 1,
    "view_tile_size": 15,
    "observation_style": "image",
    "see_through_walls": False,
    "color": "yellow",
    "view_type": 0,
    "move_type": 0
}
puppet_interface_config = {
    "view_size": 15,
    "view_offset": 3,
    "view_tile_size": 48,
    "observation_style": "image",
    "see_through_walls": False,
    "color": "red",
    #"move_type": 1,
    #"view_type": 1,
}
configs = ScenarioConfigs().standoff

configName = 'all'
reset_configs = {**configs["defaults"],  **configs[configName]}

if isinstance(reset_configs["num_agents"], list):
    reset_configs["num_agents"] = reset_configs["num_agents"][0]
if isinstance(reset_configs["num_puppets"], list):
    reset_configs["num_puppets"] = reset_configs["num_puppets"][0]

env_config['config_name'] = configName
env_config['agents'] = [GridAgentInterface(**player_interface_config) for _ in range(reset_configs['num_agents'])]
env_config['puppets'] = [GridAgentInterface(**puppet_interface_config) for _ in range(reset_configs['num_puppets'])]
#env_config['num_agents'] = reset_configs['num_agents']
#env_config['num_puppets'] = reset_configs['num_puppets']

difficulty = 3
env_config['opponent_visible_decs'] = (difficulty < 1)
env_config['persistent_treat_images'] = (difficulty < 2)
env_config['subject_visible_decs'] = (difficulty < 3)
env_config['gaze_highlighting'] = (difficulty < 3)
env_config['persistent_gaze_highlighting'] = (difficulty < 2)

env_name = 'Standoff-S3-' + configName.replace(" ", "") + '-' + str(difficulty) + '-v0'

env = env_from_config(env_config)
if hasattr(env, "hard_reset"):
    env.hard_reset(reset_configs)

human = HumanPlayer()
human.start_episode()
#env.window = human.player_window



done = False


for i in range(5):
    obs = env.reset()
    print(env_name, env.gaze_highlighting, env.persistent_gaze_highlighting, env.opponent_visible_decs, env.subject_visible_decs, env.persistent_treat_images)

    while True:
        #print(env, "agents", env.agents, "puppets", env.puppets, "done_penalties", env.done_without_box_reward, env.distance_from_boxes_reward)
        env.render(mode="human", show_agent_views=True, tile_size=15)
        player_action = human.action_step(obs['player_0'])
        #print(np.round(obs['player_0']*10).sum(axis=0).astype(int))

        agent_actions = {'player_0': player_action}

        next_obs, rew, done, info = env.step(agent_actions)
        
        human.save_step(
            obs['player_0'], player_action, rew['player_0'], done
        )

        obs = next_obs

        if done['player_0'] == True:
            break

human.end_episode()
