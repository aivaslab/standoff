import os
import numpy as np
from src.rendering import InteractivePlayerWindow
from src.agents import GridAgentInterface
from src.pz_envs import env_from_config
from src.pz_envs.scenario_configs import ScenarioConfigs
from PIL import Image, ImageDraw
from collections import defaultdict

class HumanPlayer:
    def __init__(self):
        self.player_window = InteractivePlayerWindow(
            caption="standoff player",
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

TILE_SIZE = 16

env_config = {
    "env_class": "MiniStandoffEnv",
    "max_steps": 50,
    "respawn": True,
    "ghost_mode": False,
    "reward_decay": False,
    "width": 9,
    "height": 9,
    "use_separate_reward_layers": True
}

player_interface_config = {
    "view_size": 7,
    "view_offset": 0,
    "view_tile_size": TILE_SIZE,
    "observation_style": "image",
    "see_through_walls": False,
    "color": "yellow",
    "view_type": 0,
    "move_type": 0
}
puppet_interface_config = {
    "view_size": 7,
    "view_offset": 1,
    "view_tile_size": TILE_SIZE,
    "observation_style": "rich",
    "see_through_walls": False,
    "color": "red",
}

conf = ScenarioConfigs()
configs = conf.standoff

difficulty = 3
env_config['opponent_visible_decs'] = (difficulty < 1)
env_config['persistent_treat_images'] = (difficulty < 2)
env_config['subject_visible_decs'] = (difficulty < 3)
env_config['gaze_highlighting'] = (difficulty < 3)
env_config['persistent_gaze_highlighting'] = (difficulty < 2)

save_directory = "saved_event_images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

all_unique_events = set()
for stage_name, stage_data in conf.stages.items():
    if "1" not in stage_name:
        continue
    if 'events' in stage_data:
        for event_name in stage_data['events']:
            all_unique_events.add((event_name, stage_name))

event_groups = defaultdict(list)
for event_name, stage_name in all_unique_events:
    group_name = event_name.split('-')[0]
    event_groups[group_name].append((event_name, stage_name))

print(f"Found {len(event_groups)} event groups with {len(all_unique_events)} total events")

human = HumanPlayer()
human.start_episode()

for group_idx, (group_name, events_in_group) in enumerate(sorted(event_groups.items())):
    print(f"Processing group {group_idx + 1}/{len(event_groups)}: {group_name} with {len(events_in_group)} events")
    
    group_by_text = {}
    group_metadata = []
    
    first_event_name, first_stage_name = events_in_group[0]
    permutation_count = 1
    for perm in conf.all_event_permutations[first_event_name]:
        permutation_count *= perm
    delay_count = len(conf.all_event_delays.get(first_event_name, []))

    group_gettiers = []
    
    for event_name, stage_name in sorted(events_in_group):
        print(f"  Processing event: {event_name}")
        
        stage_with_event = None
        for s_name, stage_data in conf.stages.items():
            if 'events' in stage_data and event_name in stage_data['events']:
                stage_with_event = s_name
                break
        
        if not stage_with_event:
            continue
        
        events = {event_name: conf.stages[stage_with_event]['events'][event_name]}
        params = configs[conf.stages[stage_with_event]['params']]
        reset_configs = params.copy()
        
        if isinstance(reset_configs["num_agents"], list):
            reset_configs["num_agents"] = reset_configs["num_agents"][0]
        if isinstance(reset_configs["num_puppets"], list):
            reset_configs["num_puppets"] = reset_configs["num_puppets"][0]

        env_config['config_name'] = stage_with_event
        env_config['agents'] = [GridAgentInterface(**player_interface_config) for _ in range(reset_configs['num_agents'])]
        env_config['puppets'] = [GridAgentInterface(**puppet_interface_config) for _ in range(reset_configs['num_puppets'])]

        env_name = f'Standoff-S3-{event_name.replace(" ", "")}-{difficulty}-v1'
        
        env = env_from_config(env_config)
        env.observation_style = "image"
        params['subject_is_dominant'] = False
        params['sub_valence'] = 1
        
        list_count = len(events[event_name])
        
        env.param_groups = [{'eLists': {event_name: events[event_name]},
                             'params': params,
                             'perms': {event_name: conf.all_event_permutations[event_name]},
                             'delays': {event_name: conf.all_event_delays[event_name]}
                             }]

        env.record_info = True
        env.deterministic = True
        env.record_oracle_labels = True

        delay_variations = conf.all_event_delays.get(event_name, [[]])
        
        text_key = (event_name, stage_name, str(events[event_name]))
        if text_key not in group_by_text:
            group_by_text[text_key] = {
                'filler_variations': [],
                'metadata': {
                    'event_name': event_name,
                    'stage_name': stage_name[3:],
                    'events': events[event_name],
                    'delay_count': len(delay_variations)
                }
            }
        
        for delay_idx, delay_pattern in enumerate(delay_variations):
            env.reset(override_params=0)
            env.current_param_group_pos = delay_idx
            env.deterministic_seed = env.current_param_group_pos

            obs = env.reset()
            
            timestep_images = []
            step_count = 0
            
            while True:
                img = Image.fromarray(obs['p_0'], 'RGB')
                if step_count > 0:
                    timestep_images.append(img.copy())
                
                agent_actions = {'p_0': 2}
                next_obs, rew, done, info = env.step(agent_actions)
                if event_name == "b1w2v0fs-2":
                    print(env.big_food_locations, env.small_food_locations)
                    print(env.treat_was_wrong)
                
                human.save_step(obs['p_0'], 2, rew['p_0'], done)
                obs = next_obs
                step_count += 1
                
                if done['p_0'] or step_count >= 5:
                    break
            
            group_by_text[text_key]['filler_variations'].append(timestep_images)
        
        env.current_param_group_pos = 0
        group_gettiers.append([env.infos['p_0']['gettier_big'], env.infos['p_0']['gettier_small']])
        if event_name == "b1w2v0fs-2":
            print(env.big_food_locations, env.small_food_locations, env.infos['p_0']['gettier_big'], env.infos['p_0']['gettier_small'])
            #exit()
    
    group_timestep_images = []
    group_metadata = []
    
    for text_key, data in group_by_text.items():
        for variation_images in data['filler_variations']:
            group_timestep_images.append(variation_images)
            group_metadata.append(data['metadata'])
    
    if group_timestep_images and len(group_timestep_images[0]) > 0:
        full_img_width, full_img_height = group_timestep_images[0][0].size
        
        tile_size = TILE_SIZE
        cropped_img_width = full_img_width
        cropped_img_height = tile_size * 3
        
        num_events_in_group = len(group_timestep_images)
        max_timesteps = max(len(timesteps) for timesteps in group_timestep_images)
        
        combined_width = cropped_img_width * max_timesteps
        text_height = 40
        event_name_height = 25
        event_list_height = 25
        num_text_rows = len(group_by_text)
        num_total_rows = len(group_timestep_images)
        num_image_only_rows = num_total_rows - num_text_rows
        
        combined_height = (cropped_img_height + event_name_height + event_list_height + 10) * num_text_rows + (cropped_img_height + 5) * num_image_only_rows + text_height
        
        combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
        
        try:
            from PIL import ImageFont
            large_font = ImageFont.truetype("arial.ttf", 18)
            header_font = ImageFont.truetype("arial.ttf", 16)
        except:
            large_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(combined_img)
        header_text = f"Task Group: {group_name}"
        draw.text((10, 5), header_text, (0, 0, 0), font=header_font)
        header2_text = f"Box Permutations: {permutation_count}    Time Fillers: {delay_count}    Tasks: {len(events_in_group)}"
        draw.text((10, 25), header2_text, (0, 0, 0))
        
        y_offset = text_height
        i = 0
        event_idx = 0
        while i < len(group_timestep_images):
            group_start = i
            current_metadata = group_metadata[i]
            
            while i < len(group_timestep_images) and group_metadata[i]['event_name'] == current_metadata['event_name'] and group_metadata[i]['stage_name'] == current_metadata['stage_name']:
                timestep_images = group_timestep_images[i]
                for j, img in enumerate(timestep_images[:max_timesteps]):
                    cropped_img = img.crop((0, tile_size, full_img_width, tile_size + cropped_img_height))
                    combined_img.paste(cropped_img, (j * cropped_img_width, y_offset))
                y_offset += cropped_img_height + 5
                i += 1
            
            event_name_text = current_metadata['event_name']
            draw.text((10, y_offset), event_name_text, (0, 0, 0), font=large_font)
            
            stage_name_display = current_metadata['stage_name']
            print(group_gettiers)
            if group_gettiers[event_idx][0]:
                stage_name_display = stage_name_display.replace('T', 'G')
            if group_gettiers[event_idx][1]:
                stage_name_display = stage_name_display.replace('t', 'g')

            event_idx += 1

            stage_text = f"Informedness: {stage_name_display}"
            
            stage_text_bbox = draw.textbbox((0, 0), stage_text, font=large_font)
            stage_text_width = stage_text_bbox[2] - stage_text_bbox[0]
            draw.text((combined_width - stage_text_width - 10, y_offset), stage_text, (0, 0, 0), font=large_font)
            
            event_list_text = f"Events: {str(current_metadata['events'])}"
            draw.text((10, y_offset + event_name_height), event_list_text, (0, 0, 0))
            
            y_offset += event_name_height + event_list_height + 10
        
        image_path = os.path.join(save_directory, f"group_{group_name.replace(' ', '_')}.png")
        combined_img.save(image_path)
        print(f"Saved group image: {image_path}")

human.end_episode()

print(f"Processed all {len(event_groups)} event groups")