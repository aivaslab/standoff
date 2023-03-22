import os

import pandas as pd
import torch
import math
import numpy as np
from typing import Dict, Any, List


def _process_infos(all_infos: List[Dict[str, Any]]) -> pd.DataFrame:
    for infos in all_infos:
        infos['avoidedBig'] = infos['selectedSmall'] or infos['selectedNeither']
        infos['avoidedSmall'] = infos['selectedBig'] or infos['selectedNeither']
        infos['avoidCorrect'] = (infos['avoidedBig'] == infos['shouldAvoidBig']) or (
                infos['avoidedSmall'] == infos['shouldAvoidSmall'])
        infos['accuracy'] = 1 if infos['selection'] == infos['correctSelection'] else 0
        infos['weakAccuracy'] = 1 if infos['selection'] == infos['correctSelection'] or infos['selection'] == infos[
            'incorrectSelection'] else 0
        for i in range(5):
            infos[f'sel' + str(i)] = 1 if infos['selection'] == i else 0

        return pd.DataFrame(all_infos, index=np.arange(len(all_infos)))


def get_relative_direction(agent, path):
    sname = str(tuple(agent.pos))
    if sname in path.keys():
        direction = path[sname]
    else:
        print('unknown', sname, path.keys())
        direction = agent.dir  # random.choice([0, 1, 2, 3])
    relative_dir = (agent.dir - direction) % 4
    if relative_dir == 3 or relative_dir == 2:
        return 1
    elif relative_dir == 1:
        return 0
    elif relative_dir == 0:
        return 2


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"


def collect_rollouts(env, model, model_episode,
                     episodes: int = 100,
                     memory: int = 1,
                     deterministic_env: bool = False,
                     deterministic_model: bool = False):
    env = env.unwrapped.vec_envs[0].par_env.unwrapped
    all_infos = []
    for episode in range(episodes):
        env.deterministic = deterministic_env
        env.deterministic_seed = episode
        obs = env.reset()
        lstm_states = None
        episode_starts = torch.from_numpy(np.ones((1,), dtype=int))

        obs_shape = list(obs['player_0'].shape)
        channels = obs_shape[1]
        obs_shape[1] = channels * memory

        for t in range(50):
            obs = np.expand_dims(np.einsum('abc->cab', np.array(obs['player_0'])), 0)
            cur_obs = obs

            # print a condensed version of cur_obs, summed along 2nd axis
            # print('cur_obs', np.sum(cur_obs, axis=1).squeeze().astype(int))

            # todo: update episode starts?
            if hasattr(model, '_last_lstm_states'):
                action, lstm_states = model.predict(cur_obs, deterministic=deterministic_model,
                                                    state=lstm_states,
                                                    episode_start=episode_starts)
            else:
                action, _states = model.predict(cur_obs, deterministic=deterministic_model)

            obs, rewards, dones, info = env.step({'player_0': action})
            if dones['player_0']:
                break
            episode_starts = int(dones['player_0'])

        infos = env.infos['player_0']
        infos['r'] = rewards['player_0']
        infos['configName'] = env.configName
        infos['eval_ep'] = episode
        infos['model_ep'] = model_episode
        infos['episode_timesteps'] = t
        all_infos.append(infos)

    return _process_infos(all_infos)


def ground_truth_evals(eval_env, model, repetitions=25, memory=1, skip_to_release=True):
    df = pd.DataFrame()
    env = eval_env.unwrapped.vec_envs[0].par_env.unwrapped
    for k in range(repetitions):

        env.deterministic = True
        env.deterministic_seed = k
        env.reset()

        all_paths = env.get_all_paths(env.grid.volatile, env.instance_from_name['player_0'].pos)

        # todo: advance paths to point of divergence instead of first chance

        # advance to release
        release = int(
            get_key(env.timers, [(['release'], 1)]))  # the timestep of the subject's first 'release' timer

        all_path_infos = []

        for path in all_paths:
            total_likelihood = 0
            env.deterministic = True
            env.deterministic_seed = k
            obs = env.reset()

            episode_starts = torch.from_numpy(np.ones((1,), dtype=int))
            a = env.instance_from_name['player_0']

            taken_path = []
            lstm_states = None
            for t in range(50):

                if t < release and skip_to_release:
                    obs, rewards, dones, info = env.step({'player_0': 2})
                    continue

                act = get_relative_direction(a, path)
                cur_obs = torch.from_numpy(obs['player_0']).swapdims(0, 2).unsqueeze(0).swapdims(2, 3)

                # todo: update episode starts?
                if hasattr(model, '_last_lstm_states'):
                    value, log, entropy = model.policy.evaluate_actions(cur_obs, actions=torch.tensor(act),
                                                                        lstm_states=lstm_states,
                                                                        episode_starts=episode_starts)
                    # should get lstm states here
                else:
                    value, log, entropy = model.policy.evaluate_actions(cur_obs, actions=torch.tensor(act))

                total_likelihood += log.detach().numpy()[0]
                obs, rewards, dones, info = env.step({'player_0': act})
                taken_path += [env.instance_from_name['player_0'].pos]
                if dones['player_0']:
                    break

            infos = env.infos['player_0']
            infos['r'] = rewards['player_0']
            infos['likelihood'] = total_likelihood
            infos['configName'] = env.configName
            infos['episode_timesteps'] = t
            all_path_infos.append(infos)

        max_likelihood = max(all_path_infos, key=lambda x: x['likelihood'])['likelihood']

        prob_sum = 0
        for infos in all_path_infos:
            infos['avoidedBig'] = infos['selectedSmall'] or infos['selectedNeither']
            infos['avoidedSmall'] = infos['selectedBig'] or infos['selectedNeither']
            infos['avoidCorrect'] = (infos['avoidedBig'] == infos['shouldAvoidBig']) or (
                    infos['avoidedSmall'] == infos['shouldAvoidSmall'])
            infos['normed_likelihood'] = infos['likelihood'] - max_likelihood
            infos['probability'] = math.exp(infos['normed_likelihood'])
            infos['accuracy'] = 1 if infos['selection'] == infos['correctSelection'] else 0
            infos['weakAccuracy'] = 1 if infos['selection'] == infos['correctSelection'] or infos['selection'] == \
                                         infos['incorrectSelection'] else 0

            for i in range(5):
                infos['sel' + str(i)] = 1 if infos['selection'] == i else 0
            prob_sum += infos['probability']

        new_infos = {}
        for infos in all_path_infos:
            infos['probability'] = infos['probability'] / prob_sum
            for key in infos.keys():
                if key in ['minibatch', 'timestep']:
                    new_infos[key] = infos[key]
                    continue
                if isinstance(infos[key], bool) or isinstance(infos[key], int) or isinstance(infos[key], float):
                    value = float(infos[key])
                    key2 = key + '-c'
                    if key2 in new_infos.keys():
                        new_infos[key2] += value * infos['probability']
                    else:
                        new_infos[key2] = value * infos['probability']
                else:
                    new_infos[key] = infos[key]

        df = df.append(new_infos, ignore_index=True)
    return df


def load_checkpoint_models(path, model_class):
    full_path = os.path.join(path, 'checkpoints')
    all_models = []
    all_lengths = []
    repetition_names = []
    paths = os.scandir(full_path)
    # check if any paths are named 'rep_' followed by a number. if so, index into each and load from each
    if any([pathx.name.startswith('rep_') for pathx in paths]):
        for new_path in os.scandir(full_path):
            for checkpoint_path in os.scandir(new_path.path):
                all_models.append(model_class.load(checkpoint_path.path))
                all_lengths.append(int(checkpoint_path.path[checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")]))
                repetition_names.append(new_path.name)
    else:
        # otherwise, just load from the main folder
        for checkpoint_path in os.scandir(full_path):
            all_models.append( model_class.load(checkpoint_path.path) )
            all_lengths.append(int(checkpoint_path.path[checkpoint_path.path.find("model_")+6:checkpoint_path.path.find("_steps")]))
            repetition_names.append('rep_0')
    return all_models, all_lengths, repetition_names
