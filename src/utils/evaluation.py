import os

import pandas as pd
import torch
import math
import numpy as np
from typing import Dict, Any


def _process_info(infos: Dict[str, Any]) -> Dict[str, Any]:
    infos['avoidedBig'] = infos['selectedSmall'] or infos['selectedNeither']
    infos['avoidedSmall'] = infos['selectedBig'] or infos['selectedNeither']
    infos['avoidCorrect'] = (infos['avoidedBig'] and infos['shouldAvoidBig']) or (
            infos['avoidedSmall'] and infos['shouldAvoidSmall'])
    infos['accuracy'] = 1 if infos['selection'] == infos['correctSelection'] else 0
    infos['weakAccuracy'] = 1 if infos['selection'] == infos['correctSelection'] or infos['selection'] == infos[
        'incorrectSelection'] else 0
    for i in range(5):
        infos[f'sel' + str(i)] = 1 if infos['selection'] == i else 0

    return infos


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
                     deterministic_model: bool = False,
                     max_timesteps: int = 50,
                     tqdm=None, configName: str = ''):
    #normalizer_env = env
    #unwrapped_envs = [x.par_env.unwrapped for x in env.unwrapped.vec_envs]
    #configName = #unwrapped_envs[0].configName
    #print(env.unwrapped.__dict__)
    #using_pipe = env.unwrapped.pipes
    #env = Log(env)
    num_threads = len(env.unwrapped.pipes)
    all_infos = []
    #tracemalloc.start()
    for episode in range(episodes // num_threads):
        # code to change seed here. can't use unwrapped.
        if deterministic_env:
            for pipe in env.unwrapped.pipes:
                pipe.send(("set_attr", "deterministic_seed", episode))
                pipe.send(("set_attr", "deterministic", True))
        
        # deterministic env breaks this for some reason.
        #snapshot = tracemalloc.take_snapshot()
        #for stat in snapshot.statistics('lineno')[:10]:
        #    print('x', stat)
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((num_threads,))
        episode_timesteps = [0] * num_threads
        active_envs = set(range(num_threads))
        all_actions = ['' for _ in range(num_threads)]
        
        for t in range(max_timesteps):
                
            if hasattr(model, '_last_lstm_states'):
                action, lstm_states = model.predict(obs, deterministic=deterministic_model,
                                                    state=lstm_states,
                                                    episode_start=episode_starts)
                obs, rewards, dones, info = env.step(action)
                episode_starts = dones
            else:
                action, _ = model.predict(obs, deterministic=deterministic_model)
                obs, rewards, dones, info = env.step(action)
            # action is a list of actions for each thread
            # here we append to all_actions
            for i in range(num_threads):
                all_actions[i] += str(action[i])

            #print('obs0', obs[0], 'action0', action[0], 'reward0', rewards[0], 'done0', dones[0], 'info0', info[0])
                
            completed_envs = []
            print(info[0])
            if not active_envs:
                print('breaking')
                break
            for i in active_envs:
                if dones[i]:
                    completed_envs.append(i)
                    infos = info[i]
                    infos['r'] = rewards[i]
                    infos['all_actions'] = all_actions[i]
                    infos['configName'] = configName
                    infos['eval_ep'] = episode
                    infos['model_ep'] = model_episode
                    infos['episode_timesteps'] = t
                    infos['terminal_observation'] = 0 # overwrite huge thing
                    infos['episode'] = 0 # otherwise contains a dict {r, l, t}
                    all_infos.append(_process_info(infos))
            active_envs -= set(completed_envs)

        del info
        tqdm.update(num_threads)

    return all_infos


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


def find_checkpoint_models(path):
    full_path = os.path.join(path, 'checkpoints')
    all_models = []
    all_lengths = []
    repetition_names = []
    norm_paths = []
    paths = os.scandir(full_path)
    # check if any paths are named 'rep_' followed by a number. if so, index into each and load from each
    if any([pathx.name.startswith('rep_') for pathx in paths]):
        for new_path in os.scandir(full_path):
            for checkpoint_path in os.scandir(new_path.path):
                if "norm" not in checkpoint_path.path:
                    all_models.append(checkpoint_path.path)
                    all_lengths.append(int(checkpoint_path.path[checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps.zip")]))
                    repetition_names.append(new_path.name)
                else:
                    norm_paths.append(checkpoint_path.path)
    else:
        # otherwise, just load from the main folder
        for checkpoint_path in os.scandir(full_path):
            # print('loading from ' + checkpoint_path.path)
            if ".pkl" not in checkpoint_path.path:
                all_models.append( checkpoint_path.path)
                all_lengths.append(int(checkpoint_path.path[checkpoint_path.path.find("model_")+6:checkpoint_path.path.find("_steps.zip")]))
                repetition_names.append('rep_0')
            else:
                norm_paths.append(checkpoint_path.path)

    return all_models, all_lengths, repetition_names, norm_paths
