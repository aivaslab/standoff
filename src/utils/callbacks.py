import numpy as np

from .display import make_pic_video, plot_evals_df, plot_train, plot_evals, gtr_to_monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps, \
    BaseCallback
from tqdm.notebook import tqdm
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import time
import os
import pandas as pd
import torch
import math
import sys


def make_callbacks(save_path, env, batch_size, n_steps, record_every, model):
    # train_cb = TrainUpdateCallback(envs=eval_envs + [env, ] if eval_envs[0] is not None else [env, ], batch_size=batch_size)
    # this cb updates the minibatch variable in the environment
    train_cb = TrainUpdateCallback(envs=[env, ], batch_size=batch_size, logpath=save_path, params=str(locals()),
                                   model=model)

    tqdm_cb = EveryNTimesteps(n_steps=n_steps, callback=TqdmCallback(record_every=n_steps))

    checkpoints = CheckpointCallback(save_freq=record_every, save_path=os.path.join(save_path, 'checkpoints'),
                                     name_prefix='model')

    return CallbackList([tqdm_cb, train_cb, checkpoints])


def make_callbacks_legacy(savePath3, name, configName, eval_envs, eval_params, n_steps, size, frames, recordEvery,
                          global_log_path, log_line, vids=False):
    eval_df = pd.DataFrame()
    eval_cbs = [EvalCallback(eval_env, best_model_save_path=os.path.join(savePath3, 'logs', 'best_model'),
                             log_path=os.path.join(savePath3, 'logs'), eval_freq=recordEvery,
                             n_eval_episodes=eval_eps,
                             deterministic=True, render=False, verbose=0) for eval_env in eval_envs]
    tqdm_cb = EveryNTimesteps(n_steps=n_steps, callback=TqdmCallback(record_every=n_steps))
    plot_cb = EveryNTimesteps(n_steps=recordEvery, callback=
    PlottingCallback(savePath=savePath3, name=name, envs=eval_envs,
                     names=eval_params, eval_cbs=eval_cbs, verbose=0, log_line=log_line,
                     global_log_path=global_log_path, memory=frames, eval_df=eval_df, gtr=True,
                     mid_vids=False, image_size=size))
    plot_cb_extra = PlottingCallbackStartStop(savePath=savePath3, name=name,
                                              envs=eval_envs, names=eval_params, eval_cbs=eval_cbs, verbose=0,
                                              params=str(locals()), model=model, global_log_path=global_log_path,
                                              train_name=configName, log_line=log_line, memory=frames, eval_df=eval_df,
                                              gtr=True, start_vid=False, end_vid=vids, image_size=size)
    train_cb = TrainUpdateCallback(envs=eval_envs + [env, ] if eval_envs[0] is not None else [env, ],
                                   batch_size=batch_size)

    return CallbackList([CallbackList(eval_cbs), plot_cb, plot_cb_extra, tqdm_cb, train_cb])


class TrainUpdateCallback(BaseCallback):

    def __init__(self, envs, batch_size, logpath, params, model):
        super().__init__()
        self.envs = envs
        self.batch_size = batch_size
        self.minibatch = 0
        self.logPath = os.path.join(logpath, 'logs.txt')
        self.params = params
        self.model = model

    def _on_training_start(self):
        with open(self.logPath, 'a') as logfile:
            logfile.write(self.params)
            logfile.write("\n")
            logfile.write(str(self.model.policy))

    def _on_rollout_end(self):
        # triggered before updating the policy
        self.minibatch += self.batch_size
        for env in self.envs:
            _env = parallel_to_aec(env.unwrapped.vec_envs[0].par_env).unwrapped
            _env.minibatch = self.minibatch

    def _on_training_end(self):
        with open(self.logPath, 'a') as logfile:
            logfile.write('finished training')

    def _on_step(self):
        return True


class TqdmCallback(BaseCallback):
    def __init__(self, threads=1, record_every=1, batch_size=2048):
        super().__init__()
        self.progress_bar = None
        self.iteration_size = threads * record_every

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        self.progress_bar.update(self.iteration_size)
        return True

    def _on_rollout_start(self):
        self.progress_bar.update(self.iteration_size)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None


def update_global_logs(path, log_line, data):
    # print('update_global_logs', path, log_line, data)
    # with open(path, 'w'):
    df = pd.read_csv(path, index_col=None)
    for key, value in data.items():
        if key not in df.columns:
            df[key] = ''
        df.loc[log_line, key] = value
    df.to_csv(path, index=False)


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


def collect_rollouts(env, model, model_episode, episodes=100, memory=1, deterministic=False):
    df = pd.DataFrame()
    env = env.unwrapped.vec_envs[0].par_env.unwrapped
    # print(env.configName)
    all_infos = []
    for episode in range(episodes):
        env.deterministic = deterministic
        env.deterministic_seed = episode
        obs = env.reset()

        lstm_states = None
        episode_starts = torch.from_numpy(np.ones((1,), dtype=int))
        a = env.instance_from_name['player_0']

        obs_shape = list(torch.from_numpy(obs['player_0']).swapdims(0, 2).unsqueeze(0).shape)
        channels = obs_shape[1]
        obs_shape[1] = channels * memory
        remembered_obs = torch.zeros(obs_shape)

        for t in range(50):

            obs = torch.from_numpy(obs['player_0']).swapdims(0, 2).unsqueeze(0)
            remembered_obs = torch.cat([obs, remembered_obs], dim=1)
            cur_obs = remembered_obs[:, -memory * channels:, :, :]

            cur_obs = np.array(cur_obs)

            # todo: update episode starts?
            if hasattr(model, '_last_lstm_states'):
                action, _states = model.predict(cur_obs, deterministic=deterministic,
                                                lstm_states=model._last_lstm_states,
                                                episode_starts=episode_starts)
            else:
                action, _states = model.predict(cur_obs, deterministic=deterministic)

            obs, rewards, dones, info = env.step({'player_0': action})
            #accumulate reward here?
            
            if dones['player_0']:
                break
        infos = env.infos['player_0']
        infos['r'] = rewards['player_0']
        infos['configName'] = env.configName
        infos['eval_ep'] = episode
        infos['model_ep'] = model_episode
        all_infos.append(infos)
        # print('appended, episode', episode)

    for infos in all_infos:
        infos['avoidedBig'] = infos['selectedSmall'] or infos['selectedNeither']
        infos['avoidedSmall'] = infos['selectedBig'] or infos['selectedNeither']
        infos['avoidCorrect'] = (infos['avoidedBig'] == infos['shouldAvoidBig']) or (
                    infos['avoidedSmall'] == infos['shouldAvoidSmall'])
        infos['accuracy'] = 1 if infos['selection'] == infos['correctSelection'] else 0
        infos['weakAccuracy'] = 1 if infos['selection'] == infos['correctSelection'] or infos['selection'] == infos[
            'incorrectSelection'] else 0

    df = pd.DataFrame(all_infos, index=np.arange(len(all_infos)))
    return df


def ground_truth_evals(eval_envs, model, repetitions=25, memory=1):
    df = pd.DataFrame()
    for env in eval_envs:
        env = env.unwrapped.vec_envs[0].par_env.unwrapped
        for k in range(repetitions):

            env.deterministic = True
            env.deterministic_seed = k
            env.reset()

            all_paths = env.get_all_paths(env.grid.volatile, env.instance_from_name['player_0'].pos)

            # todo: advance paths to point of divergence instead of first chance

            # advance to release
            release = int(
                get_key(env.timers, [(['release'], 1)]))  # the timestep of the subject's first 'release' timer

            '''for _ in range(release):
                obs, rewards, dones, info = env.step({'player_0': 2})
                #todo: update LSTM states to use on envs'''

            all_path_infos = []

            for path in all_paths:
                total_likelihood = 0
                env.deterministic = True
                env.deterministic_seed = k
                obs = env.reset()

                lstm_states = None
                episode_starts = torch.from_numpy(np.ones((1,), dtype=int))
                a = env.instance_from_name['player_0']

                taken_path = []
                obs_shape = list(torch.from_numpy(obs['player_0']).swapdims(0, 2).unsqueeze(0).shape)
                # obs shape 1,3,51,51
                channels = obs_shape[1]
                obs_shape[1] = channels * memory
                remembered_obs = torch.zeros(obs_shape)
                for t in range(50):

                    act = get_relative_direction(a, path)

                    obs = torch.from_numpy(obs['player_0']).swapdims(0, 2).unsqueeze(0)
                    remembered_obs = torch.cat([obs, remembered_obs], dim=1)
                    cur_obs = remembered_obs[:, -memory * channels:, :, :]

                    # todo: update episode starts?
                    if hasattr(model, '_last_lstm_states'):
                        value, log, entropy = model.policy.evaluate_actions(cur_obs, actions=torch.tensor(act),
                                                                            lstm_states=model._last_lstm_states,
                                                                            episode_starts=episode_starts)
                    else:
                        value, log, entropy = model.policy.evaluate_actions(cur_obs, actions=torch.tensor(act))

                    total_likelihood += log.detach().numpy()[0]
                    obs, rewards, dones, info = env.step({'player_0': act})
                    taken_path += [env.instance_from_name['player_0'].pos]
                    if dones['player_0']:
                        '''print(t, taken_path[release:])
                        grid2 = env.grid.volatile.astype(int)
                        for pos in taken_path[release:]:
                            #print('pos', pos[0], pos[1])
                            grid2[pos[0], pos[1]] += 10
                        print(grid2)
                        print(env.infos['player_0'])'''
                        break

                infos = env.infos['player_0']
                infos['r'] = rewards['player_0']
                infos['likelihood'] = total_likelihood
                infos['configName'] = env.configName
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

    # print('gtr', df.head()) #does work
    return df


def plotting_evals(self, vids=False, plots=True, fig_folder=''):
    if not self.gtr:
        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
    else:
        # inplace concat
        temp = pd.concat([self.eval_df, ground_truth_evals(self.envs, self.model, memory=self.memory)],
                         ignore_index=True)
        self.eval_df.drop(self.eval_df.index[0:], inplace=True)
        self.eval_df[temp.columns] = temp

        if plots:
            plot_evals_df(self.eval_df, fig_folder, self.name)

    if not os.path.exists(os.path.join(self.savePath, 'videos')):
        os.mkdir(os.path.join(self.savePath, 'videos'))
    for env, name in zip(self.envs, self.names):

        if vids:
            if not os.path.exists(os.path.join(self.savePath, 'videos', name)):
                os.mkdir(os.path.join(self.savePath, 'videos', name))
            make_pic_video(self.model, env, name,
                           random_policy=False, video_length=350,
                           obs_size=self.image_size,
                           savePath=os.path.join(self.savePath, 'videos', name),
                           vidName='video_' + str(self.timestep) + '-det.mp4', following="player_0",
                           deterministic=True, memory=self.memory)
            make_pic_video(self.model, env, name,
                           random_policy=False, video_length=350,
                           obs_size=self.image_size,
                           savePath=os.path.join(self.savePath, 'videos', name),
                           vidName='video_' + str(self.timestep) + '.mp4', following="player_0",
                           deterministic=False, memory=self.memory)


class PlottingCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[], eval_cbs=[], global_log_path='', log_line=-1,
                 mid_vids=False, memory=1, eval_df=None, gtr=False, image_size=32):
        super().__init__(verbose)
        self.savePath = savePath
        self.logPath = os.path.join(savePath, 'logs.txt')
        self.global_log_path = global_log_path
        self.name = name
        self.envs = envs
        self.names = names
        self.eval_cbs = eval_cbs
        self.timestep = 0
        self.log_line = log_line
        self.mid_vids = mid_vids
        self.memory = memory
        self.eval_df = eval_df
        self.gtr = gtr
        self.image_size = image_size

    def _on_step(self) -> bool:
        with open(self.logPath, 'a') as logfile:
            logfile.write(f'ts: {self.eval_cbs[0].evaluations_timesteps[-1]}\n')
            # logfile.write(f'ts: {self.eval_cbs[0].evaluations_timesteps[-1]}\tkl: {self.model.approxkl}\n')

        plotting_evals(self, self.mid_vids, plots=False)
        self.timestep += 1
        return True


class PlottingCallbackStartStop(BaseCallback):
    """
    # bandaid fix to EveryNTimesteps not triggering at training start and end
    """

    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[], eval_cbs=[], params=[], model=None,
                 global_log_path='', train_name='', log_line=-1,
                 start_vid=False, end_vid=True, memory=1, eval_df=None, gtr=False, image_size=32):
        super().__init__(verbose)
        self.savePath = savePath
        self.global_log_path = global_log_path
        self.logPath = os.path.join(savePath, 'logs.txt')
        self.name = name
        self.envs = envs
        self.names = names
        self.eval_cbs = eval_cbs
        self.start_time = 0
        self.params = params
        self.model = model
        self.train_name = train_name
        self.log_line = log_line
        self.start_vid = start_vid
        self.memory = memory
        self.eval_df = eval_df
        self.gtr = gtr
        self.end_vid = end_vid
        self.image_size = image_size
        self.timestep = 0

    def _on_training_start(self) -> bool:
        super()._on_training_start()
        update_global_logs(self.global_log_path, self.log_line, {
            'timesteps': 0,
            'results': 0,
            'finished': False,
            'length': 0,
        })

        plotting_evals(self, vids=self.start_vid, plots=False)
        self.start_time = time.time()
        return True

    def _on_step(self) -> bool:
        self.timestep += 1
        return True

    def _on_training_end(self) -> bool:
        super()._on_training_end()

        fig_folder = os.path.join(self.savePath, 'figs')
        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)

        try:
            with open(self.logPath, 'a') as logfile:
                logfile.write('end of training! total time:' + str(time.time() - self.start_time) + '\n')

            update_global_logs(self.global_log_path, self.log_line, {
                'timesteps': np.mean(self.eval_cbs[0].evaluations_timesteps[-1]),
                'results': np.mean(self.eval_cbs[0].evaluations_results[-1]),
                'length': np.mean(self.eval_cbs[0].evaluations_length[-1]),
                'finished': True,
            })
        except Exception as e:
            print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), e)
        try:
            plotting_evals(self, vids=self.end_vid, plots=True, fig_folder=fig_folder)
        except Exception as e:
            print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), e)
        try:
            gtr_to_monitor(self.savePath, self.eval_df, self.envs)
        except Exception as e:
            print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), e)
        try:
            plot_train(self.savePath, fig_folder, self.name, 0, self.train_name + 'train')
        except Exception as e:
            print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), e)

        return True

    def _on_step(self):
        pass
