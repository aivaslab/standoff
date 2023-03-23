
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps, \
    BaseCallback
from tqdm import tqdm
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import os
import pandas as pd

def make_callbacks(save_path, env, batch_size, tqdm_steps, record_every, model, repetition=0, starting_timesteps=0):
    # train_cb = TrainUpdateCallback(envs=eval_envs + [env, ] if eval_envs[0] is not None else [env, ], batch_size=batch_size)
    # this cb updates the minibatch variable in the environment
    train_cb = TrainUpdateCallback(envs=[env, ], batch_size=batch_size, logpath=save_path, params=str(locals()),
                                   model=model)

    tqdm_cb = EveryNTimesteps(n_steps=tqdm_steps, callback=TqdmCallback(record_every=tqdm_steps))

    if repetition > 0:
        save_path_2 = os.path.join(save_path, 'checkpoints', 'rep_' + str(repetition))
    else:
        save_path_2 = os.path.join(save_path, 'checkpoints')

    checkpoints = CheckpointCallback(save_freq=record_every, save_path=save_path_2, name_prefix='model')
    checkpoints.num_timesteps = starting_timesteps
    # the above line doesn't make it start counting at the right number
    # we should update model num_timesteps instead since basecallback uses that
    model.num_timesteps = starting_timesteps


    return CallbackList([tqdm_cb, train_cb, checkpoints])


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
