import numpy as np

from .display import make_pic_video, plot_evals, plot_train
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps, BaseCallback
from tqdm.notebook import tqdm
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import time
import os
import pandas as pd

class TrainUpdateCallback(BaseCallback):

    def __init__(self, envs, batch_size):
        super().__init__()
        self.envs = envs
        self.batch_size = batch_size
        self.minibatch = 0

    def _on_rollout_end(self):
        self.minibatch += self.batch_size
        for env in self.envs:
            _env = parallel_to_aec(env.unwrapped.vec_envs[0].par_env).unwrapped
            _env.minibatch = self.minibatch

    def _on_step(self):
        return True
        

class TqdmCallback(BaseCallback):
    def __init__(self, threads=1, record_every=1, batch_size=2048):
        super().__init__()
        self.progress_bar = None
        self.iteration_size = threads*record_every
    
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
    #print('update_global_logs', path, log_line, data)
    #with open(path, 'w'):
    df = pd.read_csv(path, index_col=None)
    for key, value in data.items():
        if key not in df.columns:
            df[key] = ''
        df.loc[log_line, key] = value
    df.to_csv(path, index=False)

class PlottingCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[], eval_cbs=[], global_log_path='', log_line=-1,
                 mid_vids=False, memory=1):
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

    def _on_step(self) -> bool:
        update_global_logs(self.global_log_path, self.log_line, {
            'timesteps': np.mean(self.eval_cbs[0].evaluations_timesteps[-1]),
            'results': np.mean(self.eval_cbs[0].evaluations_results[-1]),
            'length': np.mean(self.eval_cbs[0].evaluations_length[-1]),
        })

        with open(self.logPath, 'a') as logfile:
            logfile.write(f'ts: {self.eval_cbs[0].evaluations_timesteps[-1]}\tkl: {self.model.approxkl}\n')

        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
        if self.mid_vids:
            for env, name in zip(self.envs, self.names):
                make_pic_video(self.model, env, name,
                    random_policy=False, video_length=350, savePath=os.path.join(self.savePath, 'videos', name),
                    vidName='video_'+str(self.timestep)+'-det.mp4', following="player_0", deterministic=True, memory=self.memory)
                make_pic_video(self.model, env, name,
                    random_policy=False, video_length=350, savePath=os.path.join(self.savePath, 'videos', name),
                    vidName='video_'+str(self.timestep)+'.mp4', following="player_0", deterministic=False, memory=self.memory)
        self.timestep += 1
        return True

class PlottingCallbackStartStop(BaseCallback):
    """
    # bandaid fix to EveryNTimesteps not triggering at training start and end
    """
    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[], eval_cbs=[], params=[], model=None, global_log_path='', train_name='', log_line=-1,
                 start_vid=False, memory=1):
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

    def _on_training_start(self) -> bool:
        super()._on_training_start()
        update_global_logs(self.global_log_path, self.log_line, {
            'timesteps': 0,
            'results': 0,
            'finished': False,
            'length': 0,
        })

        with open(self.logPath, 'a') as logfile:
            logfile.write(self.params)
            logfile.write("\n")
            logfile.write(str(self.model.policy))
        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
        if not os.path.exists(os.path.join(self.savePath, 'videos')):
            os.mkdir(os.path.join(self.savePath, 'videos'))
        for env, name in zip(self.envs, self.names):

            if not os.path.exists(os.path.join(self.savePath, 'videos', name)):
                os.mkdir(os.path.join(self.savePath, 'videos', name))
            if self.start_vid:
                make_pic_video(self.model, env, name,
                    random_policy=True, video_length=350, savePath=os.path.join(self.savePath, 'videos', name),
                    vidName='random.mp4', following="player_0", memory=self.memory)
        self.start_time = time.time()
        return True

    def _on_training_end(self) -> bool:
        super()._on_training_end()


        with open(self.logPath, 'a') as logfile:
            logfile.write('end of training! total time:' + str( time.time()-self.start_time) + '\n')

        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)

        update_global_logs(self.global_log_path, self.log_line, {
            'timesteps': np.mean(self.eval_cbs[0].evaluations_timesteps[-1]),
            'results': np.mean(self.eval_cbs[0].evaluations_results[-1]),
            'length': np.mean(self.eval_cbs[0].evaluations_length[-1]),
            'finished': True,
        })
            
        for env, name in zip(self.envs, self.names):
            make_pic_video(self.model, env, name, 
                random_policy=False, video_length=350, savePath=os.path.join(self.savePath, 'videos', name),
                vidName='end.mp4', following="player_0", deterministic=False, memory=self.memory)
            make_pic_video(self.model, env, name,
                random_policy=False, video_length=350, savePath=os.path.join(self.savePath, 'videos', name),
                vidName='end-det.mp4', following="player_0", deterministic=True, memory=self.memory)
        plot_train(self.savePath, self.name, 0, self.train_name+'train')


        return True

    def _on_step(self):
        pass
