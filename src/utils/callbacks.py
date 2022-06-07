from .display import make_pic_video, plot_evals, plot_train
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps, BaseCallback
from tqdm.notebook import tqdm
import logging
import time

class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__(threads=1, record_every=1)
        self.progress_bar = None
        self.iteration_size = threads*record_every
    
    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])
    
    def _on_step(self):
        self.progress_bar.update(iteration_size) 
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None

class PlottingCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[], eval_cbs=[]):
        super(PlottingCallback, self).__init__(verbose)
        self.savePath = savePath
        self.logPath = os.path.join(savePath, 'logs.txt')
        self.name = name
        self.envs = envs
        self.names = names
        self.eval_cbs = eval_cbs

    def _on_step(self) -> bool:
        with open(logPath, 'w') as logfile:
            logfile.write('accuracies and stuff')
            #todo: add custom cb for special infos (e.g. avoidance%, bigR%)

        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
        for env, name in zip(self.envs, self.names):
            make_pic_video(self.model, env, name, False, True, self.savePath)
        return True

class PlottingCallbackStartStop(BaseCallback):
    """
    # bandaid fix to EveryNTimesteps not triggering at training start and end
    """
    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[], eval_cbs=[], params=[]):
        super(PlottingCallbackStartStop, self).__init__(verbose)
        self.savePath = savePath
        self.logPath = os.path.join(savePath, 'logs.txt')
        self.name = name
        self.envs = envs
        self.names = names
        self.eval_cbs = eval_cbs
        self.start_time = 0
        self.params = params

    def _on_training_start(self) -> bool:
        super(PlottingCallbackStartStop, self)._on_training_start()

        with open(self.logPath, 'w') as logfile:
            logfile.write(self.params)
        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
        for env, name in zip(self.envs, self.names):
            make_pic_video(self.model, env, name, False, True, self.savePath)
        self.start_time = time.time()
        return True

    def _on_training_end(self) -> bool:
        super(PlottingCallbackStartStop, self)._on_training_end()

        with open(logPath, 'w') as logfile:
            logfile.write('end of training! total time:', time.time()-self.start_time)

        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
            
        for env, name in zip(self.envs, self.names):
            make_pic_video(self.model, env, name, False, True, self.savePath)
        plot_train(savePath, name+'train')
        return True