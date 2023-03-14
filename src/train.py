import json
import os
import time
from datetime import timedelta

from src.pz_envs import ScenarioConfigs
from src.utils.callbacks import make_callbacks
from train import init_dirs, init_policy, start_global_logs, linear_schedule
from utils.conversion import make_env, make_env_comp
import logging
from stable_baselines3 import TD3, PPO, A2C
from sb3_contrib import RecurrentPPO, TRPO

def load_last_checkpoint_model(path, model_class):
    full_path = os.path.join(path, 'checkpoints')
    best_model = None
    best_length = 0
    paths = os.scandir(full_path)
    if any([pathx.name.startswith('rep_') for pathx in paths]):
        for new_path in os.scandir(full_path):
            for checkpoint_path in os.scandir(new_path.path):
                if int(checkpoint_path.path[
                       checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")]) > best_length:
                    best_length = int(
                        checkpoint_path.path[checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")])
                    best_model = model_class.load(checkpoint_path.path)
    else:
        for checkpoint_path in os.scandir(full_path):
            if int(checkpoint_path.path[
                   checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")]) > best_length:
                best_length = int(
                    checkpoint_path.path[checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")])
                best_model = model_class.load(checkpoint_path.path)

    return best_model, best_length


envs = ['Standoff-S3-' + name.replace(" ","") + '-0' for name in ScenarioConfigs.stageNames[3]][6:]
#envs = ['Standoff-S3-' + name.replace(" ","") + '-0-v0' for name in ['swapped']]

#envs = ['Standoff-S1-0']
model_class = PPO
na_names = ["batch_norm", 'no_norm']
batch_norms = [True, False]
#na_names = ["ppo" + name.replace(" ","") for name in ScenarioConfigs.stageNames[3]]
repetitions = 1

log_dir = "/monitor"
os.makedirs(log_dir, exist_ok=True)
savePath = "drive/MyDrive/springExperiments/"
conv_mult = 1 # number of first level kernels = conv_mult*channels
frames = 1
timesteps = 3e5
hidden_size = 64 # lstm size (unused)
width = 32 # mlp features
size = 19
rate = linear_schedule(1e-3)
batch_size = 128
evals = 20
style = 'rich'
n_steps = 256 #currently just used for tqdm
vecNormalize = True

experimentName = "direct-p2"

if __name__ == '__main__':
    # todo: load hparams from a file somewhere
    global_log_path, global_logs, savePath2, dir_name, continuing = init_dirs(savePath, experimentName, continuing=False)

    recordEvery = int(timesteps / evals) if evals > 0 else 2048000000

    for env_name_temp in envs:
        for name, vecNormalize in zip(na_names, batch_norms):  # use for any variable to change
            env_name = env_name_temp + '-' + str(size) + '-' + style + '-v0'
            savePath3 = os.path.join(savePath2, env_name + name)
            if not os.path.exists(savePath3):
                os.mkdir(savePath3)

            with open(os.path.join(savePath3, 'json_data.json'), 'w') as json_file:
                json.dump({'model_class': model_class.__name__, 'size': size, 'frames': frames, 'style': style,
                           'vecNormalize': vecNormalize}, json_file)

            short_name = experimentName
            configName = name
            policy, policy_kwargs = init_policy(model_class, width, hidden_size, conv_mult, frames, name='cnn',
                                                net_arch=[dict(activation_fn=th.nn.ReLU, pi=[width], vf=[width])])
            for repetition in range(repetitions):
                start = time.time()
                print('name: ', name, dir_name)
                log_line = start_global_logs(global_logs, short_name, dir_name, configName, model_class, policy,
                                             global_log_path)
                env = make_env_comp(env_name, frames=frames, size=size, style=style, monitor_path=savePath3, rank=0,
                                    vecNormalize=vecNormalize)
                if continuing:
                    model, model_timesteps = load_last_checkpoint_model(savePath3, model_class)
                else:
                    model = model_class(policy, env=env, verbose=0, learning_rate=rate,
                                        # batch_size=batch_size, #a2c has no batch size
                                        policy_kwargs=policy_kwargs)
                callback = make_callbacks(savePath3, env, batch_size, n_steps, recordEvery, model, repetition=repetition)

                print(env_name, model_class, name, savePath3, str(timedelta(seconds=time.time() - start)))
                model.learn(total_timesteps=timesteps, callback=callback)
                print('finished, time=', str(timedelta(seconds=time.time() - start)))