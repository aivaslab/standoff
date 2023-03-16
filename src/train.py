import argparse
import json
import os
import time
from datetime import timedelta

from .pz_envs import ScenarioConfigs
from .utils.callbacks import make_callbacks
from standoff_models.train import init_dirs, init_policy, start_global_logs, linear_schedule
from .utils.conversion import make_env_comp, get_json_params
from stable_baselines3 import TD3, PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
import torch as th


def load_last_checkpoint_model(path, model_class):
    full_path = os.path.join(path, 'checkpoints')
    best_model = None
    best_length = 0
    paths = os.scandir(full_path)
    rep_folders = [pathx for pathx in paths if pathx.is_dir() and pathx.name.startswith('rep_')]
    if rep_folders:
        for rep_folder in rep_folders:
            for checkpoint_path in os.scandir(rep_folder.path):
                if int(checkpoint_path.path[
                       checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")]) > best_length:
                    best_length = int(
                        checkpoint_path.path[
                        checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")])
                    best_model = model_class.load(checkpoint_path.path)
    else:
        for checkpoint_path in os.scandir(full_path):
            if int(checkpoint_path.path[
                   checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")]) > best_length:
                best_length = int(
                    checkpoint_path.path[checkpoint_path.path.find("model_") + 6:checkpoint_path.path.find("_steps")])
                best_model = model_class.load(checkpoint_path.path)

    return best_model, best_length


parser = argparse.ArgumentParser(description='Experiment Configuration')
parser.add_argument('--experiment_name', type=str, default='test', help='Name of experiment')
parser.add_argument('--model_class', type=str, default='PPO', help='Model class to use')
parser.add_argument('--continuing', type=bool, default=False, help='Whether to continue training')
parser.add_argument('--env_group', type=int, default=3, help='Environment group to use')
parser.add_argument('--repetitions', type=int, default=1, help='Number of experiment repetitions')
parser.add_argument('--conv_mult', type=int, default=1, help='Number of first level kernels')
parser.add_argument('--frames', type=int, default=1, help='Number of frames')
parser.add_argument('--timesteps', type=float, default=3e5, help='Number of timesteps')
parser.add_argument('--hidden_size', type=int, default=64, help='LSTM size (unused)')
parser.add_argument('--width', type=int, default=32, help='MLP features')
parser.add_argument('--size', type=int, default=19, help='Board size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--checkpoints', type=int, default=20, help='Number of checkpoints to save')
parser.add_argument('--schedule', type=str, default='linear', help='Learning rate schedule')
parser.add_argument('--style', type=str, default='rich', help='Evaluation output style')
parser.add_argument('--n_steps', type=int, default=256, help='Number of steps')
parser.add_argument('--vecNormalize', type=bool, default=True, help='Whether to normalize the vectors')
parser.add_argument('--experimentName', type=str, default='direct-p2', help='Experiment name')
parser.add_argument('--log_dir', type=str, default='/monitor', help='Logging directory')
parser.add_argument('--savePath', type=str, default='drive/MyDrive/springExperiments/', help='Save path')
parser.add_argument('--variable', type=str, nargs='+', default=[],
                    help='Variable to override with multiple values, eg "batch_norm=[True,False]", "lr=[0.001,0.0001]" ')
parser.add_argument('--repetitions', type=int, default=1, help='Number of experiment repetitions')

args = parser.parse_args()

var = args.variable
var_name, var_values = var.split('=')
var_values = var_values.split(',')
na_names = [str(var_name) + str(var_value) for var_value in var_values]

envs = ['Standoff-S3-' + name.replace(" ", "") + '-0' for name in ScenarioConfigs.stageNames[args.env_group]]
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
timesteps = args.timesteps
checkpoints = args.checkpoints
repetitions = args.repetitions

class_dict = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO, 'RecurrentPPO': RecurrentPPO}

if __name__ == '__main__':
    # todo: load hparams from a file somewhere
    global_log_path, global_logs, savePath2, dir_name, continuing = init_dirs(args.savePath, args.experiment_name,
                                                                              continuing=args.continuing)

    recordEvery = int(timesteps / checkpoints) if checkpoints > 0 else 2048000000

    for env_name_temp in envs:

        for name, value in zip(na_names, var_values):  # use for any variable to change

            conv_mult = args.conv_mult
            frames = args.frames
            hidden_size = args.hidden_size
            width = args.width
            size = args.size
            rate = linear_schedule(args.lr) if args.schedule == 'linear' else args.lr
            batch_size = args.batch_size
            style = args.style
            n_steps = args.n_steps
            vecNormalize = args.vecNormalize
            model_class = class_dict[args.model_class]

            # override some variable with our test case
            if hasattr(args, var_name):
                setattr(args, var_name, value)

            env_name = env_name_temp + '-' + str(size) + '-' + style + '-v0'
            savePath3 = os.path.join(savePath2, env_name + name)
            if not os.path.exists(savePath3):
                os.mkdir(savePath3)

            if continuing:
                model_class, size, style, frames, vecNormalize = get_json_params(
                    os.path.join(savePath3, 'json_data.json'))
            else:
                with open(os.path.join(savePath3, 'json_data.json'), 'w') as json_file:
                    json.dump({'model_class': model_class.__name__, 'size': size, 'frames': frames, 'style': style,
                               'vecNormalize': vecNormalize}, json_file)
            print('model_class: ', model_class.__name__, 'size: ', size, 'style: ', style, 'frames: ', frames,
                  'vecNormalize: ', vecNormalize)

            short_name = args.experiment_name
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
                    model.set_env(env)
                else:
                    if model_class == A2C:
                        model = model_class(policy, env=env, verbose=0, learning_rate=rate,
                                            policy_kwargs=policy_kwargs)
                    else:
                        model = model_class(policy, env=env, verbose=0, learning_rate=rate,
                                            batch_size=batch_size,
                                            policy_kwargs=policy_kwargs)
                callback = make_callbacks(savePath3, env, batch_size, n_steps, recordEvery, model,
                                          repetition=repetition)

                print(env_name, model_class, name, savePath3, str(timedelta(seconds=time.time() - start)))
                model.learn(total_timesteps=timesteps, callback=callback)
                print('finished, time=', str(timedelta(seconds=time.time() - start)))
