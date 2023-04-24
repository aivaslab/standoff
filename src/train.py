import argparse
import json
import os
import time
import sys

from datetime import timedelta

from .pz_envs import ScenarioConfigs
from .utils.callbacks import make_callbacks
from .utils.train_utils import init_dirs, init_policy, start_global_logs, linear_schedule, load_last_checkpoint_model
from .utils.conversion import make_env_comp, get_json_params
from stable_baselines3 import TD3, PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
import torch as th

#import multiprocessing
#multiprocessing.set_start_method("fork")


def main(args):
    parser = argparse.ArgumentParser(description='Experiment Configuration')
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of experiment')
    parser.add_argument('--log_dir', type=str, default='/monitor', help='Logging directory')
    parser.add_argument('--savePath', type=str, default='drive/MyDrive/springExperiments/', help='Save path')
    parser.add_argument('--continuing', action='store_true', default=False, help='Whether to continue training')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite folder')
    parser.add_argument('--repetitions', type=int, default=1, help='Number of experiment repetitions')
    parser.add_argument('--timesteps', type=float, default=3e5, help='Number of timesteps per thread')
    parser.add_argument('--checkpoints', type=int, default=20, help='Number of checkpoints to save')

    parser.add_argument('--env_group', type=int, default=3, help='Environment group to use')
    parser.add_argument('--style', type=str, default='rich', help='Evaluation output style')
    parser.add_argument('--size', type=int, default=19, help='View size in tiles')
    parser.add_argument('--tile_size', type=int, default=1, help='Size of each tile in pixels') #not implemented since needs registered env
    parser.add_argument('--frames', type=int, default=1, help='Number of frames to stack')
    parser.add_argument('--threads', type=int, default=1, help='Number of cpu threads to use')
    parser.add_argument('--difficulty', type=int, default=3, help='Difficulty 0-4, lower numbers enable cheats')
    parser.add_argument('--reverse_order', action='store_true', help='Whether to reverse order of train envs')
    parser.add_argument('--start_at', type=int, default=0, help='Start at a specific environment')
    parser.add_argument('--end_at', type=int, default=100, help='End at a specific environment')

    parser.add_argument('--model_class', type=str, default='PPO', help='Model class to use')
    parser.add_argument('--conv_mult', type=int, default=1, help='Number of first level kernels')
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM hidden layer size')
    parser.add_argument('--width', type=int, default=32, help='MLP features')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--schedule', type=str, default='linear', help='Learning rate schedule')
    parser.add_argument('--tqdm_steps', type=int, default=256, help='Number of steps between tqdm updates')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per thread between weight updates')
    parser.add_argument('--vecNormalize', action='store_true', help='Whether to normalize the observations')
    parser.add_argument('--variable', type=str, default='',
                        help='Variable to override with multiple values, eg "batch_norm=[True,False]", "lr=[0.001,0.0001]" ')

    args_string = str(args)
    args = parser.parse_args(args)

    var = args.variable
    if var:
        var_name, var_values_str = var.split('=')
        var_values = eval(var_values_str)
        print(var_values)
        na_names = [f'{var_name}={var_value}' for var_value in var_values]
    else:
        var_name = 'main'
        var_values = [0]
        na_names = [f"{var_name}={var_values[0]}"]
    
    envs = []
    # f"Standoff-{configName}-{view_size}-{observation_style}-{difficulty}-v0"
    end_at = min(args.end_at, len(ScenarioConfigs.env_groups[args.env_group]))
    if args.reverse_order:
        for name in reversed(ScenarioConfigs.env_groups[args.env_group]):
            envs.append(f"{name}")
    else:
        for name in ScenarioConfigs.env_groups[args.env_group][args.start_at:end_at]:
            envs.append(f"{name}")
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timesteps = args.timesteps
    checkpoints = args.checkpoints
    repetitions = args.repetitions

    class_dict = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO, 'RecurrentPPO': RecurrentPPO}

    global_log_path, global_logs, savePath2, dir_name, continuing = init_dirs(args.savePath, args.experiment_name,
                                                                              continuing=args.continuing,
                                                                              overwrite=args.overwrite)

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
            n_steps = args.n_steps
            style = args.style
            tqdm_steps = args.tqdm_steps
            vecNormalize = args.vecNormalize
            print('vn', vecNormalize)
            model_class = class_dict[args.model_class]
            threads = args.threads
            difficulty = args.difficulty

            # override some variable with our test case
            if hasattr(args, var_name):
                setattr(args, var_name, value)

            env_name = f"Standoff-{env_name_temp}-{str(size)}-{style}-{str(difficulty)}-v0"
            savePath3 = os.path.join(savePath2, env_name + name)
            if not os.path.exists(savePath3):
                os.mkdir(savePath3)

            if continuing:
                model_class, size, style, frames, vecNormalize, difficulty, threads, _ = get_json_params(
                    os.path.join(savePath3, 'json_data.json'))
                # note that continuing will overwrite these things! It does not implement continuing under different conditions for curriculae
            else:
                with open(os.path.join(savePath3, 'json_data.json'), 'w') as json_file:
                    json.dump({'model_class': model_class.__name__, 'size': size, 'frames': frames, 'style': style,
                               'vecNormalize': vecNormalize, 'difficulty': difficulty, 'threads': threads, 'configName': env_name_temp}, json_file)
            print('model_class: ', model_class.__name__, 'size: ', size, 'style: ', style, 'frames: ', frames,
                  'vecNormalize: ', vecNormalize)

            short_name = args.experiment_name
            configName = name
            
            net_arch = [width, width] #prev one
            '''net_arch = [
                {'activation_fn': th.nn.ReLU, 'pi': [32, 32, 32, 32], 'vf': [32, 32, 32, 32]},
                {'lstm': 55},
                {'activation_fn': th.nn.ReLU, 'pi': [25], 'vf': [25]}
            ]'''
            
            for repetition in range(repetitions):
                start = time.time()
                print('name: ', name, dir_name)
                env = make_env_comp(env_name, frames=frames, size=size, style=style, monitor_path=savePath3, rank=0,
                                    vecNormalize=vecNormalize, threads=threads)
                
                policy, policy_kwargs = init_policy(model_class, env.observation_space, env.action_space, rate, 
                            width, hidden_size=hidden_size, conv_mult=conv_mult, frames=frames, name='cnn', net_arch=net_arch)
                
                log_line = start_global_logs(global_logs, short_name, dir_name, configName, model_class, policy,
                                             global_log_path)
                if continuing:
                    model, model_timesteps = load_last_checkpoint_model(savePath3, model_class)
                    model.set_env(env)
                else:
                    if model_class == A2C:
                        model = model_class(policy, env=env, verbose=0, learning_rate=rate,
                                            policy_kwargs=policy_kwargs)
                    else:
                        model = model_class(policy, env=env, verbose=0, 
                                            learning_rate=rate,
                                            n_steps=n_steps,
                                            batch_size=batch_size,
                                            policy_kwargs=policy_kwargs)
                callback = make_callbacks(savePath3, env, batch_size, tqdm_steps, recordEvery, model,
                                          repetition=repetition, threads=threads, learning_rate=rate,
                                          args_string=args_string)

                print(env_name, model_class, name, savePath3, str(timedelta(seconds=time.time() - start)), policy_kwargs)
                model.learn(total_timesteps=timesteps*threads, callback=callback)
                print('finished, time=', str(timedelta(seconds=time.time() - start)))
                
                
if __name__ == 'main':
    main(sys.argv[1:])
