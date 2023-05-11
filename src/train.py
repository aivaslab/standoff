import argparse
import json
import os
import time
import sys

from datetime import timedelta

from .pz_envs import ScenarioConfigs
from .utils.callbacks import make_callbacks
from .utils.train_utils import init_dirs, init_policy, start_global_logs, linear_schedule, find_last_checkpoint_model
from .utils.conversion import make_env_comp, get_json_params
from stable_baselines3 import TD3, PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
import torch as th


# import multiprocessing
# multiprocessing.set_start_method("fork")


def main(args):
    parser = argparse.ArgumentParser(description='Experiment Configuration')
    # arg meaning it's curriculum learning
    parser.add_argument('--curriculum', action='store_true', help='Do curriculum learning')
    parser.add_argument('--pretrain_dir', type=str, default='', help='Folder to load models from for curriculum')

    parser.add_argument('--experiment_name', type=str, default='test', help='Name of experiment')
    parser.add_argument('--log_dir', type=str, default='/monitor', help='Logging directory')
    parser.add_argument('--savePath', type=str, default='drive/MyDrive/springExperiments/', help='Save path')
    parser.add_argument('--continuing', action='store_true', default=False, help='Whether to continue training')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite folder')
    parser.add_argument('--repetitions', type=int, default=1, help='Number of experiment repetitions')
    parser.add_argument('--timesteps', type=float, default=3e5, help='Number of timesteps per thread')
    parser.add_argument('--checkpoints', type=int, default=20, help='Number of checkpoints to save')
    # supervised model use
    parser.add_argument('--use_supervised_models', action='store_true', help='Whether to use supervised model')
    parser.add_argument('--supervised_model_data', type=str, default='random', help='Supervised model data source')
    parser.add_argument('--supervised_model_label', type=str, default='loc', help='Supervised model label to use')
    parser.add_argument('--supervised_model_path', type=str, default='supervised', help='Supervised model path')

    parser.add_argument('--env_group', type=str, default='1', help='Environment group to use')
    parser.add_argument('--style', type=str, default='rich', help='Evaluation output style')
    parser.add_argument('--size', type=int, default=17, help='View size in tiles')
    parser.add_argument('--tile_size', type=int, default=1,
                        help='Size of each tile in pixels')  # not implemented since needs registered env
    parser.add_argument('--frames', type=int, default=1, help='Number of frames to stack')
    parser.add_argument('--threads', type=int, default=16, help='Number of cpu threads to use')
    parser.add_argument('--difficulty', type=int, default=3, help='Difficulty 0-4, lower numbers enable cheats')
    parser.add_argument('--reverse_order', action='store_true', help='Whether to reverse order of train envs')
    parser.add_argument('--start_at', type=int, default=0, help='Start at a specific environment')
    parser.add_argument('--end_at', type=int, default=100, help='End at a specific environment')

    parser.add_argument('--model_class', type=str, default='PPO', help='Model class to use')
    parser.add_argument('--conv_mult', type=int, default=1, help='Number of first level kernels')
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM hidden layer size')
    parser.add_argument('--shared_lstm', type=bool, default=False, help='Share LSTM layer')
    parser.add_argument('--normalize_images', type=bool, default=False, help='Divide obs by 255')
    parser.add_argument('--width', type=int, default=32, help='MLP features')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--schedule', type=str, default='linear', help='Learning rate schedule')
    parser.add_argument('--tqdm_steps', type=int, default=256, help='Number of steps between tqdm updates')
    parser.add_argument('--buffer_size', type=int, default=2048*8,
                        help='Number of steps per thread between weight updates')
    parser.add_argument('--vecNormalize', type=bool, default=True, help='Whether to normalize the observations')
    parser.add_argument('--norm_rewards', type=bool, default=True, help='Whether to normalize the rewards')
    parser.add_argument('--variable', type=str, default='',
                        help='Variable to override with multiple values, eg "batch_norm=[True,False]", "lr=[0.001,0.0001]" ')

    args = parser.parse_args(args)
    args_string = str(args)

    if args.curriculum:
        # curriculum folder args are all in the new folder, not the pretrained folder
        pretrain_dir = args.pretrain_dir
        print('curriculum learning using', pretrain_dir)
        # load all subdirs of pretrained_folder
        all_pretrained_folders = [path for path in os.scandir(os.path.join(args.savePath, pretrain_dir)) if path.is_dir()]
    else:
        all_pretrained_folders = [None]


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

    class_dict = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO, 'RecurrentPPO': RecurrentPPO}

    global_log_path, global_logs, train_path, dir_name, continuing = init_dirs(args.savePath, args.experiment_name,
                                                                              continuing=args.continuing,
                                                                              overwrite=args.overwrite)

    recordEvery = int(args.timesteps / args.checkpoints) if args.checkpoints > 0 else 2048000000

    # if curriculum, we want to loop through each pretrained folder and update savepath2 to be a new subdirectory



    for this_pretrained_folder in all_pretrained_folders:

        if this_pretrained_folder is not None:
            this_dirname = os.path.basename(this_pretrained_folder)
            this_model, start_timestep, rep_name, vec_norm_path = find_last_checkpoint_model(this_pretrained_folder)
            train_path_ext = os.path.join(train_path, this_dirname + '-pretrained')
            os.mkdir(train_path_ext)
        else:
            train_path_ext = train_path
            this_model, start_timestep, rep_name, vec_norm_path = None, None, None, None

        if args.curriculum:
            assert this_model is not None, "curriculum learning requires a pretrained model, found none"

        for env_name_temp in envs:

            for name, value in zip(na_names, var_values):  # use for any variable to change

                # override some variable with our test case
                if hasattr(args, var_name):
                    setattr(args, var_name, value)

                rate = linear_schedule(args.lr) if args.schedule == 'linear' else args.lr
                model_class = class_dict[args.model_class]

                env_name = f"Standoff-{env_name_temp}-{str(args.size)}-{args.style}-{str(args.difficulty)}-v0"
                savePath3 = os.path.join(train_path_ext, env_name + name)
                if not os.path.exists(savePath3):
                    os.mkdir(savePath3)

                if args.curriculum:
                    model_class, _, args = get_json_params(
                        os.path.join(this_pretrained_folder, 'json_data.json'), args)
                    with open(os.path.join(savePath3, 'json_data.json'), 'w') as json_file:
                        json.dump({'model_class': model_class.__name__, 'size': args.size, 'frames': args.frames, 'style': args.style,
                                   'vecNormalize': args.vecNormalize, 'norm_rewards': args.norm_rewards, 'difficulty': args.difficulty,
                                   'threads': args.threads, 'configName': env_name_temp, 'shared_lstm': args.shared_lstm,
                                   'normalize_images': args.normalize_images,
                                   'supervised_data_source': args.supervised_data_source,
                                   'supervised_model_labels': args.supervised_model_labels,
                                   'supervised_model_path': args.supervised_model_path
                                   }, json_file)
                elif continuing:
                    model_class, _, args = get_json_params(
                        os.path.join(savePath3, 'json_data.json'), args)
                    # note that continuing will overwrite these things! It does not implement continuing under different conditions for curricula
                else:
                    with open(os.path.join(savePath3, 'json_data.json'), 'w') as json_file:
                        json.dump({'model_class': model_class.__name__, 'size': args.size, 'frames': args.frames, 'style': args.style,
                                   'vecNormalize': args.vecNormalize, 'norm_rewards': args.norm_rewards, 'difficulty': args.difficulty,
                                   'threads': args.threads, 'configName': env_name_temp, 'shared_lstm': args.shared_lstm,
                                   'normalize_images': args.normalize_images,
                                   'supervised_data_source': args.supervised_data_source,
                                   'supervised_model_labels': args.supervised_model_labels,
                                   'supervised_model_path': args.supervised_model_path
                                   }, json_file)
                print('model_class: ', model_class.__name__, 'size: ', args.size, 'style: ', args.style, 'frames: ', args.frames,
                      'vecNormalize: ', args.vecNormalize)

                for repetition in range(args.repetitions):
                    start = time.time()
                    print('name: ', name, dir_name)

                    if args.use_supervised_models:
                        print('loading SL module', args.supervised_data_source, args.supervised_model_labels,
                              args.supervised_model_path)
                        sl_module = th.load(
                            args.supervised_model_path + '/' + args.supervised_data_source + '-' +
                            args.supervised_model_label + '-model.pt')
                    else:
                        sl_module = None

                    env = make_env_comp(env_name, frames=args.frames, size=args.size, style=args.style, monitor_path=savePath3, rank=0,
                                        vecNormalize=args.vecNormalize, norm_rewards=args.norm_rewards, threads=args.threads,
                                        load_path=vec_norm_path, sl_module=sl_module)

                    policy, policy_kwargs = init_policy(model_class, env.observation_space, env.action_space, rate,
                                                        args.width, hidden_size=args.hidden_size, conv_mult=args.conv_mult, frames=args.frames,
                                                        name='cnn', net_arch=[args.width], shared_lstm=args.shared_lstm, normalize_images=args.normalize_images,)

                    log_line = start_global_logs(global_logs, args.experiment_name, dir_name, name, model_class, policy,
                                                 global_log_path)
                    if args.curriculum and this_model is not None:
                        model = model_class.load(this_model, env=env)
                    elif continuing:
                        model_path, model_timesteps = find_last_checkpoint_model(savePath3)
                        model = model_class.load(model_path, env=env)
                    else:
                        if model_class == A2C:
                            model = model_class(policy, env=env, verbose=0, learning_rate=rate,
                                                policy_kwargs=policy_kwargs)
                        else:
                            model = model_class(policy, env=env, verbose=0,
                                                learning_rate=rate,
                                                n_steps=args.buffer_size // args.threads,
                                                batch_size=args.batch_size,
                                                policy_kwargs=policy_kwargs)
                    callback = make_callbacks(savePath3, env, args.batch_size, args.tqdm_steps, recordEvery, model,
                                              repetition=repetition, threads=args.threads, learning_rate=rate,
                                              args_string=args_string)

                    print(env_name, name, savePath3, str(timedelta(seconds=time.time() - start)), policy_kwargs)
                    model.learn(total_timesteps=args.timesteps * args.threads, callback=callback)
                    print('finished, time=', str(timedelta(seconds=time.time() - start)))

                    del env
                    del model
                    del sl_module

if __name__ == 'main':


    train(parser, sys.argv[1:])
