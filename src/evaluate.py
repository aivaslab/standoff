import os
import argparse
import pandas as pd
from tqdm import tqdm


from src.pz_envs import ScenarioConfigs
from src.utils.conversion import make_env_comp, get_json_params
from src.utils.display import make_pic_video
from src.utils.evaluation import collect_rollouts, ground_truth_evals, find_checkpoint_models
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize

class_dict = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO, 'RecurrentPPO': RecurrentPPO}


def make_videos(train_dir, env_names, short_names, model_path, model_class, model_timestep, size, norm_path, env_kwargs):
    vidPath = os.path.join(train_dir, 'videos')
    if not os.path.exists(vidPath):
        os.mkdir(vidPath)
    for k, (short_name, env_name) in enumerate(zip(short_names, env_names)):
        env_kwargs["threads"] = 1
        eval_env = make_env_comp(env_name, rank=k+1, load_path=norm_path, **env_kwargs)
        model = model_class.load(model_path, eval_env)
        model.set_env(eval_env)
        vidPath2 = os.path.join(vidPath, env_name)
        if not os.path.exists(vidPath2):
            os.mkdir(vidPath2)
        eval_env.reset()

        # make_pic_video(model, eval_env, random_policy=False, savePath=vidPath2, deterministic=False, vidName='rand_'+str(model_timestep)+'.mp4', obs_size=size )
        print('vid', vidPath2)
        make_pic_video(model, eval_env, random_policy=False, savePath=vidPath2, deterministic=True, vidName='det_'+str(model_timestep)+'.gif', obs_size=size)
        del eval_env
        del model


def evaluate_models(eval_env_names, short_names, models, model_class, model_timesteps, det_env, det_model, use_gtr, frames, episodes, train_dir, norm_paths, env_kwargs):
    eval_data = pd.DataFrame()

    prefix = 'gtr' if use_gtr else 'det' if det_env else 'rand'
    if not use_gtr:
        prefix += '_det' if det_model else '_rand'

    progress_bar = tqdm(total=len(models)*len(eval_env_names)*episodes)

    for k, (short_name, eval_env_name) in enumerate(zip(short_names, eval_env_names)):
        env = make_env_comp(eval_env_name, rank=k+1, skip_vecNorm=True, **env_kwargs)
        for model_name, model_timestep, norm_path in zip(models, model_timesteps, norm_paths):
            eval_env = VecNormalize.load(norm_path, env) if env_kwargs["vecNormalize"] else env
            model = model_class.load(model_name, env=eval_env)

            if use_gtr:
                eval_data_temp = ground_truth_evals(eval_env, model, memory=frames, repetitions=episodes)
                eval_data_temp['model_ep'] = model_timestep
                eval_data = eval_data.append(eval_data_temp)
            else:
                eval_data = eval_data.append(
                    collect_rollouts(eval_env, model, model_episode=model_timestep, episodes=episodes,
                                     memory=frames, deterministic_env=det_env, deterministic_model=det_model,
                                     tqdm=progress_bar, configName=short_name))
            #del model

            # save all data
            evalPath = os.path.join(train_dir, 'evaluations')
            if not os.path.exists(evalPath):
                os.mkdir(evalPath)

            pathy = os.path.join(evalPath, prefix + '_data.csv')
            with open(pathy, 'wb'):
                eval_data.to_csv(pathy, index=False)
                
            del model
            del eval_env
        del env
                
            


def main(args):
    parser = argparse.ArgumentParser(description='Evaluate models on environments.')
    parser.add_argument('--env_group', type=int, help='Environment group name')
    parser.add_argument('--path', type=str, help='Path to experiment')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to run per environment')
    parser.add_argument('--det_env', action='store_true', help='Deterministic environment')
    parser.add_argument('--det_model', action='store_true', help='Deterministic model')
    parser.add_argument('--use_gtr', action='store_true', help='Ground truth rollouts')
    parser.add_argument('--make_vids', action='store_true', help='Make vids')
    parser.add_argument('--make_evals', action='store_true', help='Make eval csvs')
    parser.add_argument('--curriculum', action='store_true', help='Do evals one directory deeper')
    args = parser.parse_args(args)

    configNames = []
    short_names = []
    for name in ScenarioConfigs.env_groups[args.env_group]:
        configNames.append(f'Standoff-{name}-')
        short_names.append(name)
    path = args.path
    episodes = args.episodes


    renamed_envs = False
    env_names = []

    if args.curriculum:
        all_trained_folders = [p for p in os.scandir(path) if p.is_dir()]
    else:
        all_trained_folders = [None]

    for this_pretrained_folder in all_trained_folders:
        train_dirs = [f.path for f in os.scandir(this_pretrained_folder) if f.is_dir()]

        for train_dir in train_dirs:
            print('evaluating at train_dir', train_dir)
            model_class, size, style, frames, vecNormalize, norm_rewards, difficulty, threads, configName = get_json_params(os.path.join(train_dir, 'json_data.json'))
            model_paths, model_timesteps, repetition_names, norm_paths = find_checkpoint_models(train_dir)
            if not model_paths:
                continue

            if not renamed_envs:
                for k, env_name in enumerate(configNames):
                    env_names.append(env_name + str(size) + '-' + style + '-' + str(difficulty) + '-v0')
                renamed_envs = True

            env_kwargs = {"size": size, "style": style, "threads": threads, "frames": frames, "monitor_path": train_dir, "vecNormalize": vecNormalize, "norm_rewards": norm_rewards}

            if args.make_evals:
                evaluate_models(env_names, short_names, model_paths, model_class, model_timesteps, det_env=args.det_env, det_model=args.det_model, use_gtr=args.use_gtr,
                                frames=frames, episodes=episodes, train_dir=train_dir, norm_paths=norm_paths, env_kwargs=env_kwargs)
            if args.make_vids:
                make_videos(train_dir, env_names, short_names, model_paths[-1], model_class, model_timesteps[-1], size, norm_path=norm_paths[-1], env_kwargs=env_kwargs)
                        

if __name__ == '__main__':
    main(sys.argv[1:])
