import os
import argparse
import json
import pandas as pd
from tqdm import tqdm

from src.pz_envs import ScenarioConfigs
from src.utils.conversion import make_env_comp, get_json_params
from src.utils.display import make_pic_video
from src.utils.evaluation import collect_rollouts, ground_truth_evals, load_checkpoint_models
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO

class_dict = {'PPO': PPO, 'A2C': A2C, 'TRPO': TRPO, 'RecurrentPPO': RecurrentPPO}


def make_videos(train_dir, eval_envs, env_names, model, model_timestep, size):
    vidPath = os.path.join(train_dir, 'videos')
    if not os.path.exists(vidPath):
        os.mkdir(vidPath)

    for eval_env, env_name in zip(eval_envs, env_names):
        vidPath2 = os.path.join(vidPath, env_name)
        if not os.path.exists(vidPath2):
            os.mkdir(vidPath2)
        eval_env.reset()

        # make_pic_video(model, eval_env, random_policy=False, savePath=vidPath2, deterministic=False, vidName='rand_'+str(model_timestep)+'.mp4', obs_size=size )
        print('vid', vidPath2)
        make_pic_video(model, eval_env, random_policy=True, savePath=vidPath2, deterministic=True, vidName='det_'+str(model_timestep)+'.gif', obs_size=size)


def evaluate_models(eval_envs, models, model_timesteps, det_env, det_model, use_gtr, frames, episodes, train_dir, configName):
    eval_data = pd.DataFrame()

    prefix = 'gtr' if use_gtr else 'det' if det_env else 'rand'
    if not use_gtr:
        prefix += '_det' if det_model else '_rand'

    progress_bar = tqdm(total=len(models)*len(eval_envs)*episodes)
    for model, model_timestep in zip(models, model_timesteps):
        # collect rollout data
        for eval_env in eval_envs:
            if use_gtr:
                eval_data_temp = ground_truth_evals(eval_env, model, memory=frames, repetitions=episodes)
                eval_data_temp['model_ep'] = model_timestep
                eval_data = eval_data.append(eval_data_temp)
            else:
                eval_data = eval_data.append(
                    collect_rollouts(eval_env, model, model_episode=model_timestep, episodes=episodes,
                                     memory=frames, deterministic_env=det_env, deterministic_model=det_model,
                                     tqdm=progress_bar, configName=configName))

    # save all data
    evalPath = os.path.join(train_dir, 'evaluations')
    if not os.path.exists(evalPath):
        os.mkdir(evalPath)

    pathy = os.path.join(evalPath, prefix + '_data.csv')
    with open(pathy, 'wb'):
        eval_data.to_csv(pathy, index=False)


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
    args = parser.parse_args(args)

    envs = []
    for name in ScenarioConfigs.env_groups[args.env_group]:
        envs.append(f'Standoff-{name}-')
    configNames = envs
    path = args.path
    train_dirs = [f.path for f in os.scandir(path) if f.is_dir()]
    episodes = args.episodes

    renamed_envs = False
    for train_dir in train_dirs:
        model_class, size, style, frames, vecNormalize, difficulty, threads, configName = get_json_params(os.path.join(train_dir, 'json_data.json'))
        models, model_timesteps, repetition_names = load_checkpoint_models(train_dir, model_class)

        env_names = []
        if not renamed_envs:
            for k, env_name in enumerate(configNames):
                env_names.append(env_name + str(size) + '-' + style + '-' + str(difficulty) + '-v0')
            renamed_envs = True
        
        # generate eval envs with proper vecmonitors

        # det env and det seed need to be here
        eval_envs = [make_env_comp(env_name, frames=frames, size=size, style=style, monitor_path=train_dir, rank=k + 1,
                                   vecNormalize=vecNormalize, threads=threads) for k, env_name in enumerate(env_names)]

        if args.make_evals:
            evaluate_models(eval_envs, models, model_timesteps, det_env=args.det_env, det_model=args.det_model, use_gtr=args.use_gtr,
                            frames=frames, episodes=episodes, train_dir=train_dir, configName=configName)
        if args.make_vids:
            make_videos(train_dir, eval_envs, env_names, models[-1], model_timesteps[-1], size)
                        

if __name__ == '__main__':
    main(sys.argv[1:])
