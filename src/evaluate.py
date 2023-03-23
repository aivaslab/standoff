import os
import argparse
import json
import pandas as pd
from tqdm import tqdm

from src.pz_envs import ScenarioConfigs
from src.utils.conversion import make_env_comp
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

        # make_pic_video(model, eval_env, random_policy=False, savePath=vidPath2, deterministic=False, vidName='rand_'+str(model_timestep)+'.mp4', obs_size=size )
        # make_pic_video(model, eval_env, random_policy=False, savePath=vidPath2, deterministic=True, vidName='det_'+str(model_timestep)+'.mp4', obs_size=size)


def evaluate_models(eval_envs, models, model_timesteps, det_env, det_model, use_gtr, frames, episodes, train_dir):
    eval_data = pd.DataFrame()

    prefix = 'gtr' if use_gtr else 'det' if det_env else 'rand'
    if not use_gtr:
        prefix += '_det' if det_model else '_rand'

    for model, model_timestep in tqdm(zip(models, model_timesteps), total=len(models)):
        # collect rollout data
        for eval_env in eval_envs:
            if use_gtr:
                eval_data_temp = ground_truth_evals(eval_env, model, memory=frames, repetitions=episodes)
                eval_data_temp['model_ep'] = model_timestep
                eval_data = eval_data.append(eval_data_temp)
            else:
                eval_data = eval_data.append(
                    collect_rollouts(eval_env, model, model_episode=model_timestep, episodes=episodes,
                                     memory=frames, deterministic_env=det_env, deterministic_model=det_model))

    # save all data
    evalPath = os.path.join(train_dir, 'evaluations')
    if not os.path.exists(evalPath):
        os.mkdir(evalPath)

    pathy = os.path.join(evalPath, prefix + '_data.csv')
    with open(pathy, 'wb'):
        eval_data.to_csv(pathy, index=False)


def get_json_params(path):
    with open(path) as json_data:
        data = json.load(json_data)
        model_class = data['model_class']
        if model_class in class_dict.keys():
            model_class = class_dict[model_class]
        size = data['size']
        style = data['style']
        frames = data['frames']
        difficulty = data['difficulty']
        vecNormalize = data['vecNormalize'] if 'vecNormalize' in data.keys() else True
    return model_class, size, style, frames, vecNormalize, difficulty

def main(args):
    parser = argparse.ArgumentParser(description='Evaluate models on environments.')
    parser.add_argument('--env_group', type=int, help='Environment group name')
    parser.add_argument('--path', type=str, help='Path to experiment')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to run per environment')
    parser.add_argument('--det_env', type=bool, default=False, help='Deterministic environment')
    parser.add_argument('--det_model', type=bool, default=False, help='Deterministic model')
    parser.add_argument('--use_gtr', type=bool, default=False, help='Ground truth rollouts')
    args = parser.parse_args(args)

    envs = []
    for name in ScenarioConfigs.env_groups[args.env_group]:
        if 'stage' in name:
            envs.append('Standoff-S' + name[name.index('_')+1:] + '-')
        else:
            envs.append('Standoff-S3-' + name.replace(" ", "") + '-')
    env_names = envs
    path = args.path
    train_dirs = [f.path for f in os.scandir(path) if f.is_dir()]
    episodes = args.episodes

    renamed_envs = False
    for train_dir in train_dirs:
        model_class, size, style, frames, vecNormalize, difficulty = get_json_params(os.path.join(train_dir, 'json_data.json'))
        models, model_timesteps, repetition_names = load_checkpoint_models(train_dir, model_class)

        if not renamed_envs:
            for k, env_name in enumerate(env_names):
                env_names[k] = env_name + str(difficulty) + '-' + str(size) + '-' + style + '-v0'
            renamed_envs = True

        # generate eval envs with proper vecmonitors
        eval_envs = [make_env_comp(env_name, frames=frames, size=size, style='rich', monitor_path=train_dir, rank=k + 1,
                                   vecNormalize=vecNormalize) for k, env_name in enumerate(env_names)]

        evaluate_models(eval_envs, models, model_timesteps, det_env=args.det_env, det_model=args.det_model, use_gtr=args.use_gtr,
                        frames=frames, episodes=episodes, train_dir=train_dir)
                        

if __name__ == '__main__':
    main(sys.argv[1:])
