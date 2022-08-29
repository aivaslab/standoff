import os
from utils.conversion import make_env
import logging


def train_model(name, train_env, eval_envs, eval_params,
                player_config,
                framework, policy, learning_rate=1e-5,
                evals=25, total_timesteps=1e6, eval_eps=25,
                batch_size=32, memory=1, size=64, reduce_color=False,
                threads=1, saveModel=True, saveVids=True, savePics=True,
                saveEval=True, saveTrain=True,
                savePath="drive/MyDrive/model/", reward_decay=True,
                extractor_features=32,
                policy_kwargs=None):
    if policy_kwargs is None:
        policy_kwargs = {}
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    savePath = os.path.join(savePath, name)
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    recordEvery = int(total_timesteps / evals) if evals > 0 else 1000

    train_env = make_env(train_env[0], player_config, train_env[1], memory=memory, threads=threads,
                         reduce_color=reduce_color, size=size, reward_decay=reward_decay,
                         path=savePath, vecMonitor=True)

    eval_envs = [make_env(env_name, player_config, env_param, memory=memory, threads=threads,
                          reduce_color=reduce_color, size=size, saveVids=saveVids, path=savePath,
                          recordEvery=recordEvery, reward_decay=reward_decay) for env_name, env_param in
                 zip(eval_envs, eval_params)]
    model = framework(policy, train_env, learning_rate=learning_rate,
                      n_steps=batch_size, tensorboard_log="logs", policy_kwargs=policy_kwargs)
    name = str(name + model.policy_class.__name__)

    model.learn(total_timesteps=total_timesteps,
                tb_log_name=name, reset_num_timesteps=True
                )

    if saveModel:
        model.save(os.path.join(savePath, name))
    return train_env
