import os
from .conversion import make_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps, BaseCallback
from .callbacks import TqdmCallback, PlottingCallback, PlottingCallbackStartStop, LoggingCallback
import logging

def train_model(name, train_env, eval_envs, eval_params,
                player_config,
                framework, policy, learning_rate=1e-5, 
                evals=25, total_timesteps=1e6, eval_eps=25,
                batch_size=32, memory=1, size=64, reduce_color=False,
                threads=1, saveModel=True, saveVids=True, savePics=True, 
                saveEval=True, saveTrain=True,
                savePath="drive/MyDrive/model/", reward_decay=True,
                extractor_features=32):

    if not os.path.exists(savePath):
        os.mkdir(savePath)
    savePath = os.path.join(savePath, name)
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    recordEvery = int(total_timesteps/evals) if evals > 0 else 1000
    policy_kwargs = dict(
        # change model hyperparams if cnn
        features_extractor_kwargs=dict(features_dim=extractor_features ),
        ) if policy[0]=='C' else {}
    
    train_env = make_env(train_env[0], player_config, train_env[1], memory=memory, threads=threads,
                         reduce_color=reduce_color, size=size, reward_decay=reward_decay,
                         path=savePath)

    eval_envs = [make_env(env_name, player_config, env_param, memory=memory, threads=threads, 
                          reduce_color=reduce_color, size=size, saveVids=saveVids, path=savePath, 
                          recordEvery=recordEvery, reward_decay=reward_decay) for env_name, env_param in 
                          zip(eval_envs, eval_params)]
    name = str(name+model.policy_class.__name__)
    model = framework(policy, train_env, learning_rate=learning_rate, 
                      n_steps=batch_size, tensorboard_log="logs", policy_kwargs=policy_kwargs)

    eval_cbs = [EvalCallback(eval_env, best_model_save_path=os.path.join(savePath,'/logs/best_model'),
                             log_path=os.path.join(savePath,'/logs/'), eval_freq=recordEvery,
                             n_eval_episodes=eval_eps,
                             deterministic=True, render=False, verbose=0) for eval_env in eval_envs]
    plot_cb = EveryNTimesteps(n_steps=recordEvery, callback=
                PlottingCallback(savePath=savePath, name=name, envs=eval_envs, 
                    names=eval_params, eval_cbs=eval_cbs, verbose=0))
    plot_cb_extra = PlottingCallbackStartStop(savePath=savePath, name=name,
                envs=eval_envs, names=eval_params, eval_cbs=eval_cbs, verbose=0,
                params=str(locals()))
    tqdm_cb = EveryNTimesteps(n_steps=batch_size, callback=
                TqdmCallback(threads=threads, record_every=batch_size))

    model.learn(total_timesteps=total_timesteps, 
                tb_log_name=name, reset_num_timesteps=True, callback=
                [eval_cbs, plot_cb, plot_cb_extra, tqdm_cb]
                )

    if saveModel:
        model.save(os.path.join(savePath, name))
    return train_env
