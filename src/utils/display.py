import matplotlib.pyplot as plt
# %matplotlib inline
from IPython import display
import os
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import get_monitor_files
import numpy as np
import pandas
import json
import imageio
import cv2


def load_results_tempfix(path: str) -> pandas.DataFrame:
    # something causes broken csvs, here we ignore extra data
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            # cols = pandas.read_csv(file_handler, nrows=1).columns
            data_frame = pandas.read_csv(file_handler, index_col=None, on_bad_lines='skip')  # , usecols=cols)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values("t", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    for _position, _value in enumerate(values):
        try:
            _new_value = float(_value)
        except ValueError:
            _new_value = 0.0
        values[_position] = _new_value
    return np.convolve(values.astype(float), weights, 'valid')


def plot_train(log_folder, title='Learning Curve', window=50):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results_tempfix(log_folder), 'timesteps')
    y = moving_average(y, window=window)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title(title + " Smoothed")

    plt.savefig(os.path.join(log_folder, title + str(len(x))), bbox_inches='tight')
    plt.close(fig)
    # plt.show()


def make_pic_video(model, env, name, random_policy=False, video_length=50, savePath='', vidName='video.mp4', following="player_0", image_size=320):

    env = parallel_to_aec(env.unwrapped.vec_envs[0].par_env).unwrapped
    images = []
    obs = env.reset()
    env.reset()
    img = cv2.resize(env.observations[following], dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)

    for i in range(video_length):
        images.append(img)
        if random_policy:
            action = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        else:
            action = {following: model.predict(obs)}
        obs, _, _, _ = env.step(action)
        img = cv2.resize(obs[following], dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)

    imageio.mimsave(os.path.join(savePath, vidName), [img for i, img in enumerate(images)], fps=30)
    env.reset()


def plot_evals(savePath, name, names, eval_cbs):
    fig, axs = plt.subplots(1)
    for env_name, cb in zip(names, eval_cbs):
        plt.plot(cb.evaluations_timesteps, [np.mean(x) for x in cb.evaluations_results], label=env_name, )
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.title(name)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(savePath, name + '_evals'), bbox_inches='tight')
    plt.close(fig)


def show_state(env, step=0, info=""):
    plt.figure(3)
    display.display(plt.clf())
    plt.imshow(env.render(mode='human'))
    plt.title("%s | Step: %d %s" % (str(env.__class__.__name__), step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())
