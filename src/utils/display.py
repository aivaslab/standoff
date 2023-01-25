import matplotlib.pyplot as plt
# %matplotlib inline
from IPython import display
import os
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import get_monitor_files
import numpy as np
import pandas as pd
import json
import imageio
import cv2
import copy


def load_results_tempfix(path: str) -> pd.DataFrame:
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
            data_frame = pd.read_csv(file_handler, index_col=None, on_bad_lines='skip')  # , usecols=cols)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pd.concat(data_frames)
    # data_frame.sort_values("t", inplace=True)
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


def plot_split(indexer, df, mypath, title, window, values=None):
    if values is None:
        values = ["accuracy"]
    new_df = df.pivot(index=indexer, columns="name", values=values)
    fig = plt.figure(title)
    new_df.plot()
    plt.xlabel('Timestep, (window={})'.format(window))
    plt.ylabel(values)
    plt.xlim(0, plt.xlim()[1])
    plt.ylim(0, 1)
    plt.title(title + " " + values[0])
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    name = title + values[0]
    plt.savefig(os.path.join(mypath, name))
    plt.close()


def plot_merged(indexer, df, mypath, title, window, values=None,
                labels=None, range=[0, 1]):
    if labels is None:
        labels = ["selected any box", "selected best box"]
    if values is None:
        values = ["valid", "accuracy"]
    fig = plt.figure(title)
    plt.ylim(range[0], range[1])
    for value, label in zip(values, labels):
        plt.plot(df[indexer], df[value], label=label)
    plt.xlim(0, plt.xlim()[1])
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 1))
    plt.xlabel('Timestep, (window={})'.format(window))
    plt.ylabel('Percent')
    plt.title(title + " " + values[0])
    plt.tight_layout()
    plt.savefig(os.path.join(mypath, title + "-merged-" + values[0]))
    plt.close()


def plot_selection(indexer, df, mypath, title, window):
    fig = plt.figure(title)
    plt.xlabel('Timestep, (window={})'.format(window))
    plt.ylabel('Reward Type')
    plt.plot([], [], label='SelectedBig', color='green')
    plt.plot([], [], label='SelectedSmall', color='blue')
    plt.plot([], [], label='SelectedNeither', color='orange')
    plt.stackplot(df[indexer], df["selectedBig"], df["selectedSmall"], df["selectedNeither"],
                  colors=['green', 'blue', 'orange', ])
    plt.xlim(0, plt.xlim()[1])
    plt.legend(['Big', 'Small', 'Neither'])
    plt.title(title + "-" + str('Reward Type'))
    plt.savefig(os.path.join(mypath, title + "-" + 'reward-type'))
    plt.close()


def plot_train(log_folder, fig_folder, configName, rank, title='Learning Curve', window=2048):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    monitor_files = get_monitor_files(log_folder)

    if not len(monitor_files):
        print(f"No monitor files found in {log_folder}")
        pass

    merged_df = pd.DataFrame()
    merged_df_small = pd.DataFrame()
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    for file_name in monitor_files:
        #print('generating figs from', file_name)
        # if file_name != os.path.join(log_folder, configName + "-" + str(rank) + ".monitor.csv"):
        #    continue
        title2 = os.path.basename(file_name).replace('.', '-')[:-12]
        rank = title2[-1]
        #print(os.path.basename(file_name), title2, rank)
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#", print(first_line)
            metadata = json.loads(first_line[1:].replace('""', ''))
            rank = metadata['rank'] if 'rank' in metadata.keys() else -1
            header = json.loads(first_line[1:])
            # cols = pandas.read_csv(file_handler, nrows=1).columns
            df = pd.read_csv(file_handler, index_col=None, on_bad_lines='skip')  # , usecols=cols)
            df['index_col'] = df.index

            if "t" in df.columns:
                df["t"] += header["t_start"]
                df['gtr'] = False
            else:
                # this is a gtr monitor file
                # remove -c from all columns
                df['gtr'] = True
                df.rename(columns={x: x[:-2] if x[-2:] == "-c" else x for x in df.columns}, inplace=True)

            df["selectedBig"] = pd.to_numeric(df["selectedBig"])
            df["selectedSmall"] = pd.to_numeric(df["selectedSmall"])
            df["selectedNeither"] = pd.to_numeric(df["selectedNeither"])

            df["selection"] = df["selection"].fillna(-1)

            '''
            # we already have these values for gtr files
            df["accuracy"] = df["selection"] == df["correctSelection"]
            # print(df.columns)
            df["weakAccuracy"] = df.apply(
                lambda row: row["selection"] == row["incorrectSelection"] or row["selection"] == row[
                    "correctSelection"], axis=1)
            df["valid"] = df["selection"] != -1
            df["accuracy"] = pd.to_numeric(df["accuracy"])
            df["valid"] = pd.to_numeric(df["valid"])
            '''

            #temp, as gtr should generate this ideally
            df["valid"] = df["selection"] != -1


            df.minibatch = df.minibatch.astype(int)

            filter = df["selection"].notnull()
            filter2 = df["selectedBig"].notnull()
            dfSmall = df[filter2]

            if not len(dfSmall):
                print('small df', file_name, 'had no good samples')
                continue


            # ValueError: Wrong number of items passed 18, placement implies 1;
            # maybe happens when no good data, so exit if so?
            dfSmall["avoidedBig"] = dfSmall.apply(lambda row: row["selectedSmall"] or row["selectedNeither"], axis=1)
            dfSmall["avoidedSmall"] = dfSmall.apply(lambda row: row["selectedBig"] or row["selectedNeither"], axis=1)

            dfSmall["avoidCorrect"] = dfSmall.apply(lambda row: (row["avoidedBig"] == row["shouldAvoidBig"]) or (
                    row["avoidedSmall"] == row["shouldAvoidSmall"]), axis=1)

            grouped = df.groupby('minibatch', as_index=False).mean().sort_values('minibatch')
            #print(df['accuracy'])
            groupedSmall = dfSmall.groupby('minibatch', as_index=False).mean().sort_values('minibatch')

            # small is valid rows

            grouped['name'] = title2
            groupedSmall['name'] = title2
            grouped['eval'] = (rank != 0)
            groupedSmall['eval'] = (rank != 0)
            merged_df = merged_df.append(grouped)
            merged_df_small = merged_df_small.append(groupedSmall)

            # dr = df.rolling(real_window).mean()
            # drSmall = dfSmall.rolling(real_window).mean()

            plot_selection(indexer='minibatch', df=groupedSmall, mypath=fig_folder, title=title2, window=window)

    #print('lenny', len(merged_df), merged_df.columns, len(merged_df_small), merged_df_small.columns)
    merged_df = merged_df.groupby(['minibatch', 'eval', 'gtr'], as_index=False).mean().sort_values('minibatch')
    merged_df_small = merged_df_small.groupby(['minibatch', 'name', 'eval', 'gtr'], as_index=False).mean().sort_values(
        'minibatch')
    merged_df_noname = merged_df_small.groupby(['minibatch', 'eval', 'gtr'], as_index=False).mean().sort_values(
        'minibatch')
    #print('lenny2', len(merged_df), merged_df.columns, len(merged_df_small), merged_df_small.columns)
    #print(merged_df_noname.head)

    for use_train in [True, False]:
        title_base = 'train' if use_train else 'eval'
        for use_gtr in [True, False]:
            if use_train and use_gtr:
                # gtr is only used for evals
                continue
            title = title_base + ('-gtr' if use_gtr else '-rollout')

            #print(merged_df['accuracy'])
            merged_df_f = merged_df[(merged_df['eval'] != use_train) & (merged_df['gtr'] == use_gtr)]
            merged_df_small_f = merged_df_small[
                (merged_df_small['eval'] != use_train) & (merged_df_small['gtr'] == use_gtr)]
            merged_df_noname_f = merged_df_noname[
                (merged_df_noname['eval'] != use_train) & (merged_df_noname['gtr'] == use_gtr)]

            # normalize r after all the filtering
            merged_df_f["r"] = (merged_df_f["r"] - merged_df_f["r"].min()) / (
                        merged_df_f["r"].max() - merged_df_f["r"].min())

            plotted = ['Plotted']
            not_plotted = ['Could not plot']

            if len(merged_df_f):
                title2 = title + '-f'
                plot_merged(indexer='minibatch', df=merged_df_f, mypath=fig_folder, title=title2, window=window,
                            values=['valid', 'weakAccuracy', 'accuracy', 'r'],
                            labels=['selected any box', 'selected any treat', 'selected correct treat',
                                    'reward (normalized)'])
                plotted += ['merged_df_f '+ title2]
            else:
                not_plotted += ['merged_df_f '+ title2]
            if len(merged_df_small_f):
                title2 = title + '-smf'
                # this graph is the same as above, but only taking into account valid samples
                # avoidcorrect should be identical to weakaccuracy when no opponent is present
                plot_split(indexer='minibatch', df=merged_df_small_f, mypath=fig_folder, title=title2,
                           window=window,
                           values=["accuracy",])
                plot_split(indexer='minibatch', df=merged_df_small_f, mypath=fig_folder, title=title2,
                           window=window,
                           values=["weakAccuracy"])
                plot_split(indexer='minibatch', df=merged_df_small_f, mypath=fig_folder, title=title2,
                           window=window,
                           values=["avoidCorrect"])
                plotted += ['merged_df_small_f '+ title2]
            else:
                not_plotted += ['merged_df_small_f '+ title2]
            if len(merged_df_noname_f):
                title2 = title + '-nnf'
                plot_merged(indexer='minibatch', df=merged_df_noname_f, mypath=fig_folder, title=title2, window=window,
                            values=["avoidCorrect"], labels=["avoided correct box"])
                plot_selection(indexer='minibatch', df=merged_df_noname_f, mypath=fig_folder, title=title2, window=window)
                plotted += ['merged_df_noname_f '+ title2]
            else:
                not_plotted += ['merged_df_noname_f '+ title2]
            '''if len(plotted) > 1:
                print(' '.join(plotted))
            if len(not_plotted) > 1:
                print(' '.join(not_plotted))'''


def make_pic_video(model, env, name, random_policy=False, video_length=50, savePath='', vidName='video.mp4',
                   following="player_0", image_size=320, deterministic=False, memory=1, obs_size=32):
    """
    make a video of the model playing the environment
    """
    _env = parallel_to_aec(env.unwrapped.vec_envs[0].par_env).unwrapped
    use_global_obs = _env.observation_style == 'rich'
    images = []
    obs = _env.reset()
    # reshape obs[following] to be channels, height, width

    _env.reset()
    if not use_global_obs:
        img = cv2.resize(obs[following], dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
    else:
        img = env.render(mode='rgb')

    for i in range(video_length):
        images.append(img)
        if random_policy:
            action = {agent: _env.action_spaces[agent].sample() for agent in _env.agents}
        else:
            obs_used = cv2.resize(obs[following], dsize=(obs_size, obs_size), interpolation=cv2.INTER_NEAREST)
            obs_used = np.transpose(obs_used, (2, 0, 1))

            action = {following: model.predict(obs_used, deterministic=deterministic)[0] if memory <= 1 else
                    model.predict(obs_used, deterministic=deterministic)}
        obs, _, dones, _ = _env.step(action)

        if not use_global_obs:
            img = cv2.resize(obs[following], dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
        else:
            img = env.render(mode='rgb')
        cv2.putText(img=img, text=str(action), org=(0, image_size), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 255, 255), thickness=2)
        if dones[following]:
            obs = _env.reset()
            _env.reset()
            if not use_global_obs:
                img = cv2.resize(obs[following], dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
            else:
                img = env.render(mode='rgb')
            cv2.putText(img=img, text=str(action), org=(0, image_size), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)

    imageio.mimsave(os.path.join(savePath, vidName), [img for i, img in enumerate(images)], fps=10)


def plot_evals(savePath, name, names, eval_cbs):
    """
    plot the results of the evaluations of the model on the environments
    """
    fig, axs = plt.subplots(1)
    for env_name, cb in zip(names, eval_cbs):
        plt.plot(cb.evaluations_timesteps, [np.mean(x) for x in cb.evaluations_results], label=env_name, )
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.title(name)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(savePath, name + '_evals'), bbox_inches='tight')
    plt.close(fig)


def gtr_to_monitor(savePath, df, envs):
    for cf in df.configName.unique():
        #get rank from envs using configName
        #start by identifying which env has this configName
        rank = -1
        for env in envs:
            if env.configName == cf:
                rank = env.rank
                break
        header = {'n': cf, 'rank': rank}
        filename = os.path.join(savePath, cf + '-gtr.monitor.csv')
        print('saving monitor', filename)

        new_df = df[df['configName'] == cf]

        with open(filename, "w") as f:
            f.write(f"#{json.dumps(header)}\n" + new_df.to_csv())


def plot_evals_df(df, savePath, name):
    """
    given dataframe of rollouts, plot things
    """
    #print('plotting evals df', name, df.columns)
    for column in ['accuracy', 'avoidCorrect', 'weakAccuracy', 'r']:
        fig, axs = plt.subplots(1)
        unique_names = df.configName.unique()
        for cf in unique_names:
            df2 = df[df['configName'] == cf].groupby('minibatch', as_index=False).mean().sort_values('minibatch')
            plt.plot(df2.minibatch, df2[column + '-c'], label=cf, )
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.title(name)
        plt.xlabel('Timestep')
        plt.ylabel(column)
        plt.savefig(os.path.join(savePath, name + '_evals-gtr'), bbox_inches='tight')
        plt.close(fig)


def show_state(env, step=0, info=""):
    plt.figure(3)
    display.display(plt.clf())
    plt.imshow(env.render(mode='human'))
    plt.title("%s | Step: %d %s" % (str(env.__class__.__name__), step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())
