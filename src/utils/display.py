import matplotlib.pyplot as plt
# %matplotlib inline
#from IPython import display
import os
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import get_monitor_files
import numpy as np
import pandas as pd
from scipy.stats import entropy
import json
import imageio
import cv2
import copy

def entropy_func(x):
    value_counts = x.value_counts(normalize=True)
    return entropy(value_counts)

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

def entropy_multi_cols(x, y, z):
    return entropy([x, y, z])

def agg_dict(frame):
    ignored_columns = ['configName', 'minibatch', 'model_ep']
    return {col: (['mean', 'std'] if col != 'selection' else ['mean', 'std', entropy_func]) if col not in ignored_columns else 'first' for col in frame.columns}



def group_dataframe(cur_df, groupby_list):
    ret_df = cur_df.groupby(groupby_list, as_index=False).agg(agg_dict(cur_df))
    ret_df.columns = ret_df.columns.map("_".join).str.replace('_first', '')
    ret_df = ret_df.sort_values('model_ep')

    # get the number of episodes that are aggregated todo: verify that this works as intended
    ret_df['num_episodes_grouped'] = ret_df.groupby(groupby_list)['model_ep'].transform('count')
    
    ret_df['treat_entropy'] = ret_df.apply(lambda row: entropy_multi_cols(row['selectedBig_mean'], row['selectedSmall_mean'], row['selectedNeither_mean']), axis=1)
    
    return ret_df


def process_csv(path, gtr=False):
    with open(path) as file_handler:
        print('opened', path, file_handler)
        
        df = pd.read_csv(file_handler, index_col=None, on_bad_lines='skip')
        df['index_col'] = df.index
        if gtr:
            for key_name in ['shouldAvoidBig',
                             'shouldAvoidSmall',
                             'selection',
                             'selectedBig',
                             'selectedSmall',
                             'selectedNeither',
                             'r',
                             'accuracy',
                             'weakAccuracy',
                             'episode_timesteps',
                             'sel0',
                             'sel1',
                             'sel2',
                             'sel3',
                             'sel4',
                             ]:
                df[key_name] = df[key_name + '-c']

        df["selectedBig"] = pd.to_numeric(df["selectedBig"])
        df["selectedSmall"] = pd.to_numeric(df["selectedSmall"])
        df["selectedNeither"] = pd.to_numeric(df["selectedNeither"])
        df["selection"] = df["selection"].fillna(-1)

        df["valid"] = df["selection"] != -1

        df_small = df[df["selectedBig"].notnull()]
        # df_small is the set of episodes with valid outcomes
        if not len(df_small):
            print('small df had no good samples... using big in lieu')
            df_small = df

        df_small.loc[:, "avoidedBig"] = (df_small["selectedSmall"].astype(bool) | \
                df_small["selectedNeither"].astype(bool))
        df_small.loc[:, "avoidedSmall"] = (df_small["selectedBig"].astype(bool) | \
                        df_small["selectedNeither"].astype(bool))
        df_small.loc[:, "avoidCorrect"] = ((df_small["avoidedBig"] ==  \
                        df_small["shouldAvoidBig"].astype(bool)) | \
                        (df_small["avoidedSmall"] == df_small["shouldAvoidSmall"].astype(bool)))




        grouped_df = group_dataframe(df, ['model_ep',
                                          'configName'])  # the mean and std at each evaluation... for each configname
        #print('cols', grouped_df.columns)
        grouped_df_small = group_dataframe(df_small, ['model_ep', 'configName'])
        grouped_df_noname_abs = group_dataframe(df, ['model_ep'])
        grouped_df_noname = group_dataframe(df_small, ['model_ep'])

        # save json file containing the maximum model_ep value and num_episodes_grouped:
        max_model_ep = grouped_df_noname['model_ep'].max()
        num_episodes_grouped = grouped_df_noname['num_episodes_grouped'].max()
        with open(os.path.join(os.path.dirname(path), 'max_model_ep.json'), 'w') as f:
            json.dump({'max_model_ep': int(max_model_ep), 'num_episodes_grouped': int(num_episodes_grouped)}, f)

        return grouped_df, grouped_df_small, grouped_df_noname_abs, grouped_df_noname

def get_transfer_matrix_row(path, column, config_names, config_names_old, only_last=False):
    grouped_df, _, _, _ = process_csv(path, gtr=False)

    if only_last:
        grouped_df = grouped_df.loc[grouped_df.groupby('configName')['model_ep'].idxmax()]

    #if config_names is not None:
    filtered_df = grouped_df[grouped_df['configName'].isin(config_names)]
    filtered_df_2 = grouped_df[grouped_df['configName'].isin(config_names_old)]
    
    print(len(filtered_df), len(filtered_df_2))
    if len(filtered_df) >= len(filtered_df_2):
        pass
    else:
        filtered_df = filtered_df_2
    #else:
    #    filtered_df = grouped_df

    pivot_table = pd.pivot_table(filtered_df, values=column, index='model_ep', columns='configName', aggfunc='first')
    return_matrix = pivot_table.reset_index().fillna(0)
    timesteps = return_matrix['model_ep'].tolist()
    return_matrix = return_matrix.drop(columns=['model_ep'])

    return return_matrix, timesteps
    
def get_transfer_matrix_row_legacy(path, column, only_last=False):
    grouped_df, _, _, _ = process_csv(path, gtr=False)
    return_matrix = pd.DataFrame()

    # for return matrix, we use the set of rows of the final (or all) model from each environment
    # each configName is an evaluation configuration
    
    labels = []
    timesteps = []
    for uname in grouped_df.configName.unique():
        rows = grouped_df[grouped_df['configName'] == uname]
        print('rows', uname, rows)
        if only_last:
            rows2 = rows[rows['model_ep'] == rows['model_ep'].max()][column]
            labels.append(uname)
            timesteps.append(rows['model_ep'].max())
            return_matrix = return_matrix.append(rows2, ignore_index=True)
        else:
            print('unique', len(rows.model_ep.unique()))
            for model_ep in rows.model_ep.unique():
                rows2 = rows[rows['model_ep'] == model_ep][column]
                print('rows2', model_ep, rows2)
                labels.append(uname)
                timesteps.append(model_ep)
                return_matrix = return_matrix.append(rows2, ignore_index=True)
                
    return return_matrix, labels, timesteps


def make_pic_video(model, env, random_policy=False, video_length=50, savePath='', vidName='video.mp4',
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
        img = env.render(mode='rgb_array')

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
            img = env.render(mode='rgb_array')
        cv2.putText(img=img, text=str(action), org=(0, image_size), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 255, 255), thickness=2)
        if dones[following]:
            obs = _env.reset()
            _env.reset()
            if not use_global_obs:
                img = cv2.resize(obs[following], dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
            else:
                img = env.render(mode='rgb_array')
            cv2.putText(img=img, text=str(action), org=(0, image_size), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)

    imageio.mimsave(os.path.join(savePath, vidName), [img for i, img in enumerate(images)], fps=10)


def gtr_to_monitor(savePath, df, envs):
    for cf in df.configName.unique():
        # get rank from envs using configName
        # start by identifying which env has this configName
        rank = -1
        for env in envs:
            _env = env.unwrapped.vec_envs[0].par_env.unwrapped
            if _env.configName == cf:
                rank = env.rank
                break
        header = {'n': cf, 'rank': rank}
        filename = os.path.join(savePath, cf + '-gtr.monitor.csv')
        print('saving monitor', filename)

        new_df = df[df['configName'] == cf]

        with open(filename, "w") as f:
            f.write(f"#{json.dumps(header)}\n" + new_df.to_csv())

'''
def show_state(env, step=0, info=""):
    plt.figure(3)
    display.display(plt.clf())
    plt.imshow(env.render(mode='human'))
    plt.title("%s | Step: %d %s" % (str(env.__class__.__name__), step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())'''
