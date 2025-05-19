import ast
import glob
import hashlib
import itertools
import math
import pickle
import re

import sys
import os
import time

import numpy as np
from functools import lru_cache

from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from ablation_configs import *

import pandas as pd
import heapq
from scipy.stats import sem, t

def calculate_statistics(df, last_epoch_df, params, skip_3x=True, skip_2x1=False, key_param=None, skip_1x=True, record_delta_pi=False, used_regimes=None, savepath=None, last_timestep=True):
    '''
    Calculates various statistics from datasets of model outputs detailing model performance.
    '''
    avg_loss = {}
    variances = {}
    ranges_1 = {}
    ranges_2 = {}
    range_dict = {}
    range_dict3 = {}

    check_labels = ['p-s-0', 'target', 'delay', 'b-loc', 'p-b-0', 'p-b-1', 'p-s-1', 'shouldAvoidSmall', 'shouldGetBig', 'vision', 'loc']

    sub_regime_keys = [
        "Nn", "Fn", "Nf", "Tn", "Nt", "Ff", "Tf", "Ft", "Tt"
    ]

    

    direct = [x + '1' for x in sub_regime_keys]
    progression_groups = {}
    progression_groups['s1'] = [x + '0' for x in sub_regime_keys]
    progression_groups['s2'] = ['Tt1']
    progression_groups['s21'] = ['Nn1a', 'Nt1a'] 
    progression_groups['s3'] = [x for x in direct if x != 'Tt1' and x not in ['Nn1a', 'Nt1a']] + ['Nn1b', 'Nt1b']

    group_mapping = {}
    for group_name, regimes_list in progression_groups.items():
        for regime in regimes_list:
            group_mapping[regime] = group_name

    print("All test_regime values:", last_epoch_df['test_regime'].unique())
    print("Mapping keys:", list(group_mapping.keys()))
    

    last_epoch_df['test_group'] = last_epoch_df['test_regime'].apply(
        lambda x: group_mapping.get(x, '?')
    )

    params.append('test_group')


    print('calculating statistics...')
    for col in last_epoch_df.columns:
        if col in check_labels:
            last_epoch_df[col] = last_epoch_df[col].apply(extract_last_value_from_list_str)

    print('making categorical')
    for param in params:
        if last_epoch_df[param].dtype == 'object':
            try:
                last_epoch_df[param] = last_epoch_df[param].astype('category')
            except TypeError:
                try:
                    last_epoch_df[param] = last_epoch_df[param].apply(tuple)
                    last_epoch_df[param] = last_epoch_df[param].astype('category')
                except TypeError:
                    print('double error on', param)
                    pass
                pass
    params = [param for param in params if param not in ['delay', 'perm']]

    # print('params:', params)

    param_pairs = itertools.combinations(params, 2)
    param_triples = itertools.combinations(params, 3)

    variable_columns = last_epoch_df.select_dtypes(include=[np.number]).nunique().index[
        last_epoch_df.select_dtypes(include=[np.number]).nunique() > 1].tolist()

    correlations = last_epoch_df[variable_columns + ['accuracy']].corr()
    target_correlations = correlations['accuracy'][variable_columns]
    stats = {
        'param_correlations': correlations,
        'accuracy_correlations': target_correlations,
        'vars': variable_columns
    }
    if used_regimes:
        print('calculating regime size')
        regime_lengths = {}
        grouped = last_epoch_df.groupby(['regime'])['accuracy'].mean()
        print(grouped)
        for regime_item in used_regimes:
            regime_length = 0
            for data_name in regime_item:
                pathy = os.path.join('supervised/', data_name, 'params.npz')
                regime_items = np.load(pathy, mmap_mode='r')['arr_0']
                regime_length += len(regime_items)
            regime_lengths[regime_item[0][3:]] = regime_length
            print(regime_item, regime_length)
        if False:
            plot_regime_lengths(regime_lengths, grouped, savepath + 'scatter.png')

    unique_vals = {param: last_epoch_df[param].unique() for param in params}
    print('found unique vals', unique_vals)
    for par, val in unique_vals.items():
        print(par, type(val[0]))
    '''if last_timestep:
        print('reducing timesteps')
        for param in unique_vals:

            last_epoch_df[param] = last_epoch_df[param].apply(get_last_timestep)'''
            #This is redundant with above extract part and also doesn't work properly.
    unique_vals = {param: last_epoch_df[param].unique() for param in params}
    print('new unique vals', unique_vals)

    if not skip_1x:
        print('calculating single params')

        for param in params:
            avg_loss[param] = df.groupby([param, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
            means = last_epoch_df.groupby([param]).mean(numeric_only=True)
            ranges_1[param] = means['accuracy'].max() - means['accuracy'].min()
        print('calculating double params')

        for param1, param2 in tqdm.tqdm(param_pairs):
            avg_loss[(param1, param2)] = df.groupby([param1, param2, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()

            means = last_epoch_df.groupby([param1, param2]).mean()
            ranges_2[(param1, param2)] = means['accuracy'].max() - means['accuracy'].min()

            if not skip_2x1:
                for value1 in unique_vals[param1]:
                    subset = last_epoch_df[last_epoch_df[param1] == value1]
                    if len(subset[param2].unique()) > 1:
                        new_means = subset.groupby(param2)['accuracy'].mean()
                        range_dict[(param1, value1, param2)] = new_means.max() - new_means.min()

    if not skip_3x:
        for param1, param2, param3 in param_triples:
            ci_df = df.groupby([param1, param2, param3, 'epoch'])['accuracy'].apply(calculate_ci).reset_index()
            avg_loss[(param1, param2, param3)] = ci_df

            for value1 in unique_vals[param1]:
                for value2 in unique_vals[param2]:
                    subset = last_epoch_df[(last_epoch_df[param2] == value2) & (last_epoch_df[param1] == param1)]
                    if len(subset[param3].unique()) > 1:
                        new_means = subset.groupby(param3)['accuracy'].mean()
                        range_dict3[(param1, value1, param2, value2, param3)] = new_means.max() - new_means.min()

    df_summary = {}
    delta_operator_summary = {}
    print('key param stats')

    key_param_stats = {}
    oracle_key_param_stats = {}
    if key_param is not None:
        for param in params:
            print(param)
            if param != key_param:
                # Initializing a nested dictionary for each unique key_param value
                for acc_type, save_dict in zip(['accuracy'], [key_param_stats]): #zip(['accuracy', 'o_acc'], [key_param_stats, oracle_key_param_stats]):
                    #print('SSSSSSSSSS', unique_vals) #unique vals of regime has only 6 of them
                    for key_val in unique_vals[key_param]:
                        subset = last_epoch_df[last_epoch_df[key_param] == key_val]

                        grouped = subset.groupby(['repetition', param])[acc_type]
                        repetition_means = grouped.mean()
                        overall_means = repetition_means.groupby(level=param).mean()
                        means_std = repetition_means.groupby(level=param).std()

                        Q1 = grouped.quantile(0.25).groupby(level=param).mean()
                        Q3 = grouped.quantile(0.75).groupby(level=param).mean()
                        counts = grouped.size()

                        z_value = 1.96  # For a 95% CI
                        standard_errors = (z_value * np.sqrt(repetition_means * (1 - repetition_means) / counts)).groupby(level=param).mean()

                        #print(key_val, param, 'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')

                        if key_val not in save_dict:
                            save_dict[key_val] = {}
                        save_dict[key_val][param] = {
                            'mean': overall_means.to_dict(),
                            'std': means_std.to_dict(),
                            'q1': Q1.to_dict(),
                            'q3': Q3.to_dict(),
                            'ci': standard_errors.to_dict(),
                        }
                        # dict order is key_val > param > mean/std > param_val

                        #print('test group', key_val)
                        #print("Unique test_regime values:", subset['test_regime'].unique())

                        for group_name in ['s1', 's2', 's21', 's3']:
                            subset2 = subset[subset['test_group'] == group_name]
                            print('ddddd', key_val, key_param, group_name, len(subset2))
                            if len(subset) > 0:
                                grouped = subset2.groupby(['repetition', param])[acc_type]
                                repetition_means = grouped.mean()

                                if key_val not in save_dict:
                                    save_dict[key_val] = {}
                                save_dict[key_val][group_name] = {
                                    'mean': repetition_means.groupby(level=param).mean().to_dict(),
                                    'std': repetition_means.groupby(level=param).std().to_dict(),
                                    'q1': grouped.quantile(0.25).groupby(level=param).mean().to_dict(),
                                    'q3': grouped.quantile(0.75).groupby(level=param).mean().to_dict(),
                                    'ci': (z_value * np.sqrt(repetition_means * (1 - repetition_means) / grouped.size())).groupby(level=param).mean().to_dict(),
                                }

        if record_delta_pi:
            delta_pi_stats(unique_vals, key_param, last_epoch_df, delta_operator_summary, df_summary)

    #print(key_param_stats['a-mix-r-perception-100-loc-s2'].keys())
    #exit()

    return avg_loss, variances, ranges_1, ranges_2, range_dict, range_dict3, stats, key_param_stats, oracle_key_param_stats, df_summary, delta_operator_summary


def delta_pi_stats(unique_vals, key_param, last_epoch_df, delta_operator_summary, df_summary):
    # todo: if delay_2nd_bait or first_swap are na, just make them 0? shouldn't affect anything
    print('calculating delta preds')

    set1 = ['T', 'F', 'N']
    set2 = ['t', 'f', 'n']
    set3 = ['0', '1']
    combinations = list(itertools.product(set1, set2, set3))
    combinations_str = [''.join(combo) for combo in combinations]

    operators = ['T-F', 't-f', 'F-N', 'f-n', 'T-N', 't-n', '1-0']
    required_columns = [f'pred_{i}' for i in range(5)]
    perm_keys = ['p-b-0', 'p-b-1', 'p-s-0', 'p-s-1', 'delay_2nd_bait', 'first_swap', 'first_bait_size', 'delay']

    for key_val in unique_vals[key_param]:  # for each train regime, etc
        unique_repetitions = last_epoch_df['repetition'].unique()

        delta_mean_rep = {key: [] for key in operators}
        delta_std_rep = {key: [] for key in operators}
        delta_mean_correct_rep = {key: [] for key in operators}
        delta_std_correct_rep = {key: [] for key in operators}
        delta_mean_accurate_rep = {key: [] for key in operators}
        delta_std_accurate_rep = {key: [] for key in operators}

        delta_mean_p_t_rep = {key: [] for key in operators}
        delta_mean_p_f_rep = {key: [] for key in operators}
        delta_mean_m_t_rep = {key: [] for key in operators}
        delta_mean_m_f_rep = {key: [] for key in operators}

        delta_mean_p_t_t_rep = {key: [] for key in operators}
        delta_mean_p_t_f_rep = {key: [] for key in operators}
        delta_mean_p_f_t_rep = {key: [] for key in operators}
        delta_mean_p_f_f_rep = {key: [] for key in operators}

        delta_mean_m_t_t_rep = {key: [] for key in operators}
        delta_mean_m_t_f_rep = {key: [] for key in operators}
        delta_mean_m_f_t_rep = {key: [] for key in operators}
        delta_mean_m_f_f_rep = {key: [] for key in operators}

        delta_mean = [{key: [] for key in operators} for _ in unique_repetitions]
        delta_mean_correct = [{key: [] for key in operators} for _ in unique_repetitions]
        delta_mean_accurate = [{key: [] for key in operators} for _ in unique_repetitions]

        dpred_mean_p_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_p_f = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_f = [{key: [] for key in operators} for _ in unique_repetitions]

        dpred_mean_p_t_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_p_t_f = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_p_f_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_p_f_f = [{key: [] for key in operators} for _ in unique_repetitions]

        dpred_mean_m_t_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_t_f = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_f_t = [{key: [] for key in operators} for _ in unique_repetitions]
        dpred_mean_m_f_f = [{key: [] for key in operators} for _ in unique_repetitions]

        for rep in unique_repetitions:
            subset_rep = last_epoch_df[last_epoch_df['repetition'] == rep]

            # print(subset_rep.columns, key_param, key_val)
            subset = subset_rep[subset_rep[key_param] == key_val]
            subset['pred'] = subset['pred'].apply(convert_to_numeric).astype(np.int8)
            conv = pd.concat([subset, pd.get_dummies(subset['pred'], prefix='pred')], axis=1)

            for col in required_columns:  # a model might not predict all 5 values
                if col not in conv.columns:
                    conv[col] = 0
            subset = conv[required_columns + ['i-informedness', 'correct-loc', 'opponents', 'accuracy'] + perm_keys]

            # OPERATOR PART
            for op in operators:

                delta_preds = []
                delta_preds_correct = []
                delta_preds_accurate = []

                dpred_p_t = []
                dpred_p_f = []
                dpred_m_t = []
                dpred_m_f = []

                dpred_p_t_t = []
                dpred_p_t_f = []
                dpred_p_f_t = []
                dpred_p_f_f = []

                dpred_m_t_t = []
                dpred_m_t_f = []
                dpred_m_f_t = []
                dpred_m_f_f = []

                for key in combinations_str:
                    # key is 1st position, descendents are 2nd
                    descendants = get_descendants(key, op, combinations_str)

                    mapping = {'T': 2, 'F': 1, 'N': 0, 't': 2, 'f': 1, 'n': 0, '0': 0, '1': 1}

                    key = [mapping[char] for char in key]
                    key_informedness = '[' + ' '.join(map(str, key[:2])) + ']'
                    key_opponents = np.float64(key[2])
                    if '0' not in op and key_opponents == 0:
                        # for most operators, we only use cases with opponents present
                        continue

                    descendants = [[mapping[char] for char in descendant] for descendant in descendants]
                    descendant_informedness = ['[' + ' '.join(map(str, descendant[:2])) + ']' for descendant in
                                               descendants]
                    descendant_opponents = [np.float64(descendant[2]) for descendant in descendants]

                    if len(descendants) < 1:
                        continue

                    inf = subset[
                        (subset['i-informedness'] == key_informedness) & (subset['opponents'] == key_opponents)].groupby(
                        perm_keys + ['i-informedness', 'opponents', 'correct-loc'],
                        observed=True).mean().reset_index()
                    noinf = subset[(subset['i-informedness'].isin(descendant_informedness)) &
                                   (subset['opponents'].isin(descendant_opponents))].groupby(
                        perm_keys + ['i-informedness', 'opponents', 'correct-loc'],
                        observed=True).mean().reset_index()
                    # print('lens1', len(inf), len(noinf), len(subset))

                    merged_df = pd.merge(
                        inf,
                        noinf,
                        on=perm_keys,
                        suffixes=('_m', ''),
                        how='inner',
                    )

                    merged_df['changed_target'] = (
                            merged_df['correct-loc_m'] != merged_df['correct-loc']).astype(int)

                    for i in range(5):
                        merged_df[f'pred_diff_{i}'] = abs(merged_df[f'pred_{i}_m'] - merged_df[f'pred_{i}'])
                    merged_df['total_pred_diff'] = merged_df[[f'pred_diff_{idx}' for idx in range(5)]].sum(axis=1) / 2
                    delta_preds.extend(merged_df['total_pred_diff'].tolist())
                    merged_df['total_pred_diff_correct'] = 1 - abs(
                        merged_df['total_pred_diff'] - merged_df['changed_target'])
                    delta_preds_correct.extend(merged_df['total_pred_diff_correct'].tolist())
                    merged_df['total_pred_diff_accurate'] = merged_df['total_pred_diff_correct'] * merged_df[
                        'accuracy'] * merged_df['accuracy_m']
                    delta_preds_accurate.extend(merged_df['total_pred_diff_accurate'].tolist())

                    merged_df['changed_target'] = merged_df['changed_target'].astype(bool)

                    merged_df['dpred_p_t'] = (merged_df['changed_target'] == 1) * (merged_df['total_pred_diff'])
                    merged_df['dpred_p_f'] = (merged_df['changed_target'] == 1) * (1 - merged_df['total_pred_diff'])
                    merged_df['dpred_m_t'] = (merged_df['changed_target'] == 0) * (merged_df['total_pred_diff'])
                    merged_df['dpred_m_f'] = (merged_df['changed_target'] == 0) * (1 - merged_df['total_pred_diff'])

                    merged_df['total_pred_diff_p_T_T'] = (merged_df['changed_target']) * (merged_df['accuracy_m']) * (
                        merged_df['accuracy'])
                    merged_df['total_pred_diff_p_T_F'] = (merged_df['changed_target']) * (merged_df['accuracy_m']) * (
                            1 - merged_df['accuracy'])
                    merged_df['total_pred_diff_p_F_T'] = (merged_df['changed_target']) * (
                            1 - merged_df['accuracy_m']) * (merged_df['accuracy'])
                    merged_df['total_pred_diff_p_F_F'] = (merged_df['changed_target']) * (
                            1 - merged_df['accuracy_m']) * (1 - merged_df['accuracy'])

                    merged_df['total_pred_diff_m_T_T'] = (1 - merged_df['changed_target']) * (
                        merged_df['accuracy_m']) * (merged_df['accuracy'])
                    merged_df['total_pred_diff_m_T_F'] = (1 - merged_df['changed_target']) * (
                        merged_df['accuracy_m']) * (1 - merged_df['accuracy'])
                    merged_df['total_pred_diff_m_F_T'] = (1 - merged_df['changed_target']) * (
                            1 - merged_df['accuracy_m']) * (merged_df['accuracy'])
                    merged_df['total_pred_diff_m_F_F'] = (1 - merged_df['changed_target']) * (
                            1 - merged_df['accuracy_m']) * (1 - merged_df['accuracy'])

                    # print(op, key, descendants, np.mean(merged_df['changed_target']))
                    dpred_p_t.extend(merged_df['dpred_p_t'].tolist())
                    dpred_p_f.extend(merged_df['dpred_p_f'].tolist())
                    dpred_m_t.extend(merged_df['dpred_m_t'].tolist())
                    dpred_m_f.extend(merged_df['dpred_m_f'].tolist())

                    dpred_p_t_t.extend(merged_df['total_pred_diff_p_T_T'].tolist())
                    dpred_p_t_f.extend(merged_df['total_pred_diff_p_T_F'].tolist())
                    dpred_p_f_t.extend(merged_df['total_pred_diff_p_F_T'].tolist())
                    dpred_p_f_f.extend(merged_df['total_pred_diff_p_F_F'].tolist())
                    dpred_m_t_t.extend(merged_df['total_pred_diff_m_T_T'].tolist())
                    dpred_m_t_f.extend(merged_df['total_pred_diff_m_T_F'].tolist())
                    dpred_m_f_t.extend(merged_df['total_pred_diff_m_F_T'].tolist())
                    dpred_m_f_f.extend(merged_df['total_pred_diff_m_F_F'].tolist())

                # first level aggregate: for each operator, within one repetition
                r = int(rep)
                delta_mean[r][op] = np.mean(delta_preds)
                delta_mean_correct[r][op] = np.mean(delta_preds_correct)
                delta_mean_accurate[r][op] = np.mean(delta_preds_accurate)

                dpred_mean_p_t[r][op] = np.mean(dpred_p_t)
                dpred_mean_p_f[r][op] = np.mean(dpred_p_f)
                dpred_mean_m_t[r][op] = np.mean(dpred_m_t)
                dpred_mean_m_f[r][op] = np.mean(dpred_m_f)

                dpred_mean_p_t_t[r][op] = np.mean(dpred_p_t_t)
                dpred_mean_p_t_f[r][op] = np.mean(dpred_p_t_f)
                dpred_mean_p_f_t[r][op] = np.mean(dpred_p_f_t)
                dpred_mean_p_f_f[r][op] = np.mean(dpred_p_f_f)

                dpred_mean_m_t_t[r][op] = np.mean(dpred_m_t_t)
                dpred_mean_m_t_f[r][op] = np.mean(dpred_m_t_f)
                dpred_mean_m_f_t[r][op] = np.mean(dpred_m_f_t)
                dpred_mean_m_f_f[r][op] = np.mean(dpred_m_f_f)

        # second level aggregate: over one repetition

        for op in operators:
            op_values_mean = [rep_dict[op] for rep_dict in delta_mean]
            op_values_mean_correct = [rep_dict[op] for rep_dict in delta_mean_correct]
            op_values_mean_accurate = [rep_dict[op] for rep_dict in delta_mean_accurate]

            op_values_mean_pt = [rep_dict[op] for rep_dict in dpred_mean_p_t]
            op_values_mean_pf = [rep_dict[op] for rep_dict in dpred_mean_p_f]
            op_values_mean_mt = [rep_dict[op] for rep_dict in dpred_mean_m_t]
            op_values_mean_mf = [rep_dict[op] for rep_dict in dpred_mean_m_f]

            op_values_mean_ptt = [rep_dict[op] for rep_dict in dpred_mean_p_t_t]
            op_values_mean_ptf = [rep_dict[op] for rep_dict in dpred_mean_p_t_f]
            op_values_mean_pft = [rep_dict[op] for rep_dict in dpred_mean_p_f_t]
            op_values_mean_pff = [rep_dict[op] for rep_dict in dpred_mean_p_f_f]
            op_values_mean_mtt = [rep_dict[op] for rep_dict in dpred_mean_m_t_t]
            op_values_mean_mtf = [rep_dict[op] for rep_dict in dpred_mean_m_t_f]
            op_values_mean_mft = [rep_dict[op] for rep_dict in dpred_mean_m_f_t]
            op_values_mean_mff = [rep_dict[op] for rep_dict in dpred_mean_m_f_f]

            delta_mean_rep[op] = np.mean(op_values_mean)
            delta_std_rep[op] = np.std(op_values_mean)

            delta_mean_correct_rep[op] = np.mean(op_values_mean_correct)
            delta_std_correct_rep[op] = np.std(op_values_mean_correct)

            delta_mean_accurate_rep[op] = np.mean(op_values_mean_accurate)
            delta_std_accurate_rep[op] = np.std(op_values_mean_accurate)

            delta_mean_p_t_rep[op] = np.mean(op_values_mean_pt)
            delta_mean_p_f_rep[op] = np.mean(op_values_mean_pf)
            delta_mean_m_t_rep[op] = np.mean(op_values_mean_mt)
            delta_mean_m_f_rep[op] = np.mean(op_values_mean_mf)

            delta_mean_p_t_t_rep[op] = np.mean(op_values_mean_ptt)
            delta_mean_p_t_f_rep[op] = np.mean(op_values_mean_ptf)
            delta_mean_p_f_t_rep[op] = np.mean(op_values_mean_pft)
            delta_mean_p_f_f_rep[op] = np.mean(op_values_mean_pff)
            delta_mean_m_t_t_rep[op] = np.mean(op_values_mean_mtt)
            delta_mean_m_t_f_rep[op] = np.mean(op_values_mean_mtf)
            delta_mean_m_f_t_rep[op] = np.mean(op_values_mean_mft)
            delta_mean_m_f_f_rep[op] = np.mean(op_values_mean_mff)

        # third level aggregate: all key_vals
        delta_operator_summary[key_val] = pd.DataFrame({
            'operator': list(delta_mean_rep.keys()),
            'dpred': [f"{delta_mean_rep[key]:.2f} ({delta_std_rep[key]:.2f})" for key in delta_mean_rep.keys()],
            'dpred_correct': [f"{delta_mean_correct_rep[key]:.2f} ({delta_std_correct_rep[key]:.2f})" for key in
                              delta_mean_correct_rep.keys()],
            'dpred_accurate': [f"{delta_mean_accurate_rep[key]:.2f} ({delta_std_accurate_rep[key]:.2f})" for key in
                               delta_mean_accurate_rep.keys()],
            'ptt': [delta_mean_p_t_t_rep[key] for key in delta_mean_p_t_t_rep.keys()],
            'ptf': [delta_mean_p_t_f_rep[key] for key in delta_mean_p_t_f_rep.keys()],
            'pft': [delta_mean_p_f_t_rep[key] for key in delta_mean_p_f_t_rep.keys()],
            'pff': [delta_mean_p_f_f_rep[key] for key in delta_mean_p_f_f_rep.keys()],
            'mtt': [delta_mean_m_t_t_rep[key] for key in delta_mean_m_t_t_rep.keys()],
            'mtf': [delta_mean_m_t_f_rep[key] for key in delta_mean_m_t_f_rep.keys()],
            'mft': [delta_mean_m_f_t_rep[key] for key in delta_mean_m_f_t_rep.keys()],
            'mff': [delta_mean_m_f_f_rep[key] for key in delta_mean_m_f_f_rep.keys()],
            'pt': [delta_mean_p_t_rep[key] for key in delta_mean_p_t_rep.keys()],
            'pf': [delta_mean_p_f_rep[key] for key in delta_mean_p_f_rep.keys()],
            'mt': [delta_mean_m_t_rep[key] for key in delta_mean_m_t_rep.keys()],
            'mf': [delta_mean_m_f_rep[key] for key in delta_mean_m_f_rep.keys()],
        })

        # print('do', key_val, delta_operator_summary[key_val])

        # CATEGORY ONE

        # for col in perm_keys + ['informedness', 'correct-loc']:
        #    print(f"{col} has {subset[col].nunique()} unique values:", subset[col].unique())

        inf = subset[subset['i-informedness'] == 'Tt'].groupby(perm_keys + ['i-informedness', 'correct-loc'],
                                                               observed=True).mean().reset_index()
        noinf = subset[subset['i-informedness'] != 'Tt'].groupby(perm_keys + ['i-informedness', 'correct-loc'],
                                                                 observed=True).mean().reset_index()

        merged_df = pd.merge(
            inf,
            noinf,
            on=perm_keys,
            suffixes=('_m', ''),
            how='inner',
        )

        merged_df['changed_target'] = (merged_df['correct-loc_m'] != merged_df['correct-loc']).astype(int)

        for i in range(5):
            merged_df[f'pred_diff_{i}'] = abs(merged_df[f'pred_{i}_m'] - merged_df[f'pred_{i}'])
        merged_df['total_pred_diff'] = merged_df[[f'pred_diff_{idx}' for idx in range(5)]].sum(axis=1) / 2
        merged_df['total_pred_diff_correct'] = 1 - abs(merged_df['total_pred_diff'] - merged_df['changed_target'])

        delta_preds = {}
        delta_preds_correct = {}

        for key in merged_df['i-informedness'].unique():
            delta_preds[key] = merged_df.loc[merged_df['i-informedness'] == key, 'total_pred_diff'].tolist()
            delta_preds_correct[key] = merged_df.loc[
                merged_df['i-informedness'] == key, 'total_pred_diff_correct'].tolist()

        delta_mean = {key: np.mean(val) for key, val in delta_preds.items()}
        delta_std = {key: np.std(val) for key, val in delta_preds.items()}

        delta_mean_correct = {key: np.mean(val) for key, val in delta_preds_correct.items()}
        delta_std_correct = {key: np.std(val) for key, val in delta_preds_correct.items()}

        delta_mean_accurate = {key: np.mean(val) for key, val in delta_preds_correct.items()}
        delta_std_accurate = {key: np.std(val) for key, val in delta_preds_correct.items()}

        df_summary[key_val] = pd.DataFrame({
            'Informedness': list(delta_mean.keys()),
            'dpred': [f"{delta_mean[key]} ({delta_std[key]})" for key in delta_mean.keys()],
            'dpred_correct': [f"{delta_mean_correct[key]} ({delta_std_correct[key]})" for key in
                              delta_mean_correct.keys()]
        })

        # print(df_summary[key_val])