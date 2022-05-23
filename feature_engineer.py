###
# This script aims to denoise the data by clustering features into bins, so that the signals for the target variable will be stronger and more coherent within each feature.
#
# At the current version, it only handles independent univariate features in binary classification problems in a greedy way. More to come in future versions.
###

import numpy as np
import pandas as pd

def get_ratio(x1, x2):
    assert x1 >= 0 and x2 >= 0
    if min(x1, x2) > 0:
        return max(x1, x2) / min(x1, x2)
    else:
        return np.inf

def get_info(elem, data, is_cat = False):
    if not is_cat:
        lo, hi = data["X"].quantile(elem[0]), data["X"].quantile(elem[1])
        if hi < 1:
            curr = data[(data["X"] >= lo) & (data["X"] < hi)]
        else:
            curr = data[(data["X"] >= lo) & (data["X"] <= hi)]
    else:
        curr = data[data["X"].isin(elem)]
    if curr.shape[0] > 0:
        pop = curr["acu_pop"].iloc[-1] - curr["acu_pop"].iloc[0] + curr["W"].iloc[0]
        y_pct = curr["acu_Y"].iloc[-1] - curr["acu_Y"].iloc[0] + curr["Y"].iloc[0]
        y_pct = y_pct / pop
    else:
        pop, y_pct = 0, 0
    return (pop, y_pct)

def is_monotonic(arr):
    if len(arr) <= 1:
        return True
    arr = np.array(arr)
    diff = arr[1:] - arr[:-1]
    return sum(diff >= 0) == len(arr) or sum(diff <= 0) == len(arr)

def get_longest_mono_seq_oneway(arr, increasing = True):
    dp = np.ones(len(arr))
    dp_pt = np.zeros(len(arr)) - 1
    for i in range(1, len(arr)):
        for j in range(i):
            if (increasing and arr[i] >= arr[j]) or (not increasing and arr[i] <= arr[j]):
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    dp_pt[i] = j
    return dp, dp_pt

def get_longest_mono_seq(arr):
    dp_up, dp_pt_up = get_longest_mono_seq_oneway(arr, increasing = True)
    dp_down, dp_pt_down = get_longest_mono_seq_oneway(arr, increasing = False)
    if max(dp_up) > max(dp_down):
        dp_opt, dp_pt_opt = dp_up, dp_pt_up
    else: # max(dp_up) < max(dp_down):
        dp_opt, dp_pt_opt = dp_down, dp_pt_down
    dp_ret = np.zeros(len(arr))
    idx = np.argmax(dp_opt)
    while idx > -1:
        dp_ret[idx] = 1
        idx = int(dp_pt_opt[idx])
    return dp_ret

def setup(x, y, w, is_cat = False):
    assert len(x) == len(y)
    if w is None:
        w = np.ones(len(x))
    data = pd.DataFrame.from_dict({"W": w, "X": x, "Y": y})
    if is_cat:
        data = data.groupby("X").sum().reset_index()
        data = data.sort_values("Y")
    else:
        data = data.sort_values("X")
    data["acu_Y"] = data["Y"].cumsum()
    data["acu_pop"] = data["W"].cumsum()
    num_1 = data["Y"].sum()
    n_row = data.shape[0]
    num_0 = n_row - num_1
    return data

def check_merges(data, curr_lst, curr_info_lst, is_cat = False, min_pop = 10000, merge_pct_factor = 1.2, check_mono = True):
    # Check for Merges
    has_movement = True
    mono_merge = []
    while has_movement:
        has_movement = False
        i = 0
        while i < len(curr_lst):
            elem, info = curr_lst[i], curr_info_lst[i]
            neighbors = []
            if i - 1 >= 0:
                neighbors.append((i - 1, curr_lst[i - 1], curr_info_lst[i - 1]))
            if i + 1 < len(curr_lst):
                neighbors.append((i + 1, curr_lst[i + 1], curr_info_lst[i + 1]))
            neighbor_pct_ratios = [get_ratio(info[1], x[2][1]) for x in neighbors]
            if len(neighbor_pct_ratios) > 0:
                min_neighbor_pct_ratios = np.min(neighbor_pct_ratios)
                min_neighbor_pct_ratios_idx = neighbors[np.argmin(neighbor_pct_ratios)][0]
            else:
                min_neighbor_pct_ratios = 1
                min_neighbor_pct_ratios_idx = -1
            if info[0] < min_pop or min_neighbor_pct_ratios < merge_pct_factor or (i < len(mono_merge) and mono_merge[i]):
                if min_neighbor_pct_ratios_idx > -1:
                    if min_neighbor_pct_ratios_idx == i - 1:
                        elem_prev, info_prev = curr_lst[i - 1], curr_info_lst[i - 1]
                        if not is_cat:
                            tup = (elem_prev[0], elem[1])
                        else:
                            tup = elem_prev + elem
                        info_tup = get_info(tup, data, is_cat = is_cat)
                        curr_lst = curr_lst[:(i - 1)] + [tup] + curr_lst[(i + 1):]
                        curr_info_lst = curr_info_lst[:(i - 1)] + [info_tup] + curr_info_lst[(i + 1):]
                        i -= 1
                    else:
                        elem_next, info_next = curr_lst[i + 1], curr_info_lst[i + 1]
                        if not is_cat:
                            tup = (elem[0], elem_next[1])
                        else:
                            tup = elem + elem_next
                        info_tup = get_info(tup, data, is_cat = is_cat)
                        curr_lst = curr_lst[:i] + [tup] + curr_lst[(i + 2):]
                        curr_info_lst = curr_info_lst[:i] + [info_tup] + curr_info_lst[(i + 2):]
                    if i < len(mono_merge) and mono_merge[i]:
                        mono_merge = []
                    has_movement = True
            i += 1
        if not has_movement and check_mono:
            arr = [x[1] for x in curr_info_lst]
            mono_seq = get_longest_mono_seq(arr)
            if sum(mono_seq) < len(arr):
                has_movement = True
                mono_merge = [x == 0 for x in mono_seq]
    return curr_lst, curr_info_lst

def num_feat_eng_single(x, y, w = None, min_pop = 10000, max_split = np.inf, split_pop_ratio = 10, merge_pct_factor = 1.2, check_mono = True):
    data = setup(x, y, w)
    
    lst = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    # Gather Info
    info_lst = []
    for elem in lst:
        pop, y_pct = get_info(elem, data)
        info_lst.append((pop, y_pct))

    # Check for Splits
    has_movement = True
    while has_movement:
        has_movement = False
        curr_lst = []
        curr_info_lst = []
        min_split_pop = min_pop #max(np.min([x[0] for x in info_lst]), min_pop) #
        for elem, info in zip(lst, info_lst):
            lo, hi = elem
            if info[0] / min_split_pop > split_pop_ratio:
                mid = lo + (hi - lo) / 2
                tup_lst = [(lo, mid), (mid, hi)]
                tup_info_lst = [get_info((lo, mid), data), get_info((mid, hi), data)]
                if tup_info_lst[0][0] * tup_info_lst[1][0] > 0 and info[0] != tup_info_lst[0][0]:
                    curr_lst += tup_lst
                    curr_info_lst += tup_info_lst
                    has_movement = True
                else:
                    curr_lst.append(elem)
                    curr_info_lst.append(info)
            else:
                curr_lst.append(elem)
                curr_info_lst.append(info)
        lst, info_lst = curr_lst, curr_info_lst
        
    # Check for Merges
    curr_lst, curr_info_lst = check_merges(data, curr_lst, curr_info_lst, is_cat = False, min_pop = min_pop, merge_pct_factor = merge_pct_factor, check_mono = check_mono)
        
    # Update
    lst, info_lst = curr_lst, curr_info_lst
    range_lst = [(data["X"].quantile(x[0]), data["X"].quantile(x[1])) for x in lst]
        
    return lst, range_lst, info_lst

def cat_feat_eng_single(x, y, w = None, min_pop = 10000, max_split = np.inf, 
                        split_pop_ratio = 10, merge_pct_factor = 1.2, check_mono = True):
    data = setup(x, y, w, is_cat = True)
    lst = [[x] for x in list(data["X"])]
    
    # Gather Info
    info_lst = []
    for elem in lst:
        pop, y_pct = get_info(elem, data, is_cat = True)
        info_lst.append((pop, y_pct))
    
    curr_lst, curr_info_lst = check_merges(data, lst, info_lst, is_cat = True, min_pop = min_pop, merge_pct_factor = merge_pct_factor, check_mono = check_mono)
        
    # Update
    lst, info_lst = curr_lst, curr_info_lst
    range_lst = lst
        
    return lst, range_lst, info_lst

###
# This function conducts feature engineering on arbitrary univariate features that can be either numerical or categorical. It tunes the hyperparameter `merge_pct_factor` by tightening the criteria of merge until we get an engineered feature with more than one bins, or until we hit the max retry limit.
# Inputs:
#   x: The univariate feature to be engineered.
#   y: The target variable that is either 0 or 1.
#   w: The weight feature representing the weight/population for each data point. Default = None.
#   is_cat: The feature `x` will be interpreted as a categorical variable if set to True, numerical if set to False. Default = False.
#   min_pop: The minimum population in each bin. Default = 10000.
#   max_split: The maximum number of bins we can generate. Currently not being used. Default = Inf.
#   split_pop_ratio: If the population of a bin is larger than the population of the smallest bin by more than `split_pop_ratio` times, then split the larger bin into smaller ones. Only used when `x` is numerical. Default = 10.
#   merge_pct_factor: If the fractions of `y` being 1 of adjacent bins are not differred by at least `merge_pct_factor`, then they will be merged with the neighbors with closest distribution of `y`. Default = 1.2.
#   max_retry: The maximum number of attempts to tune the hyperparameter. Default = 5.
#   step_size: The step size to decrease the `merge_pct_factor` until it is below 1. Default = 0.05.
#   check_mono: Ensures the proportions of `y` being 1 in the returned bins are monotonic if set to True, False otherwise. Default = True.
# Outputs:
#   x_new: The reconstructed feature after binning `x`.
#   ret: If `x` is numerical, then `ret` is the list of ranges of numbers for each bin. If `x` is categorical, then `ret` is the list of values of `x` for each bin.
#   ret_pct: The list of (population, proportion of `y` being 1) for each bin.
###
def feat_eng_single_tuning(x, y, w = None, is_cat = False, min_pop = 10000, max_split = np.inf, split_pop_ratio = 10, merge_pct_factor = 1.2, max_retry = 5, step_size = 0.05, check_mono = True):
    assert step_size > 0
    
    pct_factor = merge_pct_factor
    proceed = True
    retry = 0
    ret = None
    ret_pct = pct_factor
    attempted_pct = []
    while proceed:
        attempted_pct.append(pct_factor)
        if not is_cat:
            res = num_feat_eng_single(x, y, w = w, min_pop = min_pop, max_split = max_split, split_pop_ratio = split_pop_ratio, merge_pct_factor = pct_factor, check_mono = check_mono)
        else:
            res = cat_feat_eng_single(x, y, w = w, min_pop = min_pop, max_split = max_split, split_pop_ratio = split_pop_ratio, merge_pct_factor = pct_factor, check_mono = check_mono)
        if retry == 0:
            ret = res
            ret_pct = pct_factor
        arr = [x[1] for x in res[2]]
        if len(arr) <= 1:
            pct_factor -= step_size
        else:
            ret = res
            ret_pct = pct_factor
            proceed = False
        retry += 1
        if retry >= max_retry or pct_factor in attempted_pct or pct_factor <= 1:
            proceed = False
    df = pd.DataFrame.from_dict({"X": x, "Y": y})
    df["X_new"] = None
    for i in range(len(ret[1])):
        if is_cat:
            df.loc[df["X"].isin(ret[1][i]), "X_new"] = f"Block_{i}"
        else:
            if i < len(ret[1]) - 1:
                df.loc[(df["X"] >= ret[1][i][0]) & (df["X"] < ret[1][i][1]), "X_new"] = f"Block_{i}"
            else:
                df.loc[df["X"] >= ret[1][i][0], "X_new"] = f"Block_{i}"
    x_new = np.array(df["X_new"])
    return x_new, ret, ret_pct

###
# This function conducts feature engineering on the entire dataframe.
# Inputs:
#   x: The univariate feature to be engineered.
#   y: The target variable that is either 0 or 1.
#   w: The weight feature representing the weight/population for each data point. Default = None.
#   is_cat: The feature `x` will be interpreted as a categorical variable if set to True, numerical if set to False. Default = False.
#   min_pop: The minimum population in each bin. Default = 10000.
#   max_split: The maximum number of bins we can generate. Currently not being used. Default = Inf.
#   split_pop_ratio: If the population of a bin is larger than the population of the smallest bin by more than `split_pop_ratio` times, then split the larger bin into smaller ones. Only used when `x` is numerical. Default = 10.
#   merge_pct_factor: If the fractions of `y` being 1 of adjacent bins are not differred by at least `merge_pct_factor`, then they will be merged with the neighbors with closest distribution of `y`. Default = 1.2.
#   max_retry: The maximum number of attempts to tune the hyperparameter. Default = 5.
#   step_size: The step size to decrease the `merge_pct_factor` until it is below 1. Default = 0.05.
#   check_mono: Ensures the proportions of `y` being 1 in the returned bins are monotonic if set to True, False otherwise. Default = True.
# Outputs:
#   df_engineered: The reconstructed dataframe after feature engineering.
###
def feat_eng_single_df(df, y_name, w_name = None, cat_cols = [], num_cols = [], min_pop = 10000, max_split = np.inf, split_pop_ratio = 10, merge_pct_factor = 1.2, max_retry = 5, step_size = 0.05, check_mono = True, verbose = False):
    df_engineered = df[[y_name]].copy()
    if w_name is None:
        w = None
    else:
        w = np.array(df[w_name])
        df_engineered[w_name] = w
    y = np.array(df[y_name])
    info_dict = {}
    for feat in df.columns:
        if feat not in [y_name, w_name]:
            if feat in cat_cols:
                is_cat = True
            elif feat in num_cols:
                is_cat = False
            else:
                is_cat = None
            if is_cat is not None:
                x_new, res, ret_pct = feat_eng_single_tuning(np.array(df[feat]), np.array(df["Y"]), w = w, is_cat = is_cat, min_pop = min_pop, max_split = max_split, split_pop_ratio = split_pop_ratio, merge_pct_factor = merge_pct_factor, max_retry = max_retry, step_size = step_size, check_mono = check_mono)
                info_dict[feat] = {"keep_original": False, "levels": res[1], "details": res[2], "merge_factor": ret_pct, "engineered_values": [f"Block_{i}" for i in range(len(res[1]))]}
                if verbose:
                    print(f"{feat} at the merge factor of {round(ret_pct, 2)}:")
                    print(f"\tLevels: " + str(res[1]))
                    print(f"\tDetails: " + str(['(n = ' + str(int(x[0])) + ', r = ' + str(round(x[1] * 100, 2)) + '%)' for x in res[2]]))
                    print("")
            else:
                x_new = np.array(df[feat])
                info_dict[feat] = {"keep_original": True}
            df_engineered[feat] = x_new.copy()
    return df_engineered, info_dict
