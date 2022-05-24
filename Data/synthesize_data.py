import numpy as np
import pandas as pd

np.random.seed(0) #1234

def synthesize_y(y_dist = [], n_data = []):
    assert len(y_dist) == len(n_data)
    block_lst = []
    y_lst = []
    for i in range(len(y_dist)):
        block_lst += ["Block" + str(i + 1) for x in range(n_data[i])]
        y_lst += list(np.random.binomial(1, y_dist[i], size = n_data[i]))
    return y_lst

def synthesize_cat(n_data = [[400, 200, 100, 50, 50], [300, 200, 200], [300, 200]]):
    data = []
    for i in range(len(n_data)):
        ret = []
        for j in range(len(n_data[i])):
            curr = [f"Block_{i + 1}_{j + 1}" for x in range(n_data[i][j])]
            ret += curr
        data += ret
    return data

def synthesize_num(ranges = [(0, 1), (1, 4), (4, 10)], n_data = [1000, 500, 500]):
    assert len(ranges) == len(n_data)
    data = []
    for i in range(len(n_data)):
        curr_range = ranges[i]
        ret = list(np.random.uniform(low = curr_range[0], high = curr_range[1], size = n_data[i]))
        data += ret
    return data

def synthesize_whole(feature_dict = {"C1": {"n_data": [[400, 200, 100, 50, 50], [300, 200, 200], [300, 200]], "y_dist": [0.8, 0.5, 0.2]}, "X1": {"ranges": [(0, 1), (1, 4), (4, 10)], "n_data": [1000, 500, 500], "y_dist": [0.7, 0.3, 0.1]}}):
    sizes = [np.sum(np.sum(v["n_data"])) for k,v in feature_dict.items()]
    assert len(sizes) == 0 or np.all(sizes == sizes[0])
    for feat in feature_dict:
        assert len(feature_dict[feat]["n_data"]) == len(feature_dict[feat]["y_dist"])
    
    y_dist = 1
    ret_dct = {}
    for feat in feature_dict:
        if feat.startswith("C"):
            data = synthesize_cat(feature_dict[feat]["n_data"])
        else:
            data = synthesize_num(feature_dict[feat]["ranges"], feature_dict[feat]["n_data"])
        idx = np.random.choice(len(data), size = len(data), replace = False)
        ret_dct[feat] = [data[x] for x in idx]
        curr_y_dist = []
        for i in range(len(feature_dict[feat]["y_dist"])):
            pop = np.sum(feature_dict[feat]["n_data"][i])
            curr_y_dist += [feature_dict[feat]["y_dist"][i] for x in range(pop)]
        curr_y_dist = [curr_y_dist[x] for x in idx]
        y_dist *= np.array(curr_y_dist)
    y = synthesize_y(list(y_dist), n_data = [1] * len(y_dist))
    ret_dct["Y"] = y
    df_ret = pd.DataFrame.from_dict(ret_dct)
    return df_ret.sample(frac = 1)

#feature_dict = {
#    "C1": {"n_data": [[4000, 2000, 2000], [3000, 4000], [3000, 2000]], "y_dist": [0.99, 0.5, 0.05]},
#    "C2": {"n_data": [[6000, 4000], [3000, 7000]], "y_dist": [0.3, 0.9]},
#    "X1": {"ranges": [(0, 4), (4, 30)], "n_data": [15000, 5000], "y_dist": [0.9, 0.3]},
#    "X2": {"ranges": [(-5, 0), (0, 100), (100, 1000)], "n_data": [10000, 5000, 5000], "y_dist": [0.1, 0.5, 0.8]}
#}
feature_dict = {
    "C1": {"n_data": [[2000, 1000, 1000], [1500, 2000], [1500, 1000]], "y_dist": [0.99, 0.5, 0.05]},
    "C2": {"n_data": [[3000, 2000], [1500, 3500]], "y_dist": [0.3, 0.9]},
    "X1": {"ranges": [(0, 4), (4, 30)], "n_data": [7500, 2500], "y_dist": [0.9, 0.3]},
    "X2": {"ranges": [(-5, 0), (0, 100), (100, 1000)], "n_data": [5000, 2500, 2500], "y_dist": [0.1, 0.5, 0.8]}
}
df = synthesize_whole(feature_dict)
df.to_csv("test_data.csv", index=False)
