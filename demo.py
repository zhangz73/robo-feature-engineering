import numpy as np
import pandas as pd
from feature_engineer import feat_eng_single_tuning, feat_eng_single_df
from pprint import pprint

df = pd.read_csv("Data/toy_data.csv")
df_engineered = df[["Y"]].copy()

for feat in ["C1", "C2", "X1", "X2"]:
    is_cat = feat.startswith("C")
    x_new, res, ret_pct = feat_eng_single_tuning(np.array(df[feat]), np.array(df["Y"]), w = None, is_cat = is_cat, min_pop = 5000, max_split = np.inf, split_pop_ratio = 5, merge_pct_factor = 1.2, max_retry = 5, step_size = 0.05, check_mono = True)
    df_engineered[feat] = x_new.copy()
    print(f"{feat} at the merge factor of {round(ret_pct, 2)}:")
    print(f"\tLevels: " + str(res[1]))
    print(f"\tDetails: " + str(['(n = ' + str(int(x[0])) + ', r = ' + str(round(x[1] * 100, 2)) + '%)' for x in res[2]]))
    print("")

df_engineered, info_dict = feat_eng_single_df(df, "Y", w_name = None, cat_cols = ["C1", "C2"], num_cols = ["X1", "X2"], min_pop = 5000, max_split = np.inf, split_pop_ratio = 5, merge_pct_factor = 1.2, max_retry = 5, step_size = 0.05, check_mono = True, verbose = False)
print(df_engineered)
pprint(info_dict)

#df_engineered.to_csv("Data/toy_data_eng.csv", index=False)
