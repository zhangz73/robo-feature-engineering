import numpy as np
import pandas as pd
from feature_engineer import feat_eng_single_tuning

df = pd.read_csv("toy_data.csv")

for feat in ["C1", "C2", "X1", "X2"]:
    is_cat = feat.startswith("C")
    res, ret_pct = feat_eng_single_tuning(np.array(df[feat]), np.array(df["Y"]), w = None, is_cat = is_cat, min_pop = 5000, max_split = np.inf, split_pop_ratio = 5, merge_pct_factor = 1.2, max_retry = 5, step_size = 0.05, check_mono = True)
    print(f"{feat} at {round(ret_pct, 2)}:", res[1], res[2])
