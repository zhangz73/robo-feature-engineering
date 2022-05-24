# robo-feature-engineering

## Background
Real-world datasets are more often noisy. Even for features with strong predictive power, it is unsurprisingly common to find segments that demonstrate contradicting trends. Simple models like logistic regressions might still be doing okay, as they enforce the monotonic assumption on the features. However, for the more sophisticated models such as random forest and gradient boosting, those contradictory segments can introduce high risk of overfitting. Therefore, denoising the data has become crucial before feeding it into machine learning models.

Data scientists usually have to spend a large amount of time exploring the dataset, engineering features based on ad-hoc analysis in combination of business intelligence accumulated throughout the years. Aiming to automate this effort, we implemented a robo feature engineering pipeline that denoise the features by binning the data.

## Example Usage
A simple example for engineering the features in the synthetic dataset can be found below. This example illustrates how each individual feature is engineered and how each reconstructed feature looks like:
```python
import numpy as np
import pandas as pd
from feature_engineer import feat_eng_single_tuning, feat_eng_single_df
from pprint import pprint

df = pd.read_csv("toy_data.csv")
df_engineered = df[["Y"]].copy()

for feat in ["C1", "C2", "X1", "X2"]:
    is_cat = feat.startswith("C")
    x_new, res, ret_pct = feat_eng_single_tuning(np.array(df[feat]), np.array(df["Y"]), w = None, is_cat = is_cat, min_pop = 5000, max_split = np.inf, split_pop_ratio = 5, merge_pct_factor = 1.2, max_retry = 5, step_size = 0.05, check_mono = True)
    df_engineered[feat] = x_new.copy()
    print(f"{feat} at the merge factor of {round(ret_pct, 2)}:")
    print(f"\tLevels: " + str(res[1]))
    print(f"\tDetails: " + str(['(n = ' + str(int(x[0])) + ', r = ' + str(round(x[1] * 100, 2)) + '%)' for x in res[2]]))
    print("")
```
The output is below, where levels represent each bin of the feature, while details contains the number of (weighted) data points and the proportion of the dependent variable `Y = 1` in each bin.
```
C1 at the merge factor of 1.2:
	Levels: [['Block_3_2', 'Block_3_1'], ['Block_2_1', 'Block_2_2'], ['Block_1_2', 'Block_1_3', 'Block_1_1']]
	Details: ['(n = 5000, r = 0.78%)', '(n = 7000, r = 8.23%)', '(n = 8000, r = 16.36%)']

C2 at the merge factor of 1.2:
	Levels: [['Block_1_2', 'Block_1_1'], ['Block_2_1', 'Block_2_2']]
	Details: ['(n = 10000, r = 4.84%)', '(n = 10000, r = 14.4%)']

X1 at the merge factor of 1.2:
	Levels: [(0.0004889402859276082, 4.000479272723119), (4.000479272723119, 29.99338949396259)]
	Details: ['(n = 15000, r = 11.61%)', '(n = 5000, r = 3.66%)']

X2 at the merge factor of 1.2:
	Levels: [(-4.9996916033575385, 0.0024397819476797973), (0.0024397819476797973, 100.03284801353766), (100.03284801353766, 999.88659110659)]
	Details: ['(n = 10000, r = 2.31%)', '(n = 5000, r = 12.62%)', '(n = 5000, r = 21.24%)']
```
To obtain the reconstructed dataframe all at once, we can leverage the function `feat_eng_single_df`:
```python
df_engineered = feat_eng_single_df(df, "Y", w_name = None, cat_cols = ["C1", "C2"], num_cols = ["X1", "X2"], min_pop = 5000, max_split = np.inf, split_pop_ratio = 5, merge_pct_factor = 1.2, max_retry = 5, step_size = 0.05, check_mono = True, verbose = False)

print(df_engineered)
pprint(info_dict)
```
The reconstructed dataframe `df_engineered` after feature engineering is below:
```
       Y       C1       C2       X1       X2
0      0  Block_1  Block_0  Block_0  Block_2
1      0  Block_0  Block_1  Block_1  Block_1
2      0  Block_2  Block_0  Block_0  Block_1
3      0  Block_2  Block_0  Block_0  Block_0
4      0  Block_2  Block_0  Block_1  Block_0
...   ..      ...      ...      ...      ...
19995  0  Block_2  Block_1  Block_0  Block_0
19996  0  Block_1  Block_1  Block_1  Block_0
19997  0  Block_0  Block_1  Block_0  Block_1
19998  0  Block_1  Block_0  Block_0  Block_1
19999  0  Block_1  Block_0  Block_0  Block_0

[20000 rows x 5 columns]
```
The details about the what each level of each engineered feature represents with their corresponding metrics can be extracted from the `info_dict`:
```
{'C1': {'details': [(5000.0, 0.0078),
                    (7000.0, 0.08228571428571428),
                    (8000.0, 0.163625)],
        'engineered_values': ['Block_0', 'Block_1', 'Block_2'],
        'is_cat': True,
        'keep_original': False,
        'levels': [['Block_3_2', 'Block_3_1'],
                   ['Block_2_1', 'Block_2_2'],
                   ['Block_1_2', 'Block_1_3', 'Block_1_1']],
        'merge_factor': 1.2},
 'C2': {'details': [(10000.0, 0.0484), (10000.0, 0.144)],
        'engineered_values': ['Block_0', 'Block_1'],
        'is_cat': True,
        'keep_original': False,
        'levels': [['Block_1_2', 'Block_1_1'], ['Block_2_1', 'Block_2_2']],
        'merge_factor': 1.2},
 'X1': {'details': [(15000.0, 0.11606666666666667), (5000.0, 0.0366)],
        'engineered_values': ['Block_0', 'Block_1'],
        'is_cat': False,
        'keep_original': False,
        'levels': [(0.0004889402859276082, 4.000479272723119),
                   (4.000479272723119, 29.99338949396259)],
        'merge_factor': 1.2},
 'X2': {'details': [(10000.0, 0.0231), (5000.0, 0.1262), (5000.0, 0.2124)],
        'engineered_values': ['Block_0', 'Block_1', 'Block_2'],
        'is_cat': False,
        'keep_original': False,
        'levels': [(-4.9996916033575385, 0.0024397819476797973),
                   (0.0024397819476797973, 100.03284801353766),
                   (100.03284801353766, 999.88659110659)],
        'merge_factor': 1.2}}
```
We conducted a simple evaluation by comparing the performance of the random forest classifier with and without applying this feature engineering pipeline. Experiments have demonstrated <strong>uniformly better performance</strong> when adopting the robo feature engineering.

<table>
<tr> <td colspan=1> <td colspan=4 align="center"> <strong>Without Feature Engineering</strong> <br> (Precision = 38.67%, ROC-AUC = 0.81) <td colspan=4 align="center"> <strong>With Feature Engineering</strong> <br> (Precision = 44.13%, ROC-AUC = 0.87)
<tr> <td colspan=1> <strong>Decile</strong> <td colspan=1> <strong>Accuracy</strong> <td colspan=1> <strong>Precision</strong> <td colspan=1> <strong>Recall</strong> <td colspan=1> <strong>Lift</strong> <td colspan=1> <strong>Accuracy</strong> <td colspan=1> <strong>Precision</strong> <td colspan=1> <strong>Recall</strong> <td colspan=1> <strong>Lift</strong>
<tr> <td colspan=1> <strong>Top 1 Decile</strong> <td colspan=1> 88.55% <td colspan=1> 42.80% <td colspan=1> 42.42% <td colspan=1> 4.27 <td colspan=1> 90.34% <td colspan=1> 52.33% <td colspan=1> 40.32% <td colspan=1> 5.22
<tr> <td colspan=1> <strong>Top 2 Decile</strong> <td colspan=1> 82.95% <td colspan=1> 31.97% <td colspan=1> 62.18% <td colspan=1> 3.19 <td colspan=1> 84.51% <td colspan=1> 35.87% <td colspan=1> 69.26% <td colspan=1> 3.58
<tr> <td colspan=1> <strong>Top 3 Decile</strong> <td colspan=1> 75.41% <td colspan=1> 25.23% <td colspan=1> 74.05% <td colspan=1> 2.52 <td colspan=1> 77.69% <td colspan=1> 28.74% <td colspan=1> 82.93% <td colspan=1> 2.87
<tr> <td colspan=1> <strong>Top 4 Decile</strong> <td colspan=1> 68.47% <td colspan=1> 21.54% <td colspan=1> 81.24% <td colspan=1> 2.15 <td colspan=1> 68.63% <td colspan=1> 22.77% <td colspan=1> 89.12% <td colspan=1> 2.27
</table>
