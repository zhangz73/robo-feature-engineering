import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score, confusion_matrix
from feature_engineer import feat_eng_single_df, get_df_from_info_dict

df_train = pd.read_csv("Data/train_data.csv")
df_test = pd.read_csv("Data/test_data.csv")

def get_pipeline(cat_cols, num_cols):
    pipe = Pipeline(steps = [
        ("preprocessor",
            ColumnTransformer(
                transformers = [
                    ("cat", Pipeline(steps = [
                        ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
                        ("ohe", OneHotEncoder(handle_unknown = "ignore"))
                    ]), cat_cols),
                    ("num", Pipeline(steps = [
                        ("imputer", SimpleImputer(strategy = "median")),
                        ("scaler", MinMaxScaler())
                    ]), num_cols)
                ]
            )
        ),
        ("model", RandomForestClassifier())
    ])
    return pipe

def train_n_predict(df_train, df_test, cat_cols = [], num_cols = []):
    model = get_pipeline(cat_cols, num_cols)
    X_train = df_train[[x for x in df_train.columns if x != "Y"]]
    Y_train = df_train["Y"]
    X_test = df_test[[x for x in df_test.columns if x != "Y"]]
    Y_test = df_test["Y"]
    model = model.fit(X_train, Y_train)
    Y_pred = model.predict_proba(X_test)[:,1]
    return Y_test, Y_pred

def eval(Y_true, Y_pred):
    roc_auc = roc_auc_score(Y_true, Y_pred)
    prec = average_precision_score(Y_true, Y_pred)
    return prec, roc_auc

def eval_binary(Y_true, Y_pred, cutoff_pct = 0.5):
    cutoff = np.quantile(Y_pred, 1 - cutoff_pct)
    pred = binarize([Y_pred], threshold = cutoff)[0]
    cm = confusion_matrix(Y_true, pred)
    TN, FN, TP, FP = cm[0][0], cm[1][0], cm[1][1], cm[0][1]
    accu = (TP + TN) / (TN + FN + TP + FP)
    recall = TP / (TP + FN)
    prec = TP / (TP + FP)
    lift = prec / ((TP + FN) / (TN + FN + TP + FP))
    return accu, prec, recall, lift

def eval_deciles(Y_true, Y_pred):
    dcl_lst = []
    accu_lst = []
    prec_lst = []
    recall_lst = []
    lift_lst = []
    for dcl in range(1, 5):
        if dcl == 0:
            cutoff_pct = np.mean(Y_true)
            msg = "Natural Event Rate"
        else:
            cutoff_pct = dcl / 10
            msg = f"Top {dcl} Decile"
        accu, prec, recall, lift = eval_binary(Y_true, Y_pred, cutoff_pct = cutoff_pct)
        dcl_lst.append(msg)
        accu_lst.append(str(round(accu * 100, 2)) + "%")
        prec_lst.append(str(round(prec * 100, 2)) + "%")
        recall_lst.append(str(round(recall * 100, 2)) + "%")
        lift_lst.append(round(lift, 2))
    df_ret = pd.DataFrame.from_dict({"Decile": dcl_lst, "Accuracy": accu_lst, "Precision": prec_lst, "Recall": recall_lst, "Lift": lift_lst})
    return df_ret

df_engineered_train, info_dict = feat_eng_single_df(df_train, "Y", w_name = None, cat_cols = ["C1", "C2"], num_cols = ["X1", "X2"], min_pop = 5000, max_split = np.inf, split_pop_ratio = 5, merge_pct_factor = 1.2, max_retry = 5, step_size = 0.05, check_mono = True, verbose = False)
df_engineered_test = get_df_from_info_dict(df_test, "Y", w_name = None, info_dict = info_dict)

for feat in ["C1", "C2"]:
    df_train[feat] = pd.Categorical(df_train[feat])
    df_test[feat] = pd.Categorical(df_test[feat])

for feat in ["X1", "X2", "C1", "C2"]:
    df_engineered_train[feat] = pd.Categorical(df_engineered_train[feat])
    df_engineered_test[feat] = pd.Categorical(df_engineered_test[feat])

Y_true, Y_pred = train_n_predict(df_train, df_test, cat_cols = ["C1", "C2"], num_cols = ["X1", "X2"])
Y_true_eng, Y_pred_eng = train_n_predict(df_engineered_train, df_engineered_test, cat_cols = ["C1", "C2", "X1", "X2"])

prec, roc_auc = eval(Y_true, Y_pred)
prec_eng, roc_auc_eng = eval(Y_true_eng, Y_pred_eng)

print(f"Without Feature Engineering: Precision = {round(prec * 100, 2)}%, ROC-AUC = {round(roc_auc, 2)}")
print(f"With Feature Engineering: Precision = {round(prec_eng * 100, 2)}%, ROC-AUC = {round(roc_auc_eng, 2)}")

df_ret = eval_deciles(Y_true, Y_pred)
df_ret_eng = eval_deciles(Y_true_eng, Y_pred_eng)
print(df_ret)
print(df_ret_eng)
