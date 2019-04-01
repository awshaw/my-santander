import lightgbm as lgb
import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold


def create_row_features(df, index=None):
    base = df.copy()
    if index is None:
        var_add = ''
    else:
        df = df[index]
        var_add = str(idx[0]) + '-' + str(idx[-1])

    df['row_pos_sum' + var_add] = base[base > 0].sum(axis=1, skipna=True)
    df['row_neg_sum' + var_add] = base[base < 0].sum(axis=1, skipna=True)
    df['tot' + var_add] = base.sum(axis=1)
    df['row_max' + var_add] = base.max(axis=1, skipna=True)
    df['row_min' + var_add] = base.min(axis=1, skipna=True)
    df['row_range' + var_add] = df['row_max' + var_add] - df['row_min' + var_add]
    df['row_std' + var_add] = base.std(axis=1)
    df['row_skew' + var_add] = base.skew(axis=1)
    df['row_kurtosis' + var_add] = base.kurtosis(axis=1)
    df['row_median' + var_add] = base.median(axis=1)

    return df


def augment(train,num_n=1,num_p=2):
    tmp=train.copy()
    
    n=tmp[tmp.target==0]
    for i in range(num_n):
        tmp.append(n.apply(lambda x:x.values.take(np.random.permutation(len(n)))))
    
    for i in range(num_p):
        p=tmp[tmp.target>0]
        tmp.append(p.apply(lambda x:x.values.take(np.random.permutation(len(p)))))
    
    return pd.concat(tmp)


path=Path("../input/")
train=pd.read_csv(path/"train.csv").drop("ID_code",axis=1)
test=pd.read_csv(path/"test.csv").drop("ID_code",axis=1)

exp_tr = train.copy()
exp_test = test.copy()

exp_tr = create_row_features(exp_tr)
exp_test = create_row_features(exp_test)
print("Train shape: {}, test shape: {}".format(exp_tr.shape, exp_test.shape))

param = {
   "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "bagging_seed" : 10,
    "verbosity" : 1,
}

result=np.zeros(test.shape[0])

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=10)
for counter,(train_index, valid_index) in enumerate(rskf.split(exp_tr, train.target),1):
    print (counter)
    
    #Train data
    t=exp_tr.iloc[train_index]
    t=augment(t)
    trn_data = lgb.Dataset(t.drop("target",axis=1), label=t.target)
    
    #Validation data
    v=exp_tr.iloc[valid_index]
    val_data = lgb.Dataset(v.drop("target",axis=1), label=v.target)
    
    #Train & Predict
    model = lgb.train(param, trn_data, 1000000, valid_sets = [val_data], verbose_eval=500, early_stopping_rounds = 3000)
    result +=model.predict(test)


submission = pd.read_csv(path/'sample_submission.csv')
submission['target'] = result/counter
filename="lgb_{:%Y-%m-%d_%H_%M}_sub.csv".format(datetime.now())
submission.to_csv(filename, index=False)