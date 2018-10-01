
# coding: utf-8

# ### Ref: Kernels
# 
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
# 
# https://www.kaggle.com/artgor/nn-baseline
# 
# https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, 
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train_raw = load_df('../input/train.csv')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_test_raw = load_df('../input/test.csv')")


# ###  keep _raw for resuming point

# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train = df_train_raw.copy()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_test = df_test_raw.copy()')


# ### apply adjustments to both train and test datasets

# In[ ]:


def applyEdits(df):
    # fill NA's
    df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    df['trafficSource.isTrueDirect'].fillna(False, inplace=True)

    l_fillwithzero = ['totals.bounces','totals.newVisits','totals.pageviews',
    'trafficSource.adContent','trafficSource.keyword','trafficSource.adwordsClickInfo.adNetworkType',
    'trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.gclId',
    'trafficSource.adwordsClickInfo.gclId','trafficSource.adwordsClickInfo.page',
    'trafficSource.adwordsClickInfo.slot']
    for x in l_fillwithzero:
        df[x] = df[x].fillna(0)

    # add / edit columns
    df['browser_category'] = df['device.browser'] + '_' + df['device.deviceCategory']
    df['browser_operatingSystem'] = df['device.browser'] + '_' + df['device.operatingSystem']
    df['source_country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']                                                                   

    
    df['dummy'] = 1
    df['user_cumcnt_per_day'] = (df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].cumcount()+1)
    df['user_sum_per_day'] = df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].transform(sum)
    df['user_cumcnt_sum_ratio_per_day'] = df['user_cumcnt_per_day'] / df['user_sum_per_day'] 
    df.drop('dummy', axis=1, inplace=True)

    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train = applyEdits(df_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_test = applyEdits(df_test)')


# ### apply only for train dataset

# In[ ]:


# to float and log
df_train['totals.transactionRevenue'] = df_train['totals.transactionRevenue'].astype(float)
df_train['totals.transactionRevenue'] = np.log1p(df_train['totals.transactionRevenue'].fillna(0))


# ### delete columns with no valid info

# In[ ]:


get_ipython().run_cell_magic('time', '', 'cols_to_drop = [col for col in df_train.columns if df_train[col].nunique() == 1]\ndf_train.drop(cols_to_drop, axis=1, inplace=True)\ndf_test.drop([col for col in cols_to_drop if col in df_test.columns], axis=1, inplace=True)')


# ### clean up date column

# In[ ]:


def cleanDate(df):
    df['date'] = pd.to_datetime(df['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
    return df


df_train = cleanDate(df_train)
df_test = cleanDate(df_test)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef parseDates(df):\n    df['date'] = pd.to_datetime(df['date'])\n    df['month'] = df['date'].dt.month\n    df['day'] = df['date'].dt.day\n    df['weekday'] = df['date'].dt.weekday\n    df['weekofyear'] = df['date'].dt.weekofyear\n    \n    return df\n\n\ndf_train = parseDates(df_train)\ndf_test = parseDates(df_test)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def strToNum(df):\n    l = [ 'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews',]\n    for x in l:\n        df[x] = df[x].values.astype(np.int64)\n    return df\n\ndf_train = strToNum(df_train)\ndf_test = strToNum(df_test)")


# In[ ]:


no_use = ['date', 'fullVisitorId', 'sessionId', 'visitId', 'visitStartTime', 'totals.transactionRevenue', 'trafficSource.referralPath']

cat_cols = [x for x in df_train.columns if x not in no_use and type(df_train[x][0]) == str]

num_cols = [x for x in df_train.columns if x not in no_use and x not in cat_cols and type(df_train[x][0]) != str and type(df_train[x][0]) != bool]


# ### Encode category strings to number labels ** use same number for same category accross train & test datasets

# In[ ]:


get_ipython().run_cell_magic('time', '', "max_values = {}\nfor col in cat_cols:\n    lbl = LabelEncoder()\n    lbl.fit(list(df_train[col].values.astype('str')) + list(df_test[col].values.astype('str')))\n    df_train[col] = lbl.transform(list(df_train[col].values.astype('str')))\n    df_test[col] = lbl.transform(list(df_test[col].values.astype('str')))\n    max_values[col] = max(df_train[col].max(), df_test[col].max())  + 2")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def boolToNum(df):\n    l = ['trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect' ]\n    for x in l:\n        df[x] = df[x].values.astype(np.int64) * 1\n    return df\ndf_train = boolToNum(df_train)\ndf_test = boolToNum(df_test)")


# In[ ]:


df_train.drop('trafficSource.referralPath', axis=1, inplace=True)
df_test.drop('trafficSource.referralPath', axis=1, inplace=True)


# ## -------------------------------------------------------------for reseting ---------------------------------------------------------------

# In[ ]:


get_ipython().run_cell_magic('time', '', "#export\ndf_train.to_csv('train_wip.csv')\ndf_test.to_csv('test_wip.csv')\n\n#save\n#df_train_forReset = df_train.copy()\n#df_test_forReset = df_test.copy()\n\n#load\n#df_train = df_train_forReset.copy()\n#df_test = df_test_forReset.copy()")


# In[ ]:



def aggregate_by_users(df, cat_cols):
    aggs = {
        'date': ['min', 'max'],
        'totals.transactionRevenue': ['sum', 'size'],
        'totals.hits': ['sum', 'min', 'max', 'mean', 'median'],
        'totals.pageviews': ['sum', 'min', 'max', 'mean', 'median'],
        'totals.bounces': ['sum', 'mean', 'median'],
        'totals.newVisits': ['sum', 'mean', 'median']
    }

    for f in cat_cols + ['weekday', 'day', 'month', 'weekofyear']:
        aggs[f] = ['min', 'max', 'mean', 'median', 'var', 'std']

    users = df.groupby('fullVisitorId').agg(aggs)

    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    users.columns = new_columns

    users['date_diff'] = (users.date_max - users.date_min).astype(np.int64) // (24 * 3600 * 1e9)
    
    return users


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_user_train = aggregate_by_users(df_train, cat_cols)\n\ndf_test['totals.transactionRevenue'] = 0\ndf_user_test = aggregate_by_users(df_test, cat_cols)\n")


# ## save aggregated csv

# In[ ]:


get_ipython().run_cell_magic('time', '', "#export\ndf_user_train.to_csv('user_train_wip.csv')\ndf_user_test.to_csv('user_test_wip.csv')")


# ## Load saved CSV

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf_user_train = pd.read_csv('../input/trying-google-analytics-competition/user_train_wip.csv')\ndf_user_test = pd.read_csv('../input/trying-google-analytics-competition/user_test_wip.csv')")


# ## model

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf_y = df_user_train['totals.transactionRevenue_sum']\ndf_x = df_user_train.drop(['date_min', 'date_max', 'totals.transactionRevenue_sum'], axis=1)\ndf_tgt = df_user_test.drop(['date_min', 'date_max', 'totals.transactionRevenue_sum'], axis=1)")


# 
# # ------------------------ work in progress --------------------

# In[ ]:


get_ipython().run_cell_magic('time', '', "import lightgbm as lgb\nfrom sklearn.model_selection import KFold\n\nfolds = KFold(n_splits=5, shuffle=True, random_state=1)\n\npred_tgt = np.zeros(df_tgt.shape[0])\npred_train = np.zeros(df_x.shape[0])\nl_scores = []\n\nlgb_params = {\n    'learning_rate': 0.03,\n    'n_estimators': 2000,\n    'num_leaves': 128,\n    'subsample': 0.2217,\n    'colsample_bytree': 0.6810,\n    'min_split_gain': np.power(10.0, -4.9380),\n    'reg_alpha': np.power(10.0, -3.2454),\n    'reg_lambda': np.power(10.0, -4.8571),\n    'min_child_weight': np.power(10.0, 2),\n    'silent': True\n}\n\nfor i_fold, (i_trainset, i_testset) in enumerate(folds.split(df_x)):\n    model = lgb.LGBMRegressor(**lgb_params)\n\n    model.fit(\n        df_x.iloc[i_trainset], df_y.iloc[i_trainset],\n        evalset=[(df_x.iloc[i_trainset], df_y.iloc[i_trainset]), (df_x.iloc[i_testset], df_y.iloc[i_testset])],\n        evalmetric='rmse',\n        early_stopping_rounds=100,\n        verbose=0\n    )\n    \n    oof_preds[i_testset] = model.predict(df_x.iloc[i_testset])\n\n    print('Fold %d RMSE (raw output) : %.5f' % (i_fold + 1, rmse(df_y.iloc[i_testset], oof_preds[i_testset])))\n    pred_train[pred_train < 0] = 0\n    c_scores.append(rmse(df_y.iloc[i_testset], pred_train[i_testset]))\n    print('Fold %d RMSE : %.5f' % (i_fold + 1, l_scores[-1]))\n\n    pred_target_tmp = model.predict(df_tgt)\n    pred_target_tmp[pred_target_tmp < 0] = 0\n    pred_tgt += pred_target_tmp / folds.n_splits\n\nprint('Full OOF RMSE (zero clipped): %.5f +/- %.5f' % (rmse(df_y, pred_train), float(np.std(l_scores))))")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_tgt[\'PredictedLogRevenue\'] = pred_tgt\ndf_tgt[[\'PredictedLogRevenue\']].to_csv("output.csv", index=True)\n\nprint(\'Submission data shape : {}\'.format(sub_users[[\'PredictedLogRevenue\']].shape))')


# In[ ]:


hist, bin_edges = np.histogram(np.hstack((pred_train, pred_tgt)), bins=25)
plt.figure(figsize=(12, 7))
plt.title('Distributions of OOF and TEST predictions', fontsize=15, fontweight='bold')
plt.hist(pred_train, label='OOF predictions', alpha=.6, bins=bin_edges, density=True, log=True)
plt.hist(pred_tgt, label='TEST predictions', alpha=.6, bins=bin_edges, density=True, log=True)
plt.legend()
plt.savefig('distributions.png')

