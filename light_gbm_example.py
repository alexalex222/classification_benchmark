# %%
import os
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

# %%
print('Loading data...')
# load or create your dataset
binary_example_dir = './examples/binary_classification'
df_train = pd.read_csv(os.path.join(binary_example_dir, 'binary.train'), header=None, sep='\t')
df_test = pd.read_csv(os.path.join(binary_example_dir, 'binary.test'), header=None, sep='\t')
W_train = pd.read_csv(os.path.join(binary_example_dir, 'binary.train.weight'), header=None)[0]
W_test = pd.read_csv(os.path.join(binary_example_dir, 'binary.test.weight'), header=None)[0]

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

num_train, num_feature = X_train.shape


# %%
# create dataset for lightgbm
# if you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(X_train, y_train,
                        weight=W_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                       weight=W_test, free_raw_data=False)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# generate feature names
feature_name = [f'feature_{col}' for col in range(num_feature)]

# %%
print('Starting training...')
# feature_name and categorical_feature
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,  # eval training data
                feature_name=feature_name,
                categorical_feature=[21])

print('Finished first 10 rounds...')
# check feature name
print(f'7th feature name is: {lgb_train.feature_name[6]}')


# %%
print('Saving model...')
# save model to file
gbm.save_model('./examples/binary_classification/model.txt')

print('Dumping model to JSON...')
# dump model to JSON (and save to file)
model_json = gbm.dump_model()

with open('./examples/binary_classification/model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)


# %%
# feature names
print(f'Feature names: {gbm.feature_name()}')

# feature importances
print(f'Feature importances: {list(gbm.feature_importance())}')


# %%
# load model to predict
bst = lgb.Booster(model_file='./examples/binary_classification/model.txt')
# can only predict with the best iteration (or the saving iteration)
y_pred = bst.predict(X_test)
# eval with loaded model
rmse_loaded_model = mean_squared_error(y_test, y_pred) ** 0.5
print(f"The RMSE of loaded model's prediction is: {rmse_loaded_model}")


# %%
# continue training
# init_model accepts:
# 1. model file name
# 2. Booster()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model='./examples/binary_classification/model.txt',
                valid_sets=lgb_eval)

print('Finished 10 - 20 rounds with model file...')

# decay learning rates
# reset_parameter callback accepts:
# 1. list with length = num_boost_round
# 2. function(curr_iter)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=lgb_eval,
                callbacks=[lgb.reset_parameter(learning_rate=lambda iter: 0.05 * (0.99 ** iter))])

print('Finished 20 - 30 rounds with decay learning rates...')


# %%
# change other parameters during training
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=lgb_eval,
                callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])

print('Finished 30 - 40 rounds with changing bagging_fraction...')


# %%
