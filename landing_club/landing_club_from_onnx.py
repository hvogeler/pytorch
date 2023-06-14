# %%
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pickle as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import onnx
import onnxruntime

from landing_club_wrapper import Landing_Club_Wrapper

import matplotlib.pyplot as plt

# %%
# Read Data
SMALL_SAMPLE_DATASET_NAME = 'data/landing_club/accepted_2007_to_2018Q4_small_sample.csv'
acc_raw_df = pd.read_csv(SMALL_SAMPLE_DATASET_NAME)
# col_names = ['loan_amnt', 'home_ownership', 'annual_inc', 'purpose', 'addr_state', 'term', 'emp_length', 'int_rate']
# col_names = ['loan_amnt', 'home_ownership', 'annual_inc','term', 'emp_length', 'int_rate']
col_names = ['loan_amnt', 'home_ownership', 'annual_inc', 'int_rate']
col_label = 'int_rate'


# %%
# Remove records with nan
acc_df = acc_raw_df.loc[:, col_names]
print(acc_df)

# %%
# DataFrame to Numpy array
X_df = acc_df.drop(columns=[col_label])
y_df = acc_df[col_label]
X = X_df.to_numpy()
y = y_df.to_numpy()

# %%
# Test
model = onnx.load("landing_club_model.onnx")
onnx.checker.check_model(model)

ort_session = onnxruntime.InferenceSession("landing_club_model.onnx")

# %%
wrapper_file = open('landing_club_wrapper.pickle', 'rb')
wrapper: Landing_Club_Wrapper = p.load(wrapper_file)

for i in range(2):
    print(X[i])
    prepared_features = np.asarray(wrapper.preprocess(X[i, 0], X[i, 1], X[i, 2])).astype(np.float32)
    predict = ort_session.run(['prediction'], {'features': [prepared_features]})
    post_predict = wrapper.postprocess(predict[0].item())
    print(f'{X[i]} -> {prepared_features} -> {post_predict}')


# %%
