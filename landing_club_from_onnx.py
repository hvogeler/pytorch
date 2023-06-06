# %%
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import onnx
import onnxruntime

import matplotlib.pyplot as plt

# %%
# Read Data
acc_raw_df = pd.read_csv('data/landing_club/accepted_2007_to_2018Q4.csv')
# col_names = ['loan_amnt', 'home_ownership', 'annual_inc', 'purpose', 'addr_state', 'term', 'emp_length', 'int_rate']
# col_names = ['loan_amnt', 'home_ownership', 'annual_inc','term', 'emp_length', 'int_rate']
col_names = ['loan_amnt', 'home_ownership', 'annual_inc', 'int_rate']


# %%
# Remove records with nan
acc_df = acc_raw_df.loc[:, col_names]
#             .query('loan_amnt != 0')

acc_df = acc_df.dropna()
print(acc_df.sample(10))
acc_df.describe()
acc_df.int_rate.describe()


# %%
# Clean Features


# %%
# Declare helper methods

def encode_one_hot(df):
   categorical_cols = list(acc_df.select_dtypes(include=['object']))
   print(categorical_cols)
   one_hot_enc = OneHotEncoder(sparse_output=False)
   for col in categorical_cols:
      label_encoded = one_hot_enc.fit_transform(df[col].to_numpy().reshape(-1, 1))
      print(col)
      df[col] = label_encoded.tolist()
   return df

def encode_label(df):
   categorical_cols = list(acc_df.select_dtypes(include=['object']))
   print(categorical_cols)
   label_enc = LabelEncoder()
   for col in categorical_cols:
      label_encoded = label_enc.fit_transform(df[col])
      df[col] = label_encoded
   return df


# %%
# Encode Categories
pd.options.display.max_rows=999

# acc_df.dropna(subset=['int_rate'], inplace=True)
acc_df.dropna(inplace=True)
acc_df_c = encode_label(acc_df.copy())
acc_df_c.head(9)



# %%
# Model definition
X_df = acc_df_c.drop(columns=['int_rate'])
y_df = acc_df_c['int_rate']
feature_dim = len(X_df.columns)



# %%
# Create Tensors
X = X_df.to_numpy()
y = y_df.to_numpy()
scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tsor = torch.from_numpy(X_train.astype(np.float32))
X_test_tsor = torch.from_numpy(X_test.astype(np.float32))
y_train_tsor = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
y_test_tsor = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)


print(X_train_tsor[0:9])
print(y_train_tsor[0:9])



# %%
# Test
model = onnx.load("seq_model.onnx")
onnx.checker.check_model(model)

ort_session = onnxruntime.InferenceSession("seq_model.onnx")

for i in range(0, 10):
   ort_inputs = {ort_session.get_inputs()[0].name: X_test[i].astype(np.float32)}
   pred = ort_session.run(['int_rate'], ort_inputs)
   print(pred[0][0])

# with torch.no_grad():
#    preds = seq_model(X_test_tsor)
#    X_test_feat_0 = X_test[2:, 0:1].reshape(-1)
#    print(X_test_feat_0)
#    print(y_test)
#    plt.cla()
#    plt.scatter(X_test_feat_0[0:30], y_test[0:30])
#    plt.scatter(X_test_feat_0[0:30], preds[0:30])



# %%
