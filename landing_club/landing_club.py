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
import torch.onnx

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
# int_rates = np.unique(acc_df['int_rate'])
# len(int_rates)
# np.min(int_rates)
# int_rates.describe()

# %%
# Clean Features

# terms_ = acc_df['term'].str.strip().str.split(' ')
# terms_ = list(map(lambda x: int(x[0]), terms_))
# acc_df['term'] = terms_
# print(acc_df.sample(10))

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
# Model deklaration
class MyModel(nn.Module):
   def __init__(self, input_dim, output_dim):
      super().__init__()
      self.fc1 = nn.Linear(input_dim, 5)
      self.relu1 = nn.ReLU()
      self.fc2 = nn.Linear(5, 5)
      self.relu2 = nn.ReLU()
      self.fc3 = nn.Linear(5, 1)

   def forward(self, X):
      x = self.fc1(X)
      x = self.relu1(x)
      x = self.fc2(x)
      x = self.relu2(x)
      x = self.fc3(x)
      return x


# %%
# Model definition
X_df = acc_df_c.drop(columns=['int_rate'])
y_df = acc_df_c['int_rate']
feature_dim = len(X_df.columns)
myModel = MyModel(feature_dim, 1)

seq_model = nn.Sequential(
   nn.Linear(len(X_df.columns), 5),
   nn.Sigmoid(),
   nn.Linear(5, 5),
   nn.Sigmoid(),
   nn.Linear(5, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(lr=0.1, params=seq_model.parameters())


# %%
# Create Tensors
X = X_df.to_numpy()
y = y_df.to_numpy()
sc_x = StandardScaler()
sc_y = StandardScaler()

X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2)
y_train_orig = y_train_orig.reshape(-1, 1)
y_test_orig = y_test_orig.reshape(-1, 1)
X_train = sc_x.fit_transform(X_train_orig)
X_test = sc_x.transform(X_test_orig)
y_train = sc_y.fit_transform(y_train_orig)
y_test = sc_y.transform(y_test_orig)

X_train_tsor = torch.from_numpy(X_train.astype(np.float32))
X_test_tsor = torch.from_numpy(X_test.astype(np.float32))
y_train_tsor = torch.from_numpy(y_train.astype(np.float32))
y_test_tsor = torch.from_numpy(y_test.astype(np.float32))


print(X_train_tsor[0:9])
print(y_train_tsor[0:9])
print(f'Encode {y_train_orig[5][0]} = {y_train[5][0]}, Decode = {sc_y.inverse_transform(y_train[5][0].reshape(-1, 1)).reshape(1)[0]}')

# %%
# Training

# o = myModel.forward(X_train_tsor)
# print(o[10:30])

losses = []
n_epochs = 100
for n in range(n_epochs):
   optimizer.zero_grad()
   # outputs = myModel.forward(X_train_tsor)
   outputs = seq_model(X_train_tsor)
   
   loss = criterion(outputs, y_train_tsor)
   losses.append(loss)
   # print(loss.item())
   if (n % 10 == 0):
      print(f'Loss on epoch {n:04}: {loss.item():.4f} - interest rate: {outputs[10].item():.4f} / {y_train_tsor[10].item():.4f}')
   loss.backward()
   optimizer.step()

# %%
# Test

with torch.no_grad():
   preds = seq_model(X_test_tsor)
   X_test_feat_0 = X_test[2:, 0:1].reshape(-1)
   print(X_test_feat_0)
   for i in range(0, 10):
      print(preds[i])
   plt.cla()
   plt.scatter(X_test_feat_0[0:30], y_test[0:30])
   plt.scatter(X_test_feat_0[0:30], preds[0:30])


# %%
# ONNX Export

dummy_input = X_test_tsor[0]
print(dummy_input)
print(col_names)
seq_model.eval()
torch.onnx.export(seq_model,
                  dummy_input,
                  "seq_model.onnx",
                  export_params=True,
                  opset_version=10,
                  input_names=['model_input'],
                  output_names=['int_rate'])

# %%
