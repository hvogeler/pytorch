# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

# %%
# Get Data
raw_data = load_breast_cancer()
type(raw_data)
raw_data.keys()
raw_data.feature_names
X = raw_data.data
y = raw_data.target.reshape(-1, 1)
X.shape # 569 rows, 30 features per row
y.shape # (569,)
raw_data.target_names # ['malignant', 'benign']
N, D = X.shape

# %%
# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f'Training Set: {X_train.shape}, {y_train.shape}')
print(f'Test Set: {X_test.shape}, {y_test.shape}. {X_train.shape[0]} + {X_test.shape[0]} = {X_train.shape[0] + X_test.shape[0]}')

# Normalize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

# %%
# Create torch tensors for train inputs and train targets as well as
#    test inputs and targets
X_train_tensor = torch.from_numpy(X_train_std.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))

# Create the model and criterion and optimizer
model = nn.Sequential(
    nn.Linear(D, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Run training loop
n_epochs = 1000
losses = []
errors_per_epoch = []
for epoch in range(n_epochs):
   optimizer.zero_grad()
   outputs = model(X_train_tensor) # forward processing model
   loss = criterion(outputs, y_train_tensor) # calculate loss
   loss.backward() # backward gradient

   losses.append(loss.item())
   err_cnt = 0
   for i in range(len(outputs)):
      outputs_bin = outputs.round()
      if (outputs_bin[i].item() != y_train_tensor[i].item()):
         err_cnt += 1
   errors_per_epoch.append(1 - (err_cnt / len(outputs)))
   if (epoch % (n_epochs / 10) == 0):
      print(f'Epoch {epoch:>4} of {n_epochs}   Loss: {losses[epoch]:.4f}   Errors: {errors_per_epoch[epoch]:.4f}')

   optimizer.step()

plt.plot(losses)
plt.plot(errors_per_epoch)

# %%
# Validate model against test data
X_test_tensor = torch.from_numpy(X_test_std.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

# with torch.no_grad():
# predictions need to be rounded because the sigmoid returns them as floats between 0 and 1
predicts = model(X_test_tensor).detach().round()
print(predicts.shape)
print(y_test_tensor.shape)
predict_error_cnt = 0
false_items = []
for i in range(len(predicts)):
   if (predicts[i].item() != y_test_tensor[i].item()):
      predict_error_cnt += 1
      false_items.append(i)
   
print(f'False items = {false_items}')

print(f'Wrong predictions: {predict_error_cnt}')


# %%
# save the model