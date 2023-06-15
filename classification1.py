# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------
# Get data
#--------------------------------------------------
data = load_breast_cancer()
data.keys()

tgs = []
for tg in data.target:
    tgs.append(data.target_names[tg])

data.data.shape
data.target.shape
# clearprint(tgs)

# %%
#--------------------------------------------------
# Split data
#--------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.33
)

N, D = X_train.shape

print(f'X_train: {X_train.size}, X_test.size: {X_test.size}, ratio: {X_test.size/X_train.size:.2f}')
print(f'Y_train: {y_train.size}, y_test.size: {y_test.size}, ratio: {y_test.size/y_train.size:.2f}')
# print(X_test[0])

# %%
#--------------------------------------------------
# Pre-process (Scale) data
#--------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(X_test[0])

# %%
#--------------------------------------------------
# Build model
#--------------------------------------------------
learn_rate = 0.1
model = nn.Sequential(
    nn.Linear(D, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
optimizer = torch.optim.Adam(model.parameters())

# %%
#--------------------------------------------------
# Make tensors
#--------------------------------------------------
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
# print(y_train_tensor)

# %%
#--------------------------------------------------
# Train model
#--------------------------------------------------
n_epochs = 1000
train_losses = []
train_accuracies = []
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)

    # loss is a single value describing the overall loss for 
    # the whole training set. Loss is a tensor including the backward 
    # gradient function
    loss = criterion(outputs, y_train_tensor)
   
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    print(f'Epoch {epoch + 1}: loss = {loss.item():.4f}')

    with torch.no_grad():
        accuracy_train = np.mean(outputs.numpy())
        train_accuracies.append(accuracy_train)
        # print(f'  Accuracy = {accuracy_train:.4f}')

# %%
plt.plot(train_losses, 'g', label='train losses')
plt.plot(train_accuracies)
plt.legend()

# %%
with torch.no_grad():
    predict = model(X_test_tensor)
    predict = np.round(predict.numpy())

    pos_cnt = 0
    for p in predict:
        if (p):
            pos_cnt = pos_cnt + 1

    print(f'Prediction: {pos_cnt} of {predict.shape[0]}: {pos_cnt * 100 / predict.shape[0]:.2f}%')
    print(f'Prediction: {pos_cnt} of {predict.shape[0]}: {np.mean(predict) * 100:.2f}%')

    py_cnt = 0
    for py in y_test:
        if (py):
            py_cnt = py_cnt + 1

    print(f'Reality: {py_cnt} of {y_test.shape[0]}: {np.mean(y_test) * 100:.2f}%')


# %%
