# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

# %%
# Get Data
test_dataset = datasets.MNIST(
    root='.',
    train=True,
    download=True
)

X_test = test_dataset.data
y_test = test_dataset.targets

X_test.shape
y_test.shape


# %%
print('X_test = ', X_test)
print(47040000 / 784)
y = X_test.view(-1, 28 * 28)
print("y = ", y, 28*28)

# z = torch.IntTensor([[1,[10,11]],[2,[12,13]],[3,[14,15]]])
# print(z.view(-1, 3))

# %%
