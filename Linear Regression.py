# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N = 20

X = np.random.random(N)*10 - 5

Y = 0.5 * X - 1 + np.random.randn(N)


# %%
plt.scatter(X, Y)

# %%
model = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(lr=0.01, params=model.parameters())

# %%
X = X.reshape(N, 1)
Y = Y.reshape(N, 1)

inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

type(inputs)
# %%
n_epochs = 300
losses = []
for it in range(n_epochs):
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, targets)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()

    print(f'Epoch {it + 1}/{n_epochs} - Loss: {loss.item(): .4f}')

# %%
plt.plot(losses)
# %%
plt.clf()
predicted = model(inputs).detach().numpy()
plt.scatter(inputs, targets)
plt.plot(inputs, predicted, color='r', label="Predicted")
# plt.spines['left'].set_position(('data', 0))
plt.legend()
#plt.show()


# %%
m = model.weight.data.item()
b = model.bias.item()
print(f'Steigung = {m:.2}')
print(f'Y-Achsenabschnitt = {b:.2}')

# %%
