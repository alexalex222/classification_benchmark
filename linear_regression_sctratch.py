# %%
import numpy as np
import torch
import math


# %%
x = np.random.randn(200, 2)
y = 0.4 * x[:, 0] - 0.7 * x[:, 1] + np.random.randn(200) * 0.1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
dtype = torch.float

theta_1 = torch.randn((), device=device, dtype=dtype, requires_grad=True)
theta_2 = torch.randn((), device=device, dtype=dtype, requires_grad=True)

x = torch.tensor(x, device=device, dtype=dtype)
y = torch.tensor(y, device=device, dtype=dtype)

learning_rate = 1e-3

for t in range(200):
    y_pred = theta_1 * x[:, 0] + theta_2 * x[:, 1]

    loss = (y_pred - y).pow(2).sum()

    loss.backward()

    with torch.no_grad():
        theta_1 -= learning_rate * theta_1.grad
        theta_2 -= learning_rate * theta_2.grad

        # manually zero out the gradients
        theta_1.grad = None
        theta_2.grad = None



