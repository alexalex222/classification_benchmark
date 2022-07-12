# %%
import numpy as np
import torch
import math
from matplotlib import pyplot as plt


# %%
x = np.random.randn(200, 2)
y = 0.4 * x[:, 0] - 0.7 * x[:, 1] + np.random.randn(200) * 0.1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
dtype = torch.float

theta_1 = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
theta_2 = torch.zeros((), device=device, dtype=dtype, requires_grad=True)

x_train = torch.tensor(x, device=device, dtype=dtype)
y_train = torch.tensor(y, device=device, dtype=dtype)

learning_rate = 1e-4

# %%
def linear_model(
    x: torch.Tensor
):
    return theta_1 * x[:, 0] + theta_2 * x[:, 1]

def mean_squared_error(
    y_pred: torch.Tensor, 
    y: torch.Tensor,
    ):
    return (y_pred - y).pow(2).sum()


# %%
loss_list = []
theta_1_list = []
theta_2_list = []
theta_1_list.append(theta_1.item())
theta_2_list.append(theta_2.item())
for t in range(200):
    y_pred = linear_model(x_train)

    loss = (y_pred - y_train).pow(2).sum()

    loss.backward()

    with torch.no_grad():
        theta_1 -= learning_rate * theta_1.grad
        theta_2 -= learning_rate * theta_2.grad

        # manually zero out the gradients
        theta_1.grad = None
        theta_2.grad = None

    loss_list.append(loss.item())
    theta_1_list.append(theta_1.item())
    theta_2_list.append(theta_2.item())


# %%
fig, axes = plt.subplots(3, 1, figsize=(15, 15))
axes[0].plot(theta_1_list, label=r'$\theta_1$')
axes[0].legend()
axes[1].plot(theta_2_list, label=r'$\theta_1$')
axes[1].legend()
axes[2].plot(loss_list, label='loss')
axes[2].legend()
plt.show()

# %%
ax = plt.axes(projection='3d')
ax.plot3D(theta_1_list, theta_2_list, loss_list, 'gray')
plt.show()
# %%
