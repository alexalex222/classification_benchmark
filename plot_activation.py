# %%
import torch
from torch import nn
from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 200
plt.rcParams['text.usetex'] = True



# %%
tanh_func = nn.Tanh()
relu_func = nn.ReLU()
sigmoid_func = nn.Sigmoid()
x = torch.linspace(-10, 10, 2000)
y_relu = relu_func(x).numpy()
y_tanh = tanh_func(x).numpy()
y_sigmoid = sigmoid_func(x).numpy()
x = x.numpy()


# %%
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].plot(x, y_sigmoid, color='black')
axes[0].grid()
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel(r'$\sigma_{\mathrm{Sigmoid}}(x)$')
axes[1].plot(x, y_tanh, color='black')
axes[1].grid()
axes[1].set_xlabel(r'$x$')
axes[1].set_ylabel(r'$\sigma_{\mathrm{Tanh}}(x)$')
axes[2].plot(x, y_relu, color='black')
axes[2].grid()
axes[2].set_xlabel(r'$x$')
axes[2].set_ylabel(r'$\sigma_{\mathrm{ReLU}}(x)$')
plt.tight_layout()
plt.show()


# %%
# fig.savefig('saved_models/activation_functions.eps')
