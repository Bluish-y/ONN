import onn

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from torch.distributions.normal import Normal

onn_exact = onn.OpticalONN(approx = False, alpha0=10).to(onn.device)
onn_approx = onn.OpticalONN(approx=True, alpha0=10)

#plot the model.sa_forward and model.sa_approx_grad and model.sa_exact_grad in one plot for x vals from -20 to 20
x_vals = np.linspace(-20, 20, 100)
y_g = onn_exact.sa_forward(torch.tensor(x_vals)).detach().numpy()
y_exact = onn_exact.grad(torch.tensor(x_vals)).detach().numpy()
y_approx = onn_approx.grad(torch.tensor(x_vals)).detach().numpy()
y_approx_rescaled = onn_approx.sa_approx_grad(torch.tensor(x_vals), scale=0.1).detach().numpy()
plt.figure(figsize=(10, 5))

# Customize axes
ax = plt.gca()
ax.spines['left'].set_color('blue')
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_color('red')
ax.spines['right'].set_linewidth(2)

# Add secondary y-axis for g'(.) values
ax2 = ax.twinx()
ax2.set_ylabel("g'(.)", color="red")
ax2.tick_params(axis='y', colors='red')

ax.set_ylim(-15, 15)
ax2.set_ylim(0, 2.2)

ax.plot(x_vals, y_g, label="g(.)", color="blue")
ax2.plot(x_vals, y_exact, label="g'(.) Exact", color="red")
ax2.plot(x_vals, y_approx, label="g'(.) Approx", linestyle="--", color="red")
ax2.plot(x_vals, y_approx_rescaled, label="g'(.) Approx (Rescaled)", linestyle=":", color="orange")



# Labels and legend
plt.xlabel("x")
ax.set_ylabel("g(.)", color="blue")
ax.tick_params(axis='y', colors='blue')
plt.legend()

plt.title("Activation Function and Gradients")
plt.grid()
plt.tight_layout()
plt.savefig("/data.nst/ysinha/projects/ONN/figs/fig3c.png")
