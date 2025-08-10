import time
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def compute_similarity_S(f, gp, sigma=4.0, num_points=1000):
    """
    Compute the similarity score S between f(z) and g'(z) weighted by Gaussian p(z).
    """
    # Domain wide enough for Gaussian with sigma=4
    z = np.linspace(-5*sigma, 5*sigma, num_points)
    dz = z[1] - z[0]  # uniform spacing

    f_vals = f(z)
    g_vals = gp(z)

    p = norm.pdf(z, loc=0, scale=sigma)

    numerator = np.abs(np.sum(f_vals * g_vals * p) * dz) ** 2
    denom1 = np.sum(f_vals**2 * p) * dz
    denom2 = np.sum(g_vals**2 * p) * dz

    S = numerator / (denom1 * denom2 + 1e-12)  # avoid divide-by-zero
    return S

def sa_exact_grad_numpy(alpha,z):
    return (1 + (alpha * z**2) / (1 + z**2)**2) * np.exp(-alpha / (2 * (1 + z**2)))

def sa_approx_grad_numpy(alpha, scale, z):
    return (1 + alpha * scale) * np.exp(-alpha / (2 * (1 + z**2)))

approx_errors = []
for alpha in range(0,50):
    S = compute_similarity_S(
        lambda z: sa_exact_grad_numpy(alpha, z),
        lambda z: sa_approx_grad_numpy(alpha, 0, z))
    approx_errors.append(1 - S)

plt.figure()
plt.plot(range(50), approx_errors, marker='o')
plt.xlabel(r"$\alpha_0$")
plt.ylabel("Approximation Error")
plt.title(r"Approximation Error vs $\alpha_0$")
plt.grid(True)
plt.savefig("/data.nst/ysinha/projects/ONN/figs/fig3e.png")