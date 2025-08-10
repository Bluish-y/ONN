import onn
import time
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.normal import Normal
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm, binned_statistic

onn.EPOCHS = 20
start = time.time()
train_loader, val_loader, test_loader = onn.load_mnist_data(normalize=True)

onn_exact = onn.OpticalONN(approx=False, alpha0=10.0).to(onn.device)
onn_exact.init_sa_weights(onn_exact)
onn_exact_opt = torch.optim.Adam(onn_exact.parameters(), lr=onn.LR)

start_oa = time.time()
for epoch in range(onn.EPOCHS):
    onn.train_onn(onn_exact, train_loader, onn_exact_opt)
acc_onn_exact = onn.test_model(onn_exact, test_loader, is_onn=True)
end_oa = time.time()
print(f"(ONN with exact grad training time: {end_oa - start_oa:.2f} seconds")

def sa_exact_grad_numpy(z):
    return (1 + (onn_exact.ALPHA_0 * z**2) / (1 + z**2)**2) * np.exp(-onn_exact.ALPHA_0 / (2 * (1 + z**2)))

def generate_symmetric_function(num_points=50):
    """Generate smooth symmetric random function using shape-preserving interpolation."""
    x_half = np.linspace(0, 30, num_points)
    y_half = np.random.rand(num_points)

    # Symmetric extension
    x_full = np.concatenate([-x_half[:0:-1], x_half])
    y_full = np.concatenate([y_half[:0:-1], y_half])

    # Shape-preserving cubic Hermite interpolator
    interp = PchipInterpolator(x_full, y_full)
    return interp


class InterpolatedFunctionTorch:
    def __init__(self, interp_fn, z_min=-20, z_max=20, num_points=2000, device='cpu'):
        # Sample the interpolated function onto a dense grid
        self.device = device
        self.z_grid = torch.linspace(z_min, z_max, num_points, device=device)
        z_np = self.z_grid.cpu().numpy()
        self.f_grid = torch.tensor(interp_fn(z_np), dtype=torch.float32, device=device)
        self.z_min = z_min
        self.z_max = z_max
        self.num_points = num_points
        self.dx = (z_max - z_min) / (num_points - 1)

    def __call__(self, z_query):
        """Evaluate the interpolated function at arbitrary z_query (Tensor)."""
        z_query = z_query.clamp(min=self.z_min, max=self.z_max)  # clip to bounds

        # Normalize and get fractional indices
        idx_float = (z_query - self.z_min) / self.dx
        idx_lower = torch.floor(idx_float).long()
        idx_upper = idx_lower + 1

        # Prevent out-of-bounds indexing
        idx_upper = idx_upper.clamp(max=self.num_points - 1)
        idx_lower = idx_lower.clamp(min=0)

        frac = (z_query - (self.z_min + idx_lower * self.dx)) / self.dx
        val = (1 - frac) * self.f_grid[idx_lower] + frac * self.f_grid[idx_upper]
        return val


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

similarities, accs = [], []

for _ in range(40):
    f = generate_symmetric_function()
    gp = sa_exact_grad_numpy
    S = compute_similarity_S(f, gp)

    f_torch = InterpolatedFunctionTorch(f, z_min=-30, z_max=30, device='cpu')


    random_onn = onn.RandomONN(alpha0=10, grad=f_torch).to(onn.device)
    random_onn.init_sa_weights(random_onn)
    random_onn_opt = torch.optim.Adam(random_onn.parameters(), lr=onn.LR)

    start_ro = time.time()
    for epoch in range(onn.EPOCHS):
        onn.train_onn(random_onn, train_loader, random_onn_opt)
    acc_random = onn.test_model(random_onn, test_loader, is_onn=True)
    end_ro = time.time()
    print(f"(Random ONN with custom grad training time: {end_ro - start_ro:.2f} seconds")

    similarities.append(S)
    accs.append(acc_random)


approx_errors = 1 - np.array(similarities)
np.savez("/data.nst/ysinha/projects/ONN/output/fig3d.npz",
         approx_errors=approx_errors,
         accs=accs,
         acc_onn_exact=acc_onn_exact)

# --- a) Bin the similarities and compute avg accuracies and error bars ---
num_bins = 13
bins = np.linspace(0, 0.5, num_bins + 1)

# Mean accuracy per bin
bin_means, bin_edges, _ = binned_statistic(approx_errors, accs, statistic='mean', bins=bins)

# Standard error per bin
bin_stds, _, _ = binned_statistic(approx_errors, accs, statistic='std', bins=bins)
bin_counts, _, _ = binned_statistic(approx_errors, accs, statistic='count', bins=bins)
bin_sems = bin_stds / np.sqrt(bin_counts)

# Bin centers for plotting
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# --- b) Plot bar graph with error bars ---
plt.figure(figsize=(8, 5))
plt.bar(bin_centers, bin_means, width=0.08, yerr=bin_sems, capsize=4, align='center', alpha=0.7, color='cornflowerblue', edgecolor='k')
plt.axhline(y=acc_onn_exact, color='g', linestyle='--', label='SA (alpha0=10, exact)')
plt.xlabel('Approximation Error')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy vs Approximation Error')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("/data.nst/ysinha/projects/ONN/figs/fig3d.png")

