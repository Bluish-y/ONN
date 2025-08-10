import onn
import time
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from torch.distributions.normal import Normal

# --- Part A: Run training with alpha0 = 30 and plot accuracy vs epochs ---
start = time.time()
train_loader, val_loader, test_loader = onn.load_mnist_data(normalize=False)

onn.EPOCHS = 50
onn_exact = onn.OpticalONN(approx = False, alpha0=30).to(onn.device)
onn_exact.init_sa_weights(onn_exact)
onn_approx = onn.OpticalONN(approx = True, alpha0=30).to(onn.device)
onn_approx.init_sa_weights(onn_approx)

onn_exact_opt = torch.optim.Adam(onn_exact.parameters(), lr=onn.LR)
onn_approx_opt = torch.optim.Adam(onn_approx.parameters(), lr=onn.LR)

density_logs_exact, density_logs_approx = [], []

start_oe = time.time()
for epoch in range(onn.EPOCHS):
    z_vals_exact = onn.train_onn(onn_exact, train_loader, onn_exact_opt)
    acc_onn1 = onn.validate_model(onn_exact, val_loader, is_onn=True)

    # calculate neuron density in bins from <=-20 to >=20
    density_exact = torch.histc(z_vals_exact, bins=100, min=-20, max=20)
    density_logs_exact.append(density_exact)
end_oe = time.time()
print(f"ONN with exact grad training time: {end_oe - start_oe:.2f} seconds")

start_oa = time.time()
for epoch in range(onn.EPOCHS):
    z_vals_approx = onn.train_onn(onn_approx, train_loader, onn_approx_opt)
    acc_onn2 = onn.validate_model(onn_approx, val_loader, is_onn=True)

    # calculate neuron density in bins from <=-20 to >=20
    density_approx = torch.histc(z_vals_approx, bins=100, min=-20, max=20)
    density_logs_approx.append(density_approx)
end_oa = time.time()
print(f"ONN with approx grad training time: {end_oa - start_oa:.2f} seconds")

# Save plot 
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(torch.stack(density_logs_exact).numpy(), aspect='auto', origin='upper', extent=[-20, 20, onn.EPOCHS-1, 0])
plt.colorbar(label="Neuron Count")
plt.xlabel("Neuron Value")
plt.ylabel("Epoch")
plt.title("Neuron Activation Density (SA Exact)")
plt.subplot(1, 2, 2)
plt.imshow(torch.stack(density_logs_approx).numpy(), aspect='auto', origin='upper', extent=[-20, 20, onn.EPOCHS-1, 0])
plt.colorbar(label="Neuron Count")
plt.xlabel("Neuron Value")
plt.ylabel("Epoch")
plt.title("Neuron Activation Density (SA Approx)")
plt.tight_layout()
plt.savefig("/data.nst/ysinha/projects/ONN/figs/fig3b_unnormalized.png")
# Save density logs in npz in output directory
np.savez("/data.nst/ysinha/projects/ONN/output/fig3b_unnormalized.npz",
         epochs=np.arange(0, onn.EPOCHS),
         density_logs_exact=torch.stack(density_logs_exact).numpy(),
         density_logs_approx=torch.stack(density_logs_approx).numpy())