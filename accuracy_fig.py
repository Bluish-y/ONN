import onn
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from torch.distributions.normal import Normal

# --- Part A: Run training with alpha0 = 30 and plot accuracy vs epochs ---
start = time.time()
train_loader, val_loader, test_loader = onn.load_data()

# # Store original alpha
# onn.EPOCHS = 30  # Set epochs for training
# onn.ALPHA_0 = 30  # Set alpha for training

# # Capture accuracy per epoch from onn.py
# accs_ann, accs_onn_exact, accs_onn_approx, density_logs_exact, density_logs_approx = onn.run_all(train_loader, val_loader, test_loader, return_density=True)

# # # Save plot (a)
# # plt.figure()
# # plt.plot(accs_ann, label="ReLU")
# # plt.plot(accs_onn_exact, label=f"SA(α={onn.ALPHA_0}) Exact")
# # plt.plot(accs_onn_approx, label=f"SA(α={onn.ALPHA_0}) Approx")
# # plt.xlabel("Epoch")
# # plt.ylabel("Validation Accuracy")
# # plt.title("Accuracy vs Epochs")
# # plt.legend()
# # plt.grid(True)
# # plt.savefig("/data.nst/ysinha/projects/ONN/figs/accuracy_vs_epochs_unnormalized.png")

# # #save accs in npz in output directory
# # np.savez("/data.nst/ysinha/projects/ONN/output/accuracy_vs_epochs_unnormalized.npz",
# #          epochs = np.arange(1, onn.EPOCHS + 1),
# #          accs_ann=accs_ann,
# #          accs_onn_exact=accs_onn_exact,
# #          accs_onn_approx=accs_onn_approx)

#  # Convert to numpy for plotting
# density_logs_exact_np = torch.stack(density_logs_exact).numpy()
# density_logs_approx_np = torch.stack(density_logs_approx).numpy()
# # x_vals = torch.linspace(-20, 20, steps=100).numpy()

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(density_logs_exact_np, aspect='auto', origin='lower', extent=[-20, 20, 0, onn.EPOCHS])
# plt.colorbar(label="Neuron Count")
# plt.xlabel("Neuron Value")
# plt.ylabel("Epoch")
# plt.title("Neuron Activation Density (SA Exact)")

# plt.subplot(1, 2, 2)
# plt.imshow(density_logs_approx_np, aspect='auto', origin='lower', extent=[-20, 20, 0, onn.EPOCHS])
# plt.colorbar(label="Neuron Count")
# plt.xlabel("Neuron Value")
# plt.ylabel("Epoch")
# plt.title("Neuron Activation Density (SA Approx)")

# plt.tight_layout()
# plt.savefig("/data.nst/ysinha/projects/ONN/figs/neuron_activation_density.png")

# # Save density logs in npz in output directory
# np.savez("/data.nst/ysinha/projects/ONN/output/neuron_activation_density.npz",
#          epochs=np.arange(1, onn.EPOCHS + 1),
#             density_logs_exact=density_logs_exact_np,
#             density_logs_approx=density_logs_approx_np)


# --- Part B: Test accuracy vs alpha0 ---
# alphas = [0.1, 0.6, 1, 10, 30, 100]
alphas = [0.1, 100]
test_acc_exact, test_acc_approx = [], []
alphas1 = []

for alpha in alphas:
    onn.set_hyperparameters(epochs = 1, alpha0=alpha)  # Set epochs for training
    model_e = onn.OpticalONN().to(onn.device)
    model_a = onn.OpticalONN().to(onn.device)
    onn.init_sa_weights(model_e, alpha)
    onn.init_sa_weights(model_a, alpha)

    opt_e = torch.optim.Adam(model_e.parameters(), lr=onn.LR)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=onn.LR)

    for epoch in range(onn.EPOCHS):
        onn.train_onn(model_e, train_loader, opt_e, grad_fn=onn.sa_exact_grad)
        onn.train_onn(model_a, train_loader, opt_a, grad_fn=onn.sa_approx_grad)

    acc_e = onn.test_model(model_e, test_loader, is_onn=True)
    acc_a = onn.test_model(model_a, test_loader, is_onn=True)

    test_acc_exact.append(acc_e)
    test_acc_approx.append(acc_a)
    alphas1.append(onn.ALPHA_0)

# Save plot (b)
print(alphas1)
plt.figure()
plt.plot(alphas, test_acc_exact, marker='o', label="SA Exact")
plt.plot(alphas, test_acc_approx, marker='x', label="SA Approx")
plt.axhline(y=onn.test_model(onn.BaseANN().to(onn.device), test_loader), color='gray', linestyle='--', label="ReLU (Imagined)")
plt.xlabel("α₀")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Optical Depth α₀")
plt.legend()
plt.grid(True)
# plt.savefig("/data.nst/ysinha/projects/ONN/figs/test_accuracy_vs_alpha.png")

# Save accuracies in npz in output directory
# np.savez("/data.nst/ysinha/projects/ONN/output/test_accuracy_vs_alpha.npz",
#          alphas=alphas,
#             test_acc_exact=test_acc_exact,
#             test_acc_approx=test_acc_approx)

# # --- Part C: Approximation error S vs accuracy ---
# def compute_S(f, gprime, p):
#     num, _ = quad(lambda z: f(z) * gprime(z) * p(z), -5, 5)
#     denom1, _ = quad(lambda z: f(z) * p(z), -5, 5)
#     denom2, _ = quad(lambda z: gprime(z) * p(z), -5, 5)
#     return (num**2) / (denom1 * denom2 + 1e-8)

# def gprime_sa(z, alpha0=30):
#     return (1 + (alpha0 * z**2) / (1 + z**2)**2) * np.exp(-alpha0 / (2 * (1 + z**2)))

# p = lambda z: Normal(0, 0.6).log_prob(torch.tensor(z)).exp().item()

# Ss = []
# accs = []

# for i in range(10):
#     coeffs = np.random.randn(5)
#     f = lambda z: np.polyval(coeffs, z) - np.polyval(coeffs, -z)  # Symmetric
#     S = compute_S(f, lambda z: gprime_sa(z), p)
#     Ss.append(S)

#     # Accuracy for approx with this function isn't directly computable,
#     # so you might want to use previously computed test_acc_approx here.
#     accs.append(test_acc_approx[-1])  # Just illustrative, you can match more meaningfully

# # Save plot (c)
# plt.figure()
# plt.scatter(Ss, accs)
# plt.xlabel("Approximation Score S")
# plt.ylabel("Test Accuracy")
# plt.title("Accuracy vs Approximation Error")
# plt.grid(True)
# plt.savefig("accuracy_vs_approx_error.png")

end = time.time()
print(f"Total time taken: {end - start:.2f} seconds")
