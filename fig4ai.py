import onn
import time
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from torch.distributions.normal import Normal


scale = 0.1
normalize = False
# --- Part A: Run training with alpha0 = 30 and plot accuracy vs epochs ---
start = time.time()
train_loader, val_loader, test_loader = onn.load_mnist_data(normalize=normalize)
ann = onn.BaseANN().to(onn.device)
ann_opt = torch.optim.Adam(ann.parameters(), lr=onn.LR)

onn_exact = onn.OpticalONN(approx = False, alpha0=30, scale=scale).to(onn.device)
onn_exact.init_sa_weights(onn_exact)
onn_approx = onn.OpticalONN(approx = True, alpha0=30, scale=scale).to(onn.device)
onn_approx.init_sa_weights(onn_approx)

onn_exact_opt = torch.optim.Adam(onn_exact.parameters(), lr=onn.LR)
onn_approx_opt = torch.optim.Adam(onn_approx.parameters(), lr=onn.LR)

accs_ann, accs_onn_exact, accs_onn_approx = [], [], []

start_a = time.time()
for epoch in range(onn.EPOCHS):
    onn.train_ann(ann, nn.CrossEntropyLoss(), train_loader, ann_opt)
    acc_ann = onn.validate_model(ann, test_loader)
    accs_ann.append(acc_ann)
end_a = time.time()
print(f"ANN training time: {end_a - start_a:.2f} seconds")

start_oe = time.time()
for epoch in range(onn.EPOCHS):
    z_vals_exact = onn.train_onn(onn_exact, train_loader, onn_exact_opt)
    acc_onn1 = onn.validate_model(onn_exact, val_loader, is_onn=True)
    accs_onn_exact.append(acc_onn1)
end_oe = time.time()
print(f"ONN with exact grad training time: {end_oe - start_oe:.2f} seconds")

start_oa = time.time()
for epoch in range(onn.EPOCHS):
    z_vals_approx = onn.train_onn(onn_approx, train_loader, onn_approx_opt)
    acc_onn2 = onn.validate_model(onn_approx, val_loader, is_onn=True)
    accs_onn_approx.append(acc_onn2)
end_oa = time.time()
print(f"ONN with approx grad training time: {end_oa - start_oa:.2f} seconds")

# Save plot (a)
plt.figure()
plt.plot(range(onn.EPOCHS), accs_ann, marker='o', label="ANN")
plt.plot(range(onn.EPOCHS), accs_onn_exact, marker='x', label="ONN Exact")
plt.plot(range(onn.EPOCHS), accs_onn_approx, marker='s', label="ONN Approx")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("/data.nst/ysinha/projects/ONN/figs/fig4ai_rescaled.png")

# Save accuracies in npz in output directory
np.savez("/data.nst/ysinha/projects/ONN/output/fig4ai_rescaled.npz",
         epochs=range(onn.EPOCHS),
         accs_ann=accs_ann,
         accs_onn_exact=accs_onn_exact,
         accs_onn_approx=accs_onn_approx)


end = time.time()
print(f"Total time taken: {end - start:.2f} seconds")