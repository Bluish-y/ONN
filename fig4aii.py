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
train_loader, val_loader, test_loader = onn.load_mnist_data(normalize=True)
alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0]

ann = onn.BaseANN().to(onn.device)
ann_opt = torch.optim.Adam(ann.parameters(), lr=onn.LR)
accs_ann = []
start_a = time.time()
for epoch in range(onn.EPOCHS):
    onn.train_ann(ann, nn.CrossEntropyLoss(), train_loader, ann_opt)
acc_ann = onn.test_model(ann, test_loader)

accs_onn_exact, accs_onn_approx = [], []
for alpha in alphas:
    onn_exact = onn.OpticalONN(approx = False, alpha0=alpha).to(onn.device)
    onn_exact.init_sa_weights(onn_exact)
    onn_approx = onn.OpticalONN(approx = True, alpha0=alpha).to(onn.device)
    onn_approx.init_sa_weights(onn_approx)

    onn_exact_opt = torch.optim.Adam(onn_exact.parameters(), lr=onn.LR)
    onn_approx_opt = torch.optim.Adam(onn_approx.parameters(), lr=onn.LR)

    start_oe = time.time()
    for epoch in range(onn.EPOCHS):
        z_vals_exact = onn.train_onn(onn_exact, train_loader, onn_exact_opt)
    acc_onn1 = onn.test_model(onn_exact, test_loader, is_onn=True)    
    accs_onn_exact.append(acc_onn1)
    end_oe = time.time()
    print(f"(Alpha = {onn_exact.ALPHA_0})(ONN with exact grad) training time: {end_oe - start_oe:.2f} seconds")

    start_oa = time.time()
    for epoch in range(onn.EPOCHS):
        z_vals_approx = onn.train_onn(onn_approx, train_loader, onn_approx_opt)
    acc_onn2 = onn.test_model(onn_approx, test_loader, is_onn=True)    
    accs_onn_approx.append(acc_onn2)
    end_oa = time.time()
    print(f"(Alpha = {onn_approx.ALPHA_0}) ONN with approx grad training time: {end_oa - start_oa:.2f} seconds")

# Save plot (a)
plt.figure()
plt.plot(alphas, accs_onn_exact, marker='x', label="ONN Exact")
plt.plot(alphas, accs_onn_approx, marker='s', label="ONN Approx")
plt.axhline(y=acc_ann, color='g', linestyle='--', label="RELU")
plt.xlabel("α")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Optical Depth α")
plt.legend()
plt.grid(True)
plt.savefig("/data.nst/ysinha/projects/ONN/figs/fig4aii.png")

# Save accuracies in npz in output directory
np.savez("/data.nst/ysinha/projects/ONN/output/fig4aii.npz",
         epochs=range(onn.EPOCHS),
         accs_ann=accs_ann,
         accs_onn_exact=accs_onn_exact,
         accs_onn_approx=accs_onn_approx)


end = time.time()
print(f"Total time taken: {end - start:.2f} seconds")