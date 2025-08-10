import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import time

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 50
# from torchvision.datasets import MNIST

# MNIST(root="/data.nst/ysinha/projects/ONN/data", train=True, download=True)
# MNIST(root="/data.nst/ysinha/projects/ONN/data", train=False, download=True)

class BaseANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.outputL = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.outputL(x)

class OpticalONN(nn.Module):
    def __init__(self, approx=True, alpha0 = 30, scale = 0):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128, bias=False)
        self.hidden2 = nn.Linear(128, 128, bias=False)
        self.outputL = nn.Linear(128, 10, bias=False)
        self.approx = approx
        self.ALPHA_0 = alpha0
        self.scale = scale

    def forward(self, x):
        x = x.view(-1, 784)
        z1 = self.hidden1(x)
        a1 = self.sa_forward(z1)
        z2 = self.hidden2(a1)
        a2 = self.sa_forward(z2)
        z3 = self.outputL(a2)
        return z3,z2,z1
    
    def sa_forward(self,z):
        return torch.exp(-self.ALPHA_0 / (2 * (1 + z ** 2))) * z

    def sa_exact_grad(self, z):
        return (1 + (self.ALPHA_0 * z**2) / (1 + z**2)**2) * torch.exp(-self.ALPHA_0 / (2 * (1 + z**2)))

    def sa_approx_grad(self, z):
        return (1+self.ALPHA_0*self.scale)*torch.exp(-self.ALPHA_0 / (2 * (1 + z ** 2)))
    
    def grad(self,z):
        if self.approx:
            return self.sa_approx_grad(z)
        else:
            return self.sa_exact_grad(z)

    def init_sa_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if self.ALPHA_0 <= 30:
                    nn.init.normal_(m.weight, mean=0.0, std=0.1)
                else:
                    shape = m.weight.shape
                    bimodal = torch.distributions.MixtureSameFamily(
                        mixture_distribution=torch.distributions.Categorical(torch.tensor([0.5, 0.5])),
                        component_distribution=torch.distributions.Normal(
                            torch.tensor([-0.15, 0.15]), torch.tensor([0.15, 0.15])
                        )
                    )
                    samples = bimodal.sample((shape[0], shape[1]))
                    m.weight.data.copy_(samples)

class RandomONN(OpticalONN):
    def __init__(self, grad, **args):
        super().__init__(**args)
        self.grad = grad

    
    
def rescale_input(x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        x_max = x_flat.max(dim=1, keepdim=True).values + 1e-8  # avoid division by zero
        x_normalized = x_flat / x_max
        return x_normalized.view_as(x)

# --- Training Loops ---
def train_ann(model, loss_fn, train_loader, optimizer):
    model.train()
    # correct = total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = rescale_input(x).view(-1, 784)  # [B, 784]
        optimizer.zero_grad()
        output = model(x)
        target = F.one_hot(y, num_classes=10).float().to(device)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # pred = output.argmax(dim=1)
        # correct += (pred == target).sum().item()
        # total += target.size(0)
    return

def train_onn(model, train_loader, optimizer):
    model.train()
    correct = total = 0
    all_z_values = []  # To store z values for each batch


    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = rescale_input(x).view(-1, 784)  # [B, 784]
        optimizer.zero_grad()

        # Forward pass
        z3, z2, z1 = model(x)  # Each z_i has shape [B, D_i]
        target = F.one_hot(y, num_classes=10).float().to(device)

        # loss = F.mse_loss(z3, target)

        # Compute output error
        delta3 = z3 - target  # [B, 10]

        # Gradients for each layer using manual backprop
        # NOTE: .weight is [out_features, in_features]

        # ---- Layer 3 ----
        # [B, 10] x [10, L2] => [B, L2]
        rho3 = torch.matmul(delta3, model.outputL.weight)  # [B, L2]
        delta2 = model.grad(z2) * rho3  # [B,L2]*[B, L2] = [B,L2]

        # ---- Layer 2 ----
        rho2 = torch.matmul(delta2, model.hidden2.weight)  # [B, L1]
        delta1 = model.grad(z1) * rho2  # [B, L1]

        # Activations
        a1 = model.sa_forward(z1)  # [B, L1]
        a2 = model.sa_forward(z2)  # [B, L2]

        # Gradients (outer products)
        grad_outputL = torch.bmm(delta3.unsqueeze(2), a2.unsqueeze(1)).mean(dim=0)  # [B, 10, 1]x[B, 1, L2] => [B,10, L2] (which is shape of outputL.weight, yay!)
        grad_hidden2 = torch.bmm(delta2.unsqueeze(2), a1.unsqueeze(1)).mean(dim=0)  # [B, L2, 1]x[B, 1, L1] => [B, L2, L1]
        grad_hidden1 = torch.bmm(delta1.unsqueeze(2), x.unsqueeze(1)).mean(dim=0)   # [B, L1, 1]x[B, 1, 784] => [B, L1, 784]

        # Assign gradients
        model.outputL.weight.grad = grad_outputL
        model.hidden2.weight.grad = grad_hidden2
        model.hidden1.weight.grad = grad_hidden1

        # Step optimizer
        optimizer.step()

        # # Track accuracy
        # pred = z3.argmax(dim=1)
        # y_actual = target.argmax(dim=1)
        # correct += (pred == y_actual).sum().item()
        # total += y_actual.size(0)
        # Store z values
        z_all = torch.cat([z1.flatten(), z2.flatten(), z3.flatten()])
        all_z_values.append(z_all.cpu())

    return torch.cat(all_z_values)



def test_model(model, loader, is_onn=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_onn:
                out, *_ = model(rescale_input(x))
            else:
                out = model(x)
            # Track accuracy
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def validate_model(model, loader, is_onn=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_onn:
                out, *_ = model(rescale_input(x))
            else:
                out = model(x)
            # Track accuracy
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def load_mnist_data(normalize=True):
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transform = transforms.ToTensor()
    full_train = datasets.MNIST('/ysinha/projects/ONN/data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('/ysinha/projects/ONN/data', train=False, download=False, transform=transform)

    val_size = len(test_dataset) // 2
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

    train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader
# --- Experiment Driver ---


