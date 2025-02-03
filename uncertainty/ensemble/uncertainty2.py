import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import scatter
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

# seeds
torch.manual_seed(42)
np.random.seed(42)

sns.set_theme(rc={'figure.figsize': (9, 7)})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Generate Fake Regression Dataset with one Feature
def sample_dataset(start, end, n):
    x = np.linspace(start, end, n)
    sample_mean = [math.sin(i / 2) for i in x]
    sample_var = [((abs(start) + abs(end)) / 2 - abs(i)) / 16 for i in x]
    y = stats.norm(sample_mean, sample_var).rvs()
    return x, y


start, end, n = -7, 7, 300
x_train, y_train = sample_dataset(start, end, n)
x_test, y_test = sample_dataset(-10, 10, 200)

# Put data into Tensor Datasets
tensor_x = torch.Tensor(x_train).unsqueeze(1)
tensor_y = torch.Tensor(y_train).unsqueeze(1)
train_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

tensor_x_test = torch.Tensor(x_test).unsqueeze(1).to(device)
tensor_y_test = torch.Tensor(y_test).unsqueeze(1).to(device)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


class SimpleNet(pl.LightningModule):
    def __init__(self):
        super(SimpleNet, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, 1)
        self.var = nn.Linear(hidden_size, 1)
        self.criterion = torch.nn.GaussianNLLLoss(eps=1e-02)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.mu(h)
        var = torch.exp(self.var(h))
        return mu, var

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, var = self(x)
        loss = self.criterion(mu, y, var)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, var = self(x)
        loss = self.criterion(mu, y, var)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mu, var = self(x)
        loss = self.criterion(mu, y, var)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


class EnsembleNet(pl.LightningModule):
    def __init__(self, num_models):
        super(EnsembleNet, self).__init__()
        self.models = nn.ModuleList([SimpleNet() for _ in range(num_models)])
        self.num_models = num_models
        self.automatic_optimization = False

    def forward(self, x):
        mus, vars = [], []
        for model in self.models:
            mu, var = model(x)
            mus.append(mu)
            vars.append(var)
        mus = torch.stack(mus, dim=0)  # (num_models, batch_size, 1)
        vars = torch.stack(vars, dim=0)
        return mus.mean(dim=0), vars.mean(dim=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizers = self.optimizers()
        losses = []
        for i, model in enumerate(self.models):
            mu, var = model(x)
            loss = model.criterion(mu, y, var)
            optimizers[i].zero_grad()
            self.manual_backward(loss)
            optimizers[i].step()
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
        self.log('train_loss', avg_loss)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        losses = []
        for model in self.models:
            mu, var = model(x)
            loss = model.criterion(mu, y, var)
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
        self.log('val_loss', avg_loss)
        return avg_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        losses = []
        for model in self.models:
            mu, var = model(x)
            loss = model.criterion(mu, y, var)
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
        self.log('test_loss', avg_loss)
        return avg_loss

    def configure_optimizers(self):
        optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        return optimizers


def make_plot_de(model: EnsembleNet):
    model.eval()
    model.to(device)

    x_test, y_test = tensor_x_test, tensor_y_test
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # mus, vars = model(x_test) # (batch_size, 1), (batch_size, 1)
    # mus = mus.cpu()
    # vars = vars.cpu()
    #
    # means = mus.detach().numpy()
    # stds = vars.detach().sqrt().numpy()

    mus = []
    for m in model.models:
        mu, _ = m(x_test)  # (batch_size, 1)
        mus.append(mu)

    mus = torch.stack(mus, dim=0)  # (num_models, batch_size, 1)
    # vars = torch.stack(vars, dim=0)

    means = mus.mean(dim=0).detach().cpu().numpy()
    stds = mus.std(axis=0).detach().cpu().numpy()  # ** (1 / 2)

    dfs = []
    y_vals = [means, means + 2 * stds, means - 2 * stds]

    for i in range(3):
        data = {
            "x": list(x_test.cpu().squeeze().numpy()),
            "y": list(y_vals[i].squeeze())
        }
        temp = pd.DataFrame.from_dict(data)
        dfs.append(temp)

    df = pd.concat(dfs).reset_index()
    sns_plot = sns.lineplot(data=df, x="x", y="y")
    plt.axvline(x=start)
    plt.axvline(x=end)
    scatter(x_train, y_train, c="green", marker="*", alpha=0.1)
    plt.show()


# Training
num_models = 50
model = EnsembleNet(num_models=num_models)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, test_loader)
make_plot_de(model)
