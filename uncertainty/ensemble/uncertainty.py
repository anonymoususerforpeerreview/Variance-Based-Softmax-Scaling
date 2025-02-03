import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.pyplot import scatter
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
import lightning as L

sns.set_theme(rc={'figure.figsize': (9, 7)})

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

h = {}
h['dataset'] = 'TOYREGRESSION'
h['redundancy_method_params'] = {'num_points': 300}
h['ensemble_size'] = 5
h['use_wandb'] = False
h['learning_rate'] = 0.001
h['epochs'] = 10
h['batch_size'] = 16
h['train'] = True
h['fast_dev_run'] = False
h['limit_train_batches'] = 1.0
h['limit_val_batches'] = 1.0
h['limit_test_batches'] = 1.0
h['overfit_batches'] = 0


class SimpleNet(L.LightningModule):
    def __init__(self):
        super(SimpleNet, self).__init__()
        hidden_size = 64

        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, 1)
        self.var = nn.Linear(hidden_size, 1)

        # negative log likelihood
        self.NLL = torch.nn.GaussianNLLLoss(eps=1e-02)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.mu(h)
        var = torch.exp(self.var(h))
        return mu, var

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, var = self.forward(x)
        loss = self.NLL(mu, y, var)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, var = self.forward(x)
        loss = self.NLL(mu, y, var)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mu, var = self.forward(x)
        loss = self.NLL(mu, y, var)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=h['learning_rate'])

    def save(self) -> None:
        os.makedirs(h['checkpoint_path'], exist_ok=True)
        torch.save(self.state_dict(), os.path.join(h['checkpoint_path'], 'model.pth'))

    def load(self) -> 'SimpleNet':
        self.load_state_dict(torch.load(os.path.join(h['checkpoint_path'], 'model.pth')))
        self.eval()
        return self


class EnsembleNet(L.LightningModule):
    def __init__(self, num_models: int):
        super(EnsembleNet, self).__init__()
        self.models = nn.ModuleList([SimpleNet() for _ in range(num_models)])
        self.num_models = num_models
        self.automatic_optimization = False

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        mus, vars = [], []
        for model in self.models:
            mu, var = model(x)
            mus.append(mu)
            vars.append(var)
        mus = torch.stack(mus, dim=0)
        vars = torch.stack(vars, dim=0)
        return mus.mean(dim=0), vars.mean(dim=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        losses = []
        opt_list = self.optimizers()
        for i, model in enumerate(self.models):
            mu, var = model(x)
            loss = model.NLL(mu, y, var)
            losses.append(loss)
            opt = opt_list[i]
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        avg_loss = torch.stack(losses).mean()
        self.log('train_loss', avg_loss)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        losses = []
        for model in self.models:
            mu, var = model(x)
            loss = model.NLL(mu, y, var)
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
        self.log('val_loss', avg_loss)
        return avg_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        losses = []
        for model in self.models:
            mu, var = model(x)
            loss = model.NLL(mu, y, var)
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
        self.log('test_loss', avg_loss)
        return avg_loss

    def configure_optimizers(self):
        optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in self.models]
        return optimizers

    def save(self) -> None:
        os.makedirs(h['checkpoint_path'], exist_ok=True)
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(h['checkpoint_path'], f'model_{i}.pth'))

    def load(self) -> 'EnsembleNet':
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(h['checkpoint_path'], f'model_{i}.pth')))
            model.eval()
        return self


def sample_dataset(start, end, n):
    x = np.linspace(start, end, n)
    sample_mean = [math.sin(i / 2) for i in x]
    sample_var = [((abs(start) + abs(end)) / 2 - abs(i)) / 16 for i in x]
    y = stats.norm(sample_mean, sample_var).rvs()
    return x, y


start, end, n = -7, 7, 300
x_train, y_train = sample_dataset(start, end, n)
x_test, y_test = sample_dataset(-10, 10, 200)

tensor_x = torch.Tensor(x_train).unsqueeze(1)
tensor_y = torch.Tensor(y_train).unsqueeze(1)
train_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

tensor_x_test = torch.Tensor(x_test).unsqueeze(1).to(device)
tensor_y_test = torch.Tensor(y_test).unsqueeze(1).to(device)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


class CustomCallbacks(L.Callback):
    pass


model = EnsembleNet(num_models=h['ensemble_size'])
if h['train']:
    trainer = L.Trainer(
        max_epochs=h['epochs'],
        limit_train_batches=h['limit_train_batches'],
        limit_val_batches=h['limit_val_batches'],
        limit_test_batches=h['limit_test_batches'],
        fast_dev_run=h['fast_dev_run'],
        overfit_batches=h['overfit_batches'],
        log_every_n_steps=10)

    try:
        trainer.fit(model, train_loader)
    except KeyboardInterrupt:
        pass
    finally:
        pass


def make_plot_de(model: EnsembleNet, start, end):
    model.eval()
    model.to(device)
    x_test, y_test = tensor_x_test, tensor_y_test
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    mus, vars = model(x_test)
    mus = mus.cpu()
    vars = vars.cpu()

    means = mus.detach().numpy()
    stds = vars.detach().sqrt().numpy()

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

    x_train, y_train = tensor_x, tensor_y
    scatter(x_train, y_train, c="green", marker="*", alpha=0.1)

    plt.show()


make_plot_de(model, -7, 7)