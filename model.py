import torch
from torch import nn
import torch.nn.functional as f
import lightning.pytorch as pl


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, 20)
        self.hidden1 = nn.Linear(20, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.hidden3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, output_dim)

    def forward(self, x):
        x = f.sigmoid(self.input(x))
        x = f.tanh(self.hidden1(x))
        x = f.tanh(self.hidden2(x))
        x = f.tanh(self.hidden3(x))

        return self.output(x)


class LitMLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr):
        super().__init__()
        self.lr = lr
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, output_dim)
        )
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.layers(x).max(axis=1)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
