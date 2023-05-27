import torch
from torch import nn
import lightning.pytorch as pl
from optimizers import LM


class LitMLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr, nesterov, momentum, opt):
        super().__init__()

        self.lr = lr
        self.m = momentum
        self.n = nesterov
        self.opt = opt
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
        self.loss = nn.CrossEntropyLoss()

        self.weights_stat = {
            'layers.2.weight': [],
            'layers.3.weight': []
        }

        for name, p in self.named_parameters():
            if name == 'layers.2.weight':
                self.weights_stat[name].append(p.detach().numpy().mean())
            if name == 'layers.3.weight':
                self.weights_stat[name].append(p.detach().numpy().mean())

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        if self.opt == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        momentum=self.m,
                                        nesterov=self.n)
        if self.opt == 'Adagrad':
            optimizer = torch.optim.Adagrad(self.parameters(),
                                            lr=self.lr)
        if self.opt == 'RMS':
            optimizer = torch.optim.RMSprop(self.parameters(),
                                            lr=self.lr, alpha=self.m)
        if self.opt == 'AdaDelta':
            optimizer = torch.optim.Adadelta(self.parameters(),
                                             lr=self.lr, rho=self.m)
        if self.opt == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr, betas=(self.m[0], self.m[1]))
        if self.opt == 'Rprop':
            optimizer = torch.optim.Rprop(self.parameters(),
                                          lr=self.lr)
        if self.opt == 'BFGS':
            optimizer = torch.optim.LBFGS(self.parameters(),
                                          lr=self.lr)
        if self.opt == 'LM':
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = LM(params=trainable_params, lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)

        with torch.no_grad():
            for name, p in self.named_parameters():
                if name == 'layers.2.weight':
                    last = self.weights_stat[name][-1]
                    self.weights_stat[name].append(abs(last - p.mean().item()))
                    self.log('layer2_change_mean', self.weights_stat[name][-1], on_epoch=True, on_step=True)
                if name == 'layers.3.weight':
                    last = self.weights_stat[name][-1]
                    self.weights_stat[name].append(abs(last - p.mean().item()))
                    self.log('layer3_change_mean', self.weights_stat[name][-1], on_epoch=True, on_step=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)