import torch
from torch import nn
import lightning.pytorch as pl
from optimizers import LM


class LitMLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr, nesterov=False,
                 momentum=0, opt='SGD', hiddens=(50, 40, 30, 20), ax='softplus',
                 reg=None, lmbda=0.001, var_noise=1, where2noise=None):
        super().__init__()

        self.lr = lr
        self.m = momentum
        self.n = nesterov
        self.opt = opt
        self.reg = reg
        self.lmbda = lmbda
        self.variance_noise = var_noise
        self.where2noise = where2noise

        if ax == 'tanh':
            func = nn.Tanh()
        elif ax == 'logistic':
            func = nn.Sigmoid()
        elif ax == 'linear':
            func = nn.ReLU()
        elif ax == 'softsign':
            func = nn.Softsign()
        elif ax == 'softplus':
            func = nn.Softplus()
        else:
            raise NotImplementedError

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hiddens[0]),
            nn.ReLU(),
            nn.Linear(hiddens[0], hiddens[1]),
            func,
            nn.Linear(hiddens[1], hiddens[2]),
            func,
            nn.Linear(hiddens[2], hiddens[3]),
            func,
            nn.Linear(hiddens[3], output_dim)
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
        elif self.opt == 'Adagrad':
            optimizer = torch.optim.Adagrad(self.parameters(),
                                            lr=self.lr)
        elif self.opt == 'RMS':
            optimizer = torch.optim.RMSprop(self.parameters(),
                                            lr=self.lr, alpha=self.m)
        elif self.opt == 'AdaDelta':
            optimizer = torch.optim.Adadelta(self.parameters(),
                                             lr=self.lr, rho=self.m)
        elif self.opt == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr, betas=(self.m, self.m))
        elif self.opt == 'Rprop':
            optimizer = torch.optim.Rprop(self.parameters(),
                                          lr=self.lr)
        elif self.opt == 'BFGS':
            optimizer = torch.optim.LBFGS(self.parameters(),
                                          lr=self.lr)
        elif self.opt == 'LM':
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = LM(params=trainable_params, lr=self.lr)
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.where2noise == 'input' or self.where2noise == 'all':
            n = torch.randn(x.shape, device=self.device)
            x = x + (self.variance_noise ** 0.5) * n

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)

        if self.reg == 'l1':
            l1_lambda = self.lmbda
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = loss + l1_lambda * l1_norm
            self.log('l1_norm', torch.norm(loss), on_step=True, on_epoch=True)
        if self.reg == 'l2':
            l2_lambda = self.lmbda
            l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
            loss = loss + l2_lambda * l2_norm
            self.log('l2_norm', torch.norm(loss), on_step=True, on_epoch=True)

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
        if self.where2noise == 'grad' or self.where2noise == 'all':
            n = torch.randn(loss.shape, device=self.device)
            loss = loss + (self.variance_noise ** 0.5) * n

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
