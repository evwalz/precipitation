import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pl_bolts.datasets import DummyDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate=0.01, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28)
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        # --------------------------

    def train_dataloader(self):
        train = DummyDataset((1, 28, 28), (1,))
        return DataLoader(train, batch_size=64)

    def val_dataloader(self):
        val = DummyDataset((1, 28, 28), (1,))
        return DataLoader(val, batch_size=64)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="dummy-AutoEncoder", log_model="all", dir="logs")
    cli = LightningCLI(
        LitAutoEncoder,
        auto_registry=True,
        trainer_defaults={"gpus": 0, "max_epochs": 5, "logger": wandb_logger},
        save_config_overwrite=True,
    )
