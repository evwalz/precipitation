import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI
from precipitation.data.data_module import PrecipitationDataModule


class PrecipitationMLP(pl.LightningModule):
    def __init__(self, learning_rate: float=0.01, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(19 * 61, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.save_hyperparameters()

    def training_step(self, batch: tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int):
        
        x, y = batch
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        
        y_hat = self.mlp(x)
        loss = F.mse_loss(y_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int) -> None:
        
        x, y = batch
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        
        y_hat = self.mlp(x)
        loss = F.mse_loss(y_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_initial.learning_rate)  # type: ignore
        return optimizer


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="dummy-MLP", log_model="all", dir="logs")
    cli = LightningCLI(
        PrecipitationMLP,
        datamodule_class=PrecipitationDataModule,
        auto_registry=True,
        trainer_defaults={"accelerator": "auto", "devices": [0], "max_epochs": 5, "logger": wandb_logger},
        save_config_overwrite=True,
    )
