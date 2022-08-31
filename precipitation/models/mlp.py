import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.cli import LightningCLI
from precipitation.data.data_module import PrecipitationDataModule


class PrecipitationMLP(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.01, grid_lat: int = 19, grid_lon: int = 61, n_features: int = 10, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(grid_lat * grid_lon * n_features, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, grid_lat * grid_lon)
        )
        self.mae_metric = MeanAbsoluteError()
        
        self.save_hyperparameters()

    def training_step(self, batch: tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int):
        
        x, y = batch
        b, c, h, w = x.size()
        x = x.view(b, -1)
        
        y_hat = self.mlp(x)
        y_hat = y_hat.view(b, self.hparams.grid_lat, self.hparams.grid_lon)  # type: ignore
        
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int) -> None:
        
        x, y = batch
        b, c, h, w = x.size()
        x = x.view(b, -1)
        
        y_hat = self.mlp(x)
        y_hat = y_hat.view(b, self.hparams.grid_lat, self.hparams.grid_lon)  # type: ignore
        
        loss = F.mse_loss(y_hat, y)
        mae = self.mae_metric(y_hat, y)
        self.log_dict({"val_loss": loss, "val_mae": mae})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_initial.learning_rate)  # type: ignore
        return optimizer


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="dummy-MLP", log_model="all", dir="logs")
    pl.seed_everything(123)
    cli = LightningCLI(
        PrecipitationMLP,
        datamodule_class=PrecipitationDataModule,
        trainer_defaults={"accelerator": "auto", "devices": 1, "max_epochs": 5, "logger": wandb_logger},
        save_config_overwrite=True,
    )
