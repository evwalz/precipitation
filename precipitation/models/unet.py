import torch
from torch import nn
from torch.nn import functional as F
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
import isodisreg
from precipitation.models.unet_architectures import UNet
from isodisreg import idr
from torchvision import transforms
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.cli import LightningCLI
from torch.optim.lr_scheduler import CosineAnnealingLR
from precipitation.data.data_module import PrecipitationDataModule, TargetLogScaler


class PrecipitationUNet(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        grid_lat: int = 19,
        grid_lon: int = 61,
        n_features: int = 21,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.unet = UNet(in_channels=n_features, dropout=dropout)
        self.mae_metric = MeanAbsoluteError()
        self.rmse_metric = MeanSquaredError(squared=False)
        self.target_scaler = TargetLogScaler()
        
        self.transform = transforms.RandomApply(
            torch.nn.ModuleList([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ]), p=0.5
            )

        self.save_hyperparameters()
        
    def setup(self, stage: str | None = None) -> None:
        #lsm = xr.open_dataset(self.trainer.datamodule.data_dir / "lsm.nc")
        #threshold = 0.5
        #lsm = lsm.drop("time")
        #lsm_values =  lsm.lsm.values[0,:,:]
        #self.mask = torch.tensor(np.where(lsm_values > threshold, 1, 0))
        lsm = np.loadtxt(self.trainer.datamodule.data_dir / "lsm.txt")
        self.mask = torch.tensor(lsm)
        if self.trainer and isinstance(self.trainer.accelerator, CUDAAccelerator):
            self.mask = self.mask.to(torch.device("cuda"))
        if self.trainer and isinstance(self.trainer.accelerator, MPSAccelerator):
            self.mask = self.mask.to(torch.device("mps"))

    def training_step(
        self, batch: tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int
    ):
        
        x, y = batch
        x = self.transform(x)
        
        b, c, h, w = x.size()

        y_hat = self.unet(x)
        y_hat = y_hat.view(b, self.hparams.grid_lat, self.hparams.grid_lon)  # type: ignore

        y_scaled = self.target_scaler.transform(y)
        loss = F.mse_loss(y_hat, y_scaled)  # type: ignore
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        
        x, y = batch
        b, c, h, w = x.size()
        
        y_hat = self.unet(x)
        y_hat = y_hat.view(b, self.hparams.grid_lat, self.hparams.grid_lon)  # type: ignore

        y_scaled = self.target_scaler.transform(y)
        loss = F.mse_loss(y_hat, y_scaled)  # type: ignore
        y_hat_rescaled = self.target_scaler.inverse_transform(y_hat)
        
        if dataloader_idx == 0:
            mae = self.mae_metric(y_hat_rescaled, y)
            masked_mae = self.mae_metric(y_hat_rescaled*self.mask, y*self.mask)
            rmse = self.rmse_metric(y_hat_rescaled, y)
            
            self.log_dict({"val_loss": loss, "val_mae": mae, "val_masked_mae": masked_mae, "val_rmse": rmse}, on_step=False, on_epoch=True)
                    
        return y_hat_rescaled, y
    
    def validation_epoch_end(self, outputs) -> None:
        # for param_group in self.trainer.optimizers[0].param_groups:
        #     print(param_group["lr"])
        
        if self.current_epoch == self.trainer.max_epochs -1:
            
            val_preds = torch.vstack([tup[0] for tup in outputs[0]]).cpu().numpy()
            val_target = torch.vstack([tup[1] for tup in outputs[0]]).cpu().numpy()
            
            train_preds = torch.vstack([tup[0] for tup in outputs[1]]).cpu().numpy()
            train_target = torch.vstack([tup[1] for tup in outputs[1]]).cpu().numpy()
            
            # crps_whole = np.zeros((self.hparams.grid_lat, self.hparams.grid_lon))
            crps_list = []
            
            for i in tqdm(range(self.hparams.grid_lat)):
                for j in tqdm(range(self.hparams.grid_lon)):
                    if self.mask[i,j]:
                        idr_per_grid = idr(y=train_target[:,i,j], X=pd.DataFrame(train_preds[:,i,j]))
                        val_dist_pred = idr_per_grid.predict(pd.DataFrame(val_preds[:,i,j]))
                        crps_per_grid = np.mean(val_dist_pred.crps(val_target[:,i,j])) # if seasonal validation, watch out here
                        # crps_whole[i,j] = crps_per_grid
                        crps_list.append(crps_per_grid)
                    
            # masked_crps = crps_whole * self.mask.cpu().numpy()
            # mean_masked_crps = np.mean(masked_crps)
            mean_masked_crps = np.mean(crps_list)
            
            self.log_dict({"val_masked_crps": mean_masked_crps})


    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_initial.learning_rate, weight_decay=self.hparams.weight_decay)  # type: ignore
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0.000001)
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == "__main__":
    # import wandb

    # wandb.finish()
    # wandb_logger = WandbLogger(project="dummy-MLP", log_model="all", dir="logs")
    tb_logger = TensorBoardLogger("latest_logs", name="dummy-UNet")
    
    pl.seed_everything(123)
    cli = LightningCLI(
        PrecipitationUNet,
        datamodule_class=PrecipitationDataModule,
        trainer_defaults={
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": 5,
            "logger": tb_logger,
        },
        save_config_overwrite=True,
    )
