import torch
from torch import nn
from torch.nn import functional as F
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
import isodisreg
from isodisreg import idr
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.cli import LightningCLI
from precipitation.data.data_module import PrecipitationDataModule, TargetLogScaler


class UNet(nn.Module):
    def __init__(self, in_channels=14, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, stride=(1,3), instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, stride=(1,1), instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**2, initial_filter_size*2**2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**2, initial_filter_size*2**2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2**2, 2, stride=2, output_padding=(0,1)),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size*2**3, initial_filter_size*2**3)
        self.expand_4_2 = self.expand(initial_filter_size*2**3, initial_filter_size*2**3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size*2**3, initial_filter_size*2**2, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(initial_filter_size*2**3, initial_filter_size*2**2)
        self.expand_3_2 = self.expand(initial_filter_size*2**2, initial_filter_size*2**2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2, 2, stride=2, output_padding=(1,0))

        self.expand_2_1 = self.expand(initial_filter_size*2**2, initial_filter_size*2)
        self.expand_2_2 = self.expand(initial_filter_size*2, initial_filter_size*2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size*2, initial_filter_size, 2, stride=(2, 2), output_padding=(1,1))

        self.expand_1_1 = self.expand(initial_filter_size*2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        
        self.upscale1 = nn.Sequential(
            nn.ConvTranspose2d(initial_filter_size, initial_filter_size, 3, stride=(1,3), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        
        # Output layer for target
        self.final = nn.Conv2d(initial_filter_size, 1, kernel_size=1)


    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, stride=1, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
            )
        return layer

    def forward(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        center = self.center(pool) # actually there's an upsampling atrous conv here, watch out

        concat = torch.cat([center, contr_3*concat_weight], 1)
        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        concat = torch.cat([upscale, contr_2*concat_weight], 1)
        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        concat = torch.cat([upscale, contr_1*concat_weight], 1)
        expand = self.expand_1_2(self.expand_1_1(concat))
        upscale = self.upscale1(expand)

        output = self.final(upscale)

        return output


class PrecipitationUNet(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        grid_lat: int = 19,
        grid_lon: int = 61,
        n_features: int = 14,
        **kwargs
    ):
        super().__init__()
        self.unet = UNet()
        self.mae_metric = MeanAbsoluteError()
        self.rmse_metric = MeanSquaredError(squared=False)
        self.target_scaler = TargetLogScaler()

        self.save_hyperparameters()
        
    def setup(self, stage: str | None = None) -> None:
        lsm = xr.open_dataset(self.trainer.datamodule.data_dir / "lsm.nc")
        threshold = 0.5
        lsm = lsm.drop("time")
        lsm_values =  lsm.lsm.values[0,:,:]
        self.mask = torch.tensor(np.where(lsm_values > threshold, 1, 0))
        if self.trainer and isinstance(self.trainer.accelerator, CUDAAccelerator):
            self.mask = self.mask.to(torch.device("cuda"))
        if self.trainer and isinstance(self.trainer.accelerator, MPSAccelerator):
            self.mask = self.mask.to(torch.device("mps"))

    def training_step(
        self, batch: tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int
    ):

        x, y = batch
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
            
            self.log_dict({"masked_crps": mean_masked_crps})


    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_initial.learning_rate, weight_decay=self.hparams.weight_decay)  # type: ignore
        return optimizer


if __name__ == "__main__":
    # import wandb

    # wandb.finish()
    # wandb_logger = WandbLogger(project="dummy-MLP", log_model="all", dir="logs")
    tb_logger = TensorBoardLogger("new_logs", name="dummy-UNet")
    
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
