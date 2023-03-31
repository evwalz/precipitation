import argparse
import torch
from torch import nn
from torch.nn import functional as F
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
import multiprocessing
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
from precipitation.evaluation.calculate_crps import calculate_crps_idr


class PrecipitationUNet(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        grid_lat: int = 19,
        grid_lon: int = 61,
        n_features: int = 21,
        initial_filter_size: int = 64,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        # self.unet = UNet(in_channels=n_features, initial_filter_size=initial_filter_size, dropout=dropout)
        self.mae_metric = MeanAbsoluteError()
        self.rmse_metric = MeanSquaredError(squared=False)
        self.target_scaler = TargetLogScaler()
        
        self.transform = transforms.RandomApply(
            torch.nn.ModuleList([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ]), p=0.5
            )
        self.training_step_preds = []
        self.training_step_targets = []
        self.validation_step_preds = []
        self.validation_step_targets = []

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
            
        self.unet = UNet(in_channels=self.trainer.datamodule.n_features, initial_filter_size=self.hparams.initial_filter_size, dropout=self.hparams.dropout)
            
    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {"val_metrics/mae": 0, "val_metrics/masked_mae": 0, "val_metrics/rmse": 0, "val_metrics/masked_crps": 0})

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
        self.log("loss/train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        
        x, y = batch
        b, c, h, w = x.size()
        
        y_hat = self.unet(x)
        y_hat = y_hat.view(b, self.hparams.grid_lat, self.hparams.grid_lon)  # type: ignore

        y_scaled = self.target_scaler.transform(y)
        loss = F.mse_loss(y_hat, y_scaled)  # type: ignore
        y_hat_rescaled = self.target_scaler.inverse_transform(y_hat)
        
        if dataloader_idx == 0:
            self.validation_step_preds.append(y_hat_rescaled)
            self.validation_step_targets.append(y)
        else:
            self.training_step_preds.append(y_hat_rescaled)
            self.training_step_targets.append(y)
            
        if dataloader_idx == 0:
            mae = self.mae_metric(y_hat_rescaled, y)
            masked_mae = self.mae_metric(y_hat_rescaled*self.mask, y*self.mask)
            rmse = self.rmse_metric(y_hat_rescaled, y)
            
            # self.log_dict({"val_loss": loss, "val_mae": mae, "val_masked_mae": masked_mae, "val_rmse": rmse}, on_step=False, on_epoch=True)
            self.log_dict({"loss/val_loss": loss, "val_metrics/mae": mae, "val_metrics/masked_mae": masked_mae, "val_metrics/rmse": rmse}, on_step=False, on_epoch=True)
                    
        return loss
    
    # def validation_epoch_end(self, outputs) -> None:
    #     # for param_group in self.trainer.optimizers[0].param_groups:
    #     #     print(param_group["lr"])
        
    #     if self.current_epoch == self.trainer.max_epochs -1:
            
    #         val_preds = torch.vstack([tup[0] for tup in outputs[0]]).cpu().numpy()
    #         val_target = torch.vstack([tup[1] for tup in outputs[0]]).cpu().numpy()
            
    #         train_preds = torch.vstack([tup[0] for tup in outputs[1]]).cpu().numpy()
    #         train_target = torch.vstack([tup[1] for tup in outputs[1]]).cpu().numpy()
            
    #         # crps_whole = np.zeros((self.hparams.grid_lat, self.hparams.grid_lon))
    #         crps_list = []
            
    #         for i in tqdm(range(self.hparams.grid_lat)):
    #             for j in tqdm(range(self.hparams.grid_lon)):
    #                 if self.mask[i,j]:
    #                     idr_per_grid = idr(y=train_target[:,i,j], X=pd.DataFrame(train_preds[:,i,j]))
    #                     val_dist_pred = idr_per_grid.predict(pd.DataFrame(val_preds[:,i,j]))
    #                     crps_per_grid = np.mean(val_dist_pred.crps(val_target[:,i,j])) # if seasonal validation, watch out here
    #                     # crps_whole[i,j] = crps_per_grid
    #                     crps_list.append(crps_per_grid)
                    
    #         # masked_crps = crps_whole * self.mask.cpu().numpy()
    #         # mean_masked_crps = np.mean(masked_crps)
                        
    #         mean_masked_crps = np.mean(crps_list)

    #         self.log_dict({"val_metrics/masked_crps": mean_masked_crps})
            
    def on_validation_epoch_end(self) -> None:
        # for param_group in self.trainer.optimizers[0].param_groups:
        #     print(param_group["lr"])
        
        if self.current_epoch == self.trainer.max_epochs -1:
            
            mask = self.mask.cpu().numpy()
            
            val_preds = torch.vstack(self.validation_step_preds).cpu().numpy()
            val_targets = torch.vstack(self.validation_step_targets).cpu().numpy()
            
            train_preds = torch.vstack(self.training_step_preds).cpu().numpy()
            train_targets = torch.vstack(self.training_step_targets).cpu().numpy()
            
            # crps_whole = np.zeros((self.hparams.grid_lat, self.hparams.grid_lon))
            
            num_processes = multiprocessing.cpu_count()
            ctx = multiprocessing.get_context('spawn')
            pool = ctx.Pool(processes=num_processes)
            # pool = multiprocessing.Pool(processes=num_processes)
            
            args_list = [(val_preds, val_targets, train_preds, train_targets, mask, i, j) for i in range(self.hparams.grid_lat) for j in range(self.hparams.grid_lon)]
            
            pbar = tqdm(total=len(args_list))
            def update_pbar(*a):
                pbar.update()
                
            crps_list = []
            for arg in args_list:
                crps_per_grid = pool.apply_async(calculate_crps_idr, args=arg, callback=update_pbar)
                crps_list.append(crps_per_grid)
            
            pool.close()
            pool.join()
            
            crps_list = [crps.get() for crps in crps_list]
            # filter out None values (i.e. where mask is 0)
            crps_list = [crps for crps in crps_list if crps is not None]
            
            mean_masked_crps = np.mean(crps_list)
            print(f"Mean masked CRPS: {mean_masked_crps}")

            self.log_dict({"val_metrics/masked_crps": mean_masked_crps})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_initial.learning_rate, weight_decay=self.hparams.weight_decay)  # type: ignore
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0.000001)
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='U-Net for precipitation prediction')
    # parser.add_argument('-f','--fold', help='Fold for tensorboard logging version', type=int, default=0, required=False)
    # args = vars(parser.parse_args())
    
    # import wandb
    # wandb_logger = WandbLogger(name='U-Net LR Schedule - Medium', project="PrecipitationUNet", log_model="all", dir="logs")
    # tb_logger = TensorBoardLogger("/home/gregor/precipitation/eva_precipitation/testlogs", name="UNet-small", version="fold0", default_hp_metric=False)
    
    pl.seed_everything(123)
    cli = LightningCLI(
        model_class=PrecipitationUNet,
        datamodule_class=PrecipitationDataModule,
        trainer_defaults={
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": 5,
            # "logger": tb_logger,
        },
        save_config_kwargs={"overwrite": True}
    )
    # wandb.finish()
