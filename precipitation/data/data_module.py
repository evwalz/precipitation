from pathlib import Path
import numpy as np
import xarray as xr
import torch
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule


@dataclass
class PrecipitationDataPaths:
    """Class to handle precipitation data."""
    
    path_kindx_train = 'kindx_2000_2017.nc'
    path_corr1_train = 'corr_lag1_train.nc'
    path_corr2_train = 'corr_lag2_train.nc'
    path_corr3_train ='corr_lag3_train.nc'
    path_pw_train = 'tcwv_2000_2017.nc'
    path_cape_train = 'cape_2000_2017.nc'
    path_d2m_train = 'd2m_2000_2017.nc'
    path_t3_train = 't300_2000_2017.nc'
    path_t5_train = 't500_2000_2017.nc'
    path_t8_train = 't800_2000_2017.nc'
    path_rh5_train = 'rh500_2000_2017.nc'
    path_rh8_train = 'rh800_2000_2017.nc'
    path_toa_train = 'toa_0pm_2000_2017.nc'
    path_geodiff_train = 'geodiff_2000_2017.nc'
    path_cc_train = 'cloudcover_2000_2017.nc'
    path_clwc_train = 'cloudwater_2000_2017.nc'
    path_ciwc5_train = 'cloudice500_2000_2017.nc'
    path_temp_train = 't2m_2000_2017.nc'
    path_g5_train = 'geo500_2000_2017.nc'
    path_g7_train = 'geo700_2000_2017.nc'
    path_sh7_train = 'specifichum700_2000_2017.nc'
    path_vo7_train = 'relvor700_2000_2017.nc'
    path_cin_train = 'cin_2000_2017.nc'
    path_shear_train = 'shear925_600_2000_2017.nc'
    path_sp_train = 'surfpressure_2000_2017.nc'
    path_sp4_train = 'surfpressure_4pm_2000_2017.nc'
    path_geo850_train = 'geo850_4pm_2000_2017.nc'
    
    path_kindx_test = 'kindx_2018_2019.nc'
    path_corr1_test = 'corr_lag1_test.nc'
    path_corr2_test = 'corr_lag2_test.nc'
    path_corr3_test = 'corr_lag3_test.nc'
    path_pw_test = 'tcwv_2018_2019.nc'
    path_cape_test = 'cape_2018_2019.nc'
    path_d2m_test = 'd2m_2018_2019.nc'
    path_t3_test = 't300_2018_2019.nc'
    path_t5_test = 't500_2018_2019.nc'
    path_t8_test = 't800_2018_2019.nc'
    path_rh5_test = 'rh500_2018_2019.nc'
    path_rh8_test = 'rh800_2018_2019.nc'
    path_toa_test = 'toa_0pm_2018_2019.nc'
    path_geodiff_test = 'geodiff_2018_2019.nc'
    path_cc_test = 'cloudcover_2018_2019.nc'
    path_clwc_test = 'cloudwater_2018_2019.nc'
    path_ciwc5_test = 'cloudice500_2018_2019.nc'
    path_temp_test = 't2m_2018_2019.nc'
    path_g5_test = 'geo500_2018_2019.nc'
    path_g7_test = 'geo700_2018_2019.nc'
    path_sh7_test = 'specifichum700_2018_2019.nc'
    path_vo7_test = 'relvor700_2018_2019.nc'
    path_cin_test = 'cin_2018_2019.nc'
    path_shear_test = 'shear925_600_2018_2019.nc'
    path_sp_test = 'surfpressure_2018_2019.nc'
    path_sp4_test = 'surfpressure_4pm_2018_2019.nc'
    path_geo850_test = 'geo850_4pm_2018_2019.nc'
        
    
    def subset_v1(self) -> tuple[list[str], list[str]]:
        train = [self.path_kindx_train, self.path_corr1_train, self.path_pw_train, self.path_cape_train,
                 self.path_rh8_train, self.path_d2m_train, self.path_geodiff_train, self.path_sp_train]
        test = [self.path_kindx_test, self.path_corr1_test, self.path_pw_test, self.path_cape_test,
                 self.path_rh8_test, self.path_d2m_test, self.path_geodiff_test, self.path_sp_test]
        
        return train, test
    

class PrecipitationDataModule(LightningDataModule):
    def __init__(self, data_dir: str = '~/datasets/precipitation', feature_set: str = 'v1', batch_size: int = 32, fold: int = 0,  **kwargs):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.feature_set = feature_set
        self.batch_size = batch_size
        self.fold = fold
    
    def load_and_concat(self, list_of_features: list[str] = ['kindx_2000_2017.nc'], folder_data: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        data_list = []
        for feature in list_of_features:
            dataset = xr.open_dataset(self.data_dir / 'predictors' / folder_data / feature)
            data_list.append(dataset[list(dataset.data_vars)[0]].values)
        
        target = xr.open_dataset(self.data_dir / 'observation' / 'obs_precip_train.nc' if folder_data == 'train' else 'obs_precip_test.nc')
             
        return np.stack(data_list, axis=1), target['precipitationCal'].values
    
    def setup(self, stage: str | None = None) -> None:
        self.data = PrecipitationDataPaths()
        if self.feature_set == 'v1':
            feature_set_train, feature_set_test = self.data.subset_v1()
        else:
            raise NotImplementedError(f"Feature set {self.feature_set} is not implemented.")
        
        if stage == 'fit' or stage is None:
            data_array_train, target_array_train = self.load_and_concat(feature_set_train, 'train')
            timeseries_cv_splitter = TimeSeriesSplit(n_splits=7, test_size=365)
            self.cv_fold = list(timeseries_cv_splitter.split(data_array_train))[self.fold]
            
            self.dataset_train = torch.from_numpy(data_array_train).float()
            self.target_train = torch.from_numpy(target_array_train).float()
        if stage == 'test' or stage is None:
            data_array_test,target_array_test = self.load_and_concat(feature_set_test, 'test')
            self.dataset_test = torch.from_numpy(data_array_test).float()
            self.target_test = torch.from_numpy(target_array_test).float()
    
    def train_dataloader(self) -> DataLoader:
        train_data = self.dataset_train[self.cv_fold[0]]
        train_target = self.target_train[self.cv_fold[0]]
        
        if self.trainer and self.trainer.on_gpu:
            train_data = train_data.cuda()
            train_target = train_target.cuda()
        
        dataset = TensorDataset(train_data, train_target)
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def val_dataloader(self) -> DataLoader:
        val_data = self.dataset_train[self.cv_fold[1]]
        val_target = self.target_train[self.cv_fold[1]]
        
        if self.trainer and self.trainer.on_gpu:
            val_data = val_data.cuda()
            val_target = val_target.cuda()
        
        dataset = TensorDataset(val_data, val_target)
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self) -> DataLoader:
        test_data = self.dataset_test
        test_target = self.target_test
        
        if self.trainer and self.trainer.on_gpu:
            test_data = test_data.cuda()
            test_data = test_data.cuda()
        
        dataset = TensorDataset(test_data, test_target)
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


if __name__ == "__main__":
    data = PrecipitationDataModule()
    data.setup(stage='fit')
    data.train_dataloader()
    data.val_dataloader()
    data.test_dataloader()
