from pathlib import Path
import numpy as np
import xarray as xr
from dataclasses import dataclass
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
        
    
    def subset_v1(self) -> tuple[list, list]:
        train = [self.path_kindx_train, self.path_corr1_train, self.path_pw_train, self.path_cape_train,
                 self.path_rh8_train, self.path_d2m_train, self.path_geodiff_train, self.path_sp_train]
        test = [self.path_kindx_test, self.path_corr1_test, self.path_pw_test, self.path_cape_test,
                 self.path_rh8_test, self.path_d2m_test, self.path_geodiff_test, self.path_sp_test]
        
        return train, test
    

class PrecipitationDataModule(LightningDataModule):
    def __init__(self, data_dir: str | Path = '~/datasets/precipitation', feature_set: str = 'v1', batch_size: int = 32, fold: int = 0,  **kwargs):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.feature_set = feature_set
        self.batch_size = batch_size
        self.fold = fold
    
    def load_and_concat(self, list_of_features: list = ['kindx_2000_2017.nc'], stage: str | None = 'train') -> np.ndarray:
        data_list = []
        for feature in list_of_features:
            dataset = xr.open_dataset(self.data_dir / 'predictors' / stage / feature)
            data_list.append(dataset[list(dataset.data_vars)[0]].values)
             
        return np.stack(data_list, axis=1)
    
    def setup(self, stage: str | None = None) -> None:
        self.data = PrecipitationDataPaths()
        if self.feature_set == 'v1':
            feature_set_train, feature_set_test = self.data.subset_v1()
        if stage == 'train':
            data_array = self.load_and_concat(feature_set_train, stage)
        else:
            data_array = self.load_and_concat(feature_set_test, stage)
        
        self.dataset = data_array
    
    def train_dataloader(self) -> list:
        return self.data.subset_v1()
    
    def val_dataloader(self) -> list:
        return self.data.subset_v1()
    
    def test_dataloader(self) -> list:
        return self.data.subset_v1()


if __name__ == "__main__":
    data = PrecipitationDataModule()
    data.setup(stage='test')
    data.train_dataloader()
    data.val_dataloader()
    data.test_dataloader()