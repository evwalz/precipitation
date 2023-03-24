from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import torch
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
import pickle

@dataclass
class PrecipitationDataPaths:
    """Class to handle precipitation data."""

    path_pw_train = "predictors/train/tcwv_2000_2018.nc"
    path_cape_train = "predictors/train/cape_2000_2018.nc"
    path_cc_train = "predictors/train/cloudcover_2000_2018.nc"
    path_clwc_train = "predictors/train/cloudwater_2000_2018.nc"
    path_rh5_train = "predictors/train/rh500_2000_2018.nc"
    path_rh3_train = "predictors/train/rh300_update_2000_2018.nc"
    path_d2m_train = "predictors/train/d2m_2000_2018.nc"
    path_cin_train = "predictors/train/cin_2000_2018.nc"
    path_vo7_train = "predictors/train/relvor700_2000_2018.nc"
    path_sh600_train = "predictors/train/spec_humid600_2000_2018.nc"
    path_sh925_train= "predictors/train/spec_humid925_2000_2018.nc"
    path_temp_train = "predictors/train/t2m_2000_2018.nc"
    path_kindx_train = "predictors/train/kindx_2000_2018.nc"
    path_sh7_train = "predictors/train/spec_humid700_2000_2018.nc"
    path_sp_train = "predictors/train/surfpressure_2000_2018.nc"
    path_shear_train = "predictors/train/shear925_600_2000_2018.nc"
    # path_stream_train = "predictors/train/stream_2000_2018.nc"
    # path_geodiff_train = "predictors/train/geodiff_2000_2018.nc"
    # path_vertvelo_train = "predictors/train/vert_velocity_mean850_500_300_2000_2018.nc"
    # path_vimd_train = "predictors/train/accum_vimd_2000_2018.nc"
    # path_pressure_tendency_train = "predictors/train/pressure_tendency_2000_2018.nc"
    # path_precip_lag1_train = "predictors/train/precip_obs_lag_1_2000_2018.nc"
    # path_precip_lag2_train = "predictors/train/precip_obs_lag_2_2000_2018.nc"
    # path_precip_lag3_train = "predictors/train/precip_obs_lag_3_2000_2018.nc"

    path_pw_test = "predictors/test/tcwv_2019.nc"
    path_cape_test = "predictors/test/cape_2019.nc"
    path_cc_test = "predictors/test/cloudcover_2019.nc"
    path_clwc_test = "predictors/test/cloudwater_2019.nc"
    path_rh5_test = "predictors/test/rh500_2019.nc"
    path_rh3_test = "predictors/test/rh300_update_2019.nc"
    path_d2m_test = "predictors/test/d2m_2019.nc"
    path_cin_test = "predictors/test/cin_2019.nc"
    path_vo7_test = "predictors/test/relvor700_2019.nc"
    path_sh600_test = "predictors/test/spec_humid600_2019.nc"
    path_sh925_test = "predictors/test/spec_humid925_2019.nc"
    path_temp_test = "predictors/test/t2m_2019.nc"
    path_kindx_test = "predictors/test/kindx_2019.nc"
    path_sh7_test = "predictors/test/spec_humid700_2019.nc"
    path_sp_test = "predictors/test/surfpressure_2019.nc"
    path_shear_test = "predictors/test/shear925_600_2019.nc"
    # path_stream_test = "predictors/test/stream_2019.nc"
    # path_geodiff_test = "predictors/test/geodiff_2019.nc"
    # path_vertvelo_test = "predictors/test/vert_velocity_mean850_500_300_2019.nc"
    # path_vimd_test = "predictors/test/accum_vimd_2019.nc"
    # path_pressure_tendency_test = "predictors/test/pressure_tendency_2019.nc"
    # path_precip_lag1_test = "predictors/test/precip_obs_lag_1_2019.nc"
    # path_precip_lag2_test = "predictors/test/precip_obs_lag_2_2019.nc"
    # path_precip_lag3_test = "predictors/test/precip_obs_lag_3_2019.nc"

class PerFeatureMinMaxScaler:
    def __init__(
        self, feature_range: tuple[float, float] = (-1, 1), axis: int = 1
    ) -> None:
        self.feature_range = feature_range
        self.axis = axis
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray) -> None:
        self.n_features = data.shape[self.axis]
        self.min = np.zeros(self.n_features)
        self.max = np.zeros(self.n_features)

        for i in range(self.n_features):
            self.min[i] = data[:, i].min()
            self.max[i] = data[:, i].max()

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min is None or self.max is None:
            raise ValueError("You must call fit before transform or inverse transform.")
        new_data = np.zeros(data.shape)
        for i in range(self.n_features):
            new_data[:, i] = (data[:, i] - self.min[i]) / (
                self.max[i] - self.min[i]
            ) * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        return new_data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.min is None or self.max is None:
            raise ValueError("You must call fit before transform or inverse transform.")
        new_data = np.zeros(data.shape)
        for i in range(self.n_features):
            new_data[:, i] = (data[:, i] - self.feature_range[0]) / (
                self.feature_range[1] - self.feature_range[0]
            ) * (self.max[i] - self.min[i]) + self.min[i]

        return new_data


class PerFeatureMeanStdScaler:
    def __init__(self, axis: int = 1) -> None:
        self.axis = axis
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> None:
        self.n_features = data.shape[self.axis]
        self.mean = np.zeros(self.n_features)
        self.std = np.zeros(self.n_features)

        for i in range(self.n_features):
            self.mean[i] = data[:, i].mean()
            self.std[i] = data[:, i].std()

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("You must call fit before transform or inverse transform.")
        new_data = np.zeros(data.shape)
        for i in range(self.n_features):
            if self.std[i] == 0:
                new_data[:, i] = data[:, i] - self.mean[i]
            else:
                new_data[:, i] = (data[:, i] - self.mean[i]) / self.std[i]

        return new_data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("You must call fit before transform or inverse transform.")
        new_data = np.zeros(data.shape)
        for i in range(self.n_features):
            if self.std[i] == 0:
                new_data[:, i] = data[:, i] + self.mean[i]
            else:
                new_data[:, i] = (data[:, i] * self.std[i]) + self.mean[i]

        return new_data


class TargetMaxScaler:
    def __init__(self) -> None:
        self.max = None

    def fit(self, data: np.ndarray) -> None:
        self.max = data.max()

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.max is None:
            raise ValueError("You must call fit before transform or inverse transform.")
        new_data = data / (self.max + 0.001)
        return new_data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.max is None:
            raise ValueError("You must call fit before transform or inverse transform.")
        new_data = data * (self.max + 0.001)
        return new_data


class TargetLogScaler:
    def __init__(self, offset: float = 0.01) -> None:
        self.offset = offset

    def fit(self, data: np.ndarray | torch.Tensor) -> None:
        pass

    def transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(data, torch.Tensor):
            new_data = torch.log(data + self.offset)
        else:
            new_data = np.log(data + self.offset)
        return new_data

    def fit_transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(data, torch.Tensor):
            new_data = torch.exp(data) - self.offset
        else:
            new_data = np.exp(data) - self.offset
        return new_data


class NoScalerDummy:
    def __init__(self) -> None:
        pass

    def fit(self, data: np.ndarray) -> None:
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data


class PrecipitationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "~/datasets/precipitation",
        feature_set: str = "v1",
        batch_size: int = 32,
        fold: int = 0,
        scaler: str = "mean-std",
        normalization_range: tuple[float, float] = (-1, 1),
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.feature_set = feature_set
        self.batch_size = batch_size
        self.fold = fold

        if scaler == "min-max":
            self.scaler_inputs = PerFeatureMinMaxScaler(
                feature_range=normalization_range, axis=1
            )
        elif scaler == "mean-std":
            self.scaler_inputs = PerFeatureMeanStdScaler(axis=1)
        else:
            raise NotImplementedError("Scaler {} is not implemented.".format(scaler))
        # self.scaler_target = TargetMaxScaler()
        self.scaler_target = NoScalerDummy()
        
        self.save_hyperparameters()

    @property
    def n_features(self) -> int:
        if self.feature_set == "v1":
            return 21
        elif self.feature_set == "v1+time":
            return 23
        else:
            raise NotImplementedError(
                "Feature set {} is not implemented.".format(self.feature_set)
            )

    def load_and_concat(
        self,
        list_of_features: list[str] = ["kindx_2000_2018.nc"],
        add_time: bool = False,
        mode: str = "train",
    ) -> tuple[np.ndarray, np.ndarray]:
        
        with open(str(self.data_dir / 'split_train_folds.pickle'), 'rb') as f:
            X = pickle.load(f)
        timeseries_cv_split_manual = X[0:8]
        self.cv_fold = timeseries_cv_split_manual[self.fold]
        
        data_list = []
        for i, feature in enumerate(list_of_features):
            if mode == "val" and "corr_predictors" in feature:
                feature = feature.replace("train", "test")  
            #if mode == "val" and "upstream_predictors" in feature:
            #   feature = feature.replace("train", "test")
            dataset = xr.open_dataset(
                self.data_dir / feature
            )
            
            data_array = dataset[list(dataset.data_vars)[0]].values

            if "corr_predictors" in feature:
                data_array = np.log(data_array + 0.001)
                
            else:
                if mode == "train":
                    data_array = data_array[self.cv_fold[0]]
                elif mode == "val":
                    data_array = data_array[self.cv_fold[1]]

            if add_time and i == 0:
                days_oty = pd.date_range(start=dataset.time.values[0], end=dataset.time.values[-1], freq="D").dayofyear.to_numpy() - 1  # type: ignore
                days_oty = np.tile(
                    np.expand_dims(days_oty, (1, 2)),
                    (1, data_array.shape[1], data_array.shape[2]),
                )
                time_encoding_1 = np.sin(2 * np.pi * days_oty / 365)
                time_encoding_2 = np.cos(2 * np.pi * days_oty / 365)

                data_list.append(time_encoding_1)
                data_list.append(time_encoding_2)

            data_list.append(data_array)

        target_filename = (
            "obs_precip_train.nc" if mode in ["train", "val"] else "obs_precip_test.nc"
        )
        target = xr.open_dataset(self.data_dir / "observation" / target_filename)
        target_array = target["precipitationCal"].values
        if mode == "train":
            target_array = target_array[self.cv_fold[0]]
        elif mode == "val":
            target_array = target_array[self.cv_fold[1]]

        return np.stack(data_list, axis=1), target_array

    def setup(self, stage: str | None = None) -> None:
        paths = PrecipitationDataPaths()
        if self.feature_set == "v1":
            feature_set_train, feature_set_test = self.subset_v1(paths)
            add_time = False
        elif self.feature_set == "v1+time":
            feature_set_train, feature_set_test = self.subset_v1(paths)
            add_time = True
        else:
            raise NotImplementedError(
                f"Feature set {self.feature_set} is not implemented."
            )

        if stage == "fit" or stage is None:
            data_array_train, target_array_train = self.load_and_concat(
                feature_set_train, add_time=add_time, mode="train"
            )

            dataset_train = self.scaler_inputs.fit_transform(data_array_train)
            target_train = self.scaler_target.fit_transform(target_array_train)

            data_array_val, target_array_val = self.load_and_concat(
                feature_set_train, add_time=add_time, mode="val"
            )
            
            dataset_val = self.scaler_inputs.transform(data_array_val)
            target_val = self.scaler_target.transform(target_array_val)

            self.train_data = torch.from_numpy(dataset_train).float()
            self.val_data = torch.from_numpy(dataset_val).float()
            self.train_target = torch.from_numpy(target_train).float()
            self.val_target = torch.from_numpy(target_val).float()

        if stage == "test" or stage is None:
            data_array_test, target_array_test = self.load_and_concat(
                feature_set_test, add_time=add_time, mode="test"
            )

            dataset_test = self.scaler_inputs.transform(data_array_test)
            target_test = self.scaler_target.transform(target_array_test)

            self.test_data = torch.from_numpy(dataset_test).float()
            self.test_target = torch.from_numpy(target_test).float()

    def train_dataloader(self) -> DataLoader:
        if self.trainer and isinstance(self.trainer.accelerator, CUDAAccelerator):
            self.train_data = self.train_data.to(torch.device("cuda"))
            self.train_target = self.train_target.to(torch.device("cuda"))
        if self.trainer and isinstance(self.trainer.accelerator, MPSAccelerator):
            self.train_data = self.train_data.to(torch.device("mps"))
            self.train_target = self.train_target.to(torch.device("mps"))

        dataset = TensorDataset(self.train_data, self.train_target)

        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        if self.trainer and isinstance(self.trainer.accelerator, CUDAAccelerator):
            self.train_data = self.train_data.to(torch.device("cuda"))
            self.train_target = self.train_target.to(torch.device("cuda"))
            self.val_data = self.val_data.to(torch.device("cuda"))
            self.val_target = self.val_target.to(torch.device("cuda"))
        if self.trainer and isinstance(self.trainer.accelerator, MPSAccelerator):
            self.train_data = self.train_data.to(torch.device("mps"))
            self.train_target = self.train_target.to(torch.device("mps"))
            self.val_data = self.val_data.to(torch.device("mps"))
            self.val_target = self.val_target.to(torch.device("mps"))
            
        val_dataset = TensorDataset(self.val_data, self.val_target)
        train_dataset = TensorDataset(self.train_data, self.train_target)

        val_set_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        train_set_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        return [val_set_loader, train_set_loader]


    def test_dataloader(self) -> DataLoader:
        if self.trainer and isinstance(self.trainer.accelerator, CUDAAccelerator):
            self.test_data = self.test_data.to(torch.device("cuda"))
            self.test_target = self.test_target.to(torch.device("cuda"))
        if self.trainer and isinstance(self.trainer.accelerator, MPSAccelerator):
            self.test_data = self.test_data.to(torch.device("mps"))
            self.test_target = self.test_target.to(torch.device("mps"))

        dataset = TensorDataset(self.test_data, self.test_target)

        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
    def subset_v1(self, paths) -> tuple[list[str], list[str]]:
        train = [
            f"corr_predictors/predictor_train_{self.fold+10}_{self.fold+11}_1lag.nc",
            f"corr_predictors/predictor_train_{self.fold+10}_{self.fold+11}_2lag.nc",
            f"corr_predictors/predictor_train_{self.fold+10}_{self.fold+11}_3lag.nc",
            paths.path_pw_train,
            paths.path_cape_train,
            paths.path_cc_train,
            paths.path_clwc_train,
            paths.path_rh5_train, 
            paths.path_rh3_train,
            paths.path_d2m_train,
            paths.path_cin_train,
            paths.path_vo7_train,
            paths.path_sh600_train,
            paths.path_sh925_train,
            paths.path_temp_train,
            paths.path_kindx_train,
            paths.path_sh7_train,
            paths.path_sp_train,
            paths.path_shear_train,
            #paths.path_geodiff_train,
            #paths.path_vertvelo_train,
            #paths.path_vimd_train,
            #paths.path_stream_train,
            #path_pressure_tendency_train,
            #path_precip_lag1_train,
            #path_precip_lag2_train,
            #path_precip_lag3_train,
            #f"upstream_predictors/geodiff/predictor_train_abs_{self.fold+10}_{self.fold+11}_0lag.nc",
            #f"upstream_predictors/vvmean/predictor_train_abs_{self.fold+10}_{self.fold+11}_0lag.nc",
        ]
        test = [
            "corr_predictors/predictor_test_18_19_1lag.nc",
            "corr_predictors/predictor_test_18_19_2lag.nc",
            "corr_predictors/predictor_test_18_19_3lag.nc",
            paths.path_pw_test,
            paths.path_cape_test,
            paths.path_cc_test,
            paths.path_clwc_test,
            paths.path_rh5_test, 
            paths.path_rh3_test,
            paths.path_d2m_test,
            paths.path_cin_test,
            paths.path_vo7_test,
            paths.path_sh600_test,
            paths.path_sh925_test,
            paths.path_temp_test,
            paths.path_kindx_test,
            paths.path_sh7_test,
            paths.path_sp_test,
            paths.path_shear_test,
            #paths.path_geodiff_test,
            #paths.path_vertvelo_test,
            #paths.path_vimd_test,
            #paths.path_stream_test,
            #path_pressure_tendency_test,
            #path_precip_lag1_test,
            #path_precip_lag2_test,
            #path_precip_lag3_test,
            #f"upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
            #f"upstream_predictors/vvmean/predictor_test_abs_18_19_0lag.nc",
        ]

        return train, test


if __name__ == "__main__":
    data = PrecipitationDataModule(feature_set="v1+time")
    data.setup(stage="fit")

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    data.setup(stage="test")
    test_loader = data.test_dataloader()
    print("Done")
