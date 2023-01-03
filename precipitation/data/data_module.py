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

    
    path_corr1_train = "corr_lag1_train.nc"
    path_corr2_train = "corr_lag2_train.nc"
    path_corr3_train = "corr_lag3_train.nc"
    path_pw_train = "tcwv_2000_2017.nc"
    path_cape_train = "cape_2000_2017.nc"
    path_cc_train = "cloudcover_2000_2017.nc"
    path_clwc_train = "cloudwater_2000_2017.nc"
    path_rh5_train = "rh500_2000_2017.nc"
    path_rh3_train = "rh300_2000_2017.nc"
    path_d2m_train = "d2m_2000_2017.nc"
    path_cin_train = "cin_2000_2017.nc"
    path_vo7_train = "relvor700_2000_2017.nc"
    path_sh600_train = "spec_humid600_2000_2017.nc"
    path_sh925_train= "spec_humid925_2000_2017.nc"
    path_temp_train = "t2m_2000_2017.nc"
    path_kindx_train = "kindx_2000_2017.nc"
    path_sh7_train = "spec_humid700_2000_2017.nc"
    path_sp_train = "surfpressure_2000_2017.nc"
    path_shear_train = "shear925_600_2000_2017.nc"
    path_geodiff_train = "geodiff_corr_2000_2017.nc"
    path_vertvelo_train = "vvmean_corr_2000_2017.nc"
    

    path_corr1_test = "corr_lag1_test.nc"
    path_corr2_test = "corr_lag2_test.nc"
    path_corr3_test = "corr_lag3_test.nc"
    path_pw_test = "tcwv_2018_2019.nc"
    path_cape_test = "cape_2018_2019.nc"
    path_cc_test = "cloudcover_2018_2019.nc"
    path_clwc_test = "cloudwater_2018_2019.nc"
    path_rh5_test = "rh500_2018_2019.nc"
    path_rh3_test = "rh300_2018_2019.nc"
    path_d2m_test = "d2m_2018_2019.nc"
    path_cin_test = "cin_2018_2019.nc"
    path_vo7_test = "relvor700_2018_2019.nc"
    path_sh600_test = "spec_humid600_2018_2019.nc"
    path_sh925_test = "spec_humid925_2018_2019.nc"
    path_temp_test = "t2m_2018_2019.nc"
    path_kindx_test = "kindx_2018_2019.nc"
    path_sh7_test = "spec_humid700_2018_2019.nc"
    path_sp_test = "surfpressure_2018_2019.nc"
    path_shear_test = "shear925_600_2018_2019.nc"
    path_geodiff_test = "geodiff_corr_2018_2019.nc"
    path_vertvelo_test = "vvmean_corr_2018_2019.nc"

    def subset_v1(self) -> tuple[list[str], list[str]]:
        train = [
            self.path_kindx_train,
            self.path_corr1_train,
            self.path_pw_train,
            self.path_cape_train,
            self.path_sh925_train,
            self.path_d2m_train,
            self.path_geodiff_train,
            self.path_sp_train,
            self.path_sh7_train,
            self.path_vertvelo_train,
            self.path_clwc_train,
            self.path_toa_train
        ]
        test = [
            self.path_kindx_test,
            self.path_corr1_test,
            self.path_pw_test,
            self.path_cape_test,
            self.path_sh925_test,
            self.path_d2m_test,
            self.path_geodiff_test,
            self.path_sp_test,
            self.path_sh7_test,
            self.path_vertvelo_test,
            self.path_clwc_test,
            self.path_toa_test,
        ]

        return train, test


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

    @property
    def n_features(self) -> int:
        if self.feature_set == "v1":
            return 8
        elif self.feature_set == "v1+time":
            return 10
        else:
            raise NotImplementedError(
                "Feature set {} is not implemented.".format(self.feature_set)
            )

    def load_and_concat(
        self,
        list_of_features: list[str] = ["kindx_2000_2017.nc"],
        add_time: bool = False,
        folder_data: str = "train",
    ) -> tuple[np.ndarray, np.ndarray]:
        data_list = []
        for i, feature in enumerate(list_of_features):
            dataset = xr.open_dataset(
                self.data_dir / "predictors" / folder_data / feature
            )
            data_array = dataset[list(dataset.data_vars)[0]].values

            if "corr_" in feature:
                data_array = np.log(data_array + 0.001)

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
            "obs_precip_train.nc" if folder_data == "train" else "obs_precip_test.nc"
        )
        target = xr.open_dataset(self.data_dir / "observation" / target_filename)

        return np.stack(data_list, axis=1), target["precipitationCal"].values

    def setup(self, stage: str | None = None) -> None:
        self.data = PrecipitationDataPaths()
        if self.feature_set == "v1":
            feature_set_train, feature_set_test = self.data.subset_v1()
            add_time = False
        elif self.feature_set == "v1+time":
            feature_set_train, feature_set_test = self.data.subset_v1()
            add_time = True
        else:
            raise NotImplementedError(
                f"Feature set {self.feature_set} is not implemented."
            )

        if stage == "fit" or stage is None:
            data_array_train, target_array_train = self.load_and_concat(
                feature_set_train, add_time=add_time, folder_data="train"
            )
            #timeseries_cv_splitter = TimeSeriesSplit(n_splits=7, test_size=365)
            #self.cv_fold = list(timeseries_cv_splitter.split(data_array_train))[
            #    self.fold
            #]
            with open('split_train_folds.pickle', 'rb') as f:
                X = pickle.load(f)
            timeseries_cv_split_manual = X[0:7]
            self.cv_fold = timeseries_cv_split_manual[self.fold]

            dataset = self.scaler_inputs.fit_transform(data_array_train)
            target = self.scaler_target.fit_transform(target_array_train)

            dataset = torch.from_numpy(dataset).float()
            target = torch.from_numpy(target).float()

            self.train_data = dataset[self.cv_fold[0]]
            self.train_target = target[self.cv_fold[0]]
            self.val_data = dataset[self.cv_fold[1]]
            self.val_target = target[self.cv_fold[1]]

        if stage == "test" or stage is None:
            data_array_test, target_array_test = self.load_and_concat(
                feature_set_test, add_time=add_time, folder_data="test"
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


if __name__ == "__main__":
    data = PrecipitationDataModule(feature_set="v1+time")
    data.setup(stage="fit")

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    data.setup(stage="test")
    test_loader = data.test_dataloader()
    print("Done")
