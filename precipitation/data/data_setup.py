from dataclasses import dataclass


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
    path_stream_train = "predictors/train/stream_2000_2018.nc"
    path_geodiff_train = "predictors/train/geodiff_2000_2018.nc"
    path_vertvelo_train = "predictors/train/vert_velocity_mean850_500_300_2000_2018.nc"
    path_vimd_train = "predictors/train/accum_vimd_2000_2018.nc"
    path_pressure_tendency_train = "predictors/train/pressure_tendency_2000_2018.nc"

    path_t850_train = "predictors/train/temp_850_2000_2018.nc"
    path_t500_train = "predictors/train/temp_500_2000_2018.nc"
    path_sh500_train = "predictors/train/spec_humid500_2000_2018.nc"

    path_precip_lag1_train = "predictors/train/precip_obs_lag_1_sel_2000_2018.nc"
    path_precip_lag2_train = "predictors/train/precip_obs_lag_2_sel_2000_2018.nc"
    path_precip_lag3_train = "predictors/train/precip_obs_lag_3_sel_2000_2018.nc"

    path_hres_train = "predictors/train/hres_2000_2018.nc"

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
    path_stream_test = "predictors/test/stream_2019.nc"
    path_geodiff_test = "predictors/test/geodiff_2019.nc"
    path_vertvelo_test = "predictors/test/vert_velocity_mean850_500_300_2019.nc"
    path_vimd_test = "predictors/test/accum_vimd_2019.nc"
    path_pressure_tendency_test = "predictors/test/pressure_tendency_2019.nc"
    path_t850_test = "predictors/train/temp_850_2019.nc"
    path_t500_test = "predictors/train/temp_500_2019.nc"
    path_sh500_test = "predictors/train/spec_humid500_2019.nc"
    path_precip_lag1_test = "predictors/test/precip_obs_lag_1_sel_2019.nc"
    path_precip_lag2_test = "predictors/test/precip_obs_lag_2_sel_2019.nc"
    path_precip_lag3_test = "predictors/test/precip_obs_lag_3_sel_2019.nc"
    path_hres_test = "predictors/test/hres_2019.nc"


def select_data_subset(paths, version='v1', fold=0):
    if version in ["v2", "v2+time"]:
        return subset_v2(paths, fold)
    else:
        raise NotImplemented("Data subset version not implemented.")


# Version: Use laged precipitation values
def subset_v2(paths, fold) -> tuple[list[str], list[str]]:
    train = [
        paths.path_pw_train,
        paths.path_cape_train,
        paths.path_cc_train,
        paths.path_clwc_train,
        paths.path_rh5_train,
        paths.path_rh3_train,
        paths.path_d2m_train,
        paths.path_cin_train,
        paths.path_sh600_train,
        paths.path_sh925_train,
        paths.path_temp_train,
        paths.path_kindx_train,
        paths.path_sh7_train,
        paths.path_shear_train,
        paths.path_vimd_train,
        paths.path_stream_train,
        paths.path_pressure_tendency_train,
        paths.path_t850_train,
        paths.path_t500_train,
        paths.path_sh500_train,
        paths.path_precip_lag1_train,
        paths.path_precip_lag2_train,
        paths.path_precip_lag3_train,
    ]
    test = [
        paths.path_pw_test,
        paths.path_cape_test,
        paths.path_cc_test,
        paths.path_clwc_test,
        paths.path_rh5_test,
        paths.path_rh3_test,
        paths.path_d2m_test,
        paths.path_cin_test,
        paths.path_sh600_test,
        paths.path_sh925_test,
        paths.path_temp_test,
        paths.path_kindx_test,
        paths.path_sh7_test,
        paths.path_shear_test,
        paths.path_vimd_test,
        paths.path_stream_test,
        paths.path_pressure_tendency_test,
        paths.path_t850_test,
        paths.path_t500_test,
        paths.path_sh500_test,
        paths.path_precip_lag1_test,
        paths.path_precip_lag2_test,
        paths.path_precip_lag3_test,
    ]

    return train, test




