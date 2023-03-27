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
    path_precip_lag1_train = "predictors/train/precip_obs_lag_1_2000_2018.nc"
    path_precip_lag2_train = "predictors/train/precip_obs_lag_2_2000_2018.nc"
    path_precip_lag3_train = "predictors/train/precip_obs_lag_3_2000_2018.nc"

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
    path_precip_lag1_test = "predictors/test/precip_obs_lag_1_2019.nc"
    path_precip_lag2_test = "predictors/test/precip_obs_lag_2_2019.nc"
    path_precip_lag3_test = "predictors/test/precip_obs_lag_3_2019.nc"


def select_data_subset(paths, version='v1', fold=0):
    if version in ["v1", "v1+time"]:
        return subset_v1(paths, fold)
    elif version in ["v1_nocorr", "v1_nocorr+time"]:
        return subset_v1_nocorr(paths, fold)
    elif version in ["v2", "v2+time"]:
        return subset_v2(paths, fold)
    elif version in ["v3", "v3+time"]:
        return subset_v3(paths, fold)
    elif version in ["v4", "v4+time"]:
        return subset_v4(paths, fold)
    elif version in ["v5", "v5+time"]:
        return subset_v5(paths, fold)
    elif version in ["v6", "v6+time"]:
        return subset_v6(paths, fold)
    elif version in ["v7", "v7+time"]:
        return subset_v7(paths, fold)
    elif version in ["v7_nocorr", "v7_nocorr+time"]:
        return subset_v7_nocorr(paths, fold)
    else:
        raise NotImplemented("Data subset version not implemented.")


# Version 1: original subset
def subset_v1(paths, fold) -> tuple[list[str], list[str]]:
    train = [
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
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
        # paths.path_geodiff_train,
        # paths.path_vertvelo_train,
        # paths.path_vimd_train,
        # paths.path_stream_train,
        # paths.pressure_tendency_train,
        # paths.precip_lag1_train,
        # paths.precip_lag2_train,
        # paths.precip_lag3_train,
        # f"upstream_predictors/geodiff/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
        # f"upstream_predictors/vert_velocity/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
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
        # paths.path_geodiff_test,
        # paths.path_vertvelo_test,
        # paths.path_vimd_test,
        # paths.path_stream_test,
        # paths.pressure_tendency_test,
        # paths.precip_lag1_test,
        # paths.precip_lag2_test,
        # paths.precip_lag3_test,
        # "upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        # "upstream_predictors/vert_velocity/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test


def subset_v1_nocorr(paths, fold) -> tuple[list[str], list[str]]:
    train = [
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
        # paths.path_geodiff_train,
        # paths.path_vertvelo_train,
        # paths.path_vimd_train,
        # paths.path_stream_train,
        # paths.pressure_tendency_train,
        # paths.precip_lag1_train,
        # paths.precip_lag2_train,
        # paths.precip_lag3_train,
        # f"upstream_predictors/geodiff/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
        # f"upstream_predictors/vert_velocity/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
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
        paths.path_vo7_test,
        paths.path_sh600_test,
        paths.path_sh925_test,
        paths.path_temp_test,
        paths.path_kindx_test,
        paths.path_sh7_test,
        paths.path_sp_test,
        paths.path_shear_test,
        # paths.path_geodiff_test,
        # paths.path_vertvelo_test,
        # paths.path_vimd_test,
        # paths.path_stream_test,
        # paths.pressure_tendency_test,
        # paths.precip_lag1_test,
        # paths.precip_lag2_test,
        # paths.precip_lag3_test,
        # "upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        # "upstream_predictors/vert_velocity/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test


# Version 2: include 2 upstream
def subset_v2(paths, fold) -> tuple[list[str], list[str]]:
    train = [
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
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
        # paths.path_geodiff_train,
        # paths.path_vertvelo_train,
        # paths.path_vimd_train,
        # paths.path_stream_train,
        # paths.pressure_tendency_train,
        # paths.precip_lag1_train,
        # paths.precip_lag2_train,
        # paths.precip_lag3_train,
        f"upstream_predictors/geodiff/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
        f"upstream_predictors/vert_velocity/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
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
        # paths.path_geodiff_test,
        # paths.path_vertvelo_test,
        # paths.path_vimd_test,
        # paths.path_stream_test,
        # paths.pressure_tendency_test,
        # paths.precip_lag1_test,
        # paths.precip_lag2_test,
        # paths.precip_lag3_test,
        "upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        "upstream_predictors/vert_velocity/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test



# Version 3: include 3 more local features
def subset_v3(paths, fold) -> tuple[list[str], list[str]]:
    train = [
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
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
        # paths.path_geodiff_train,
        # paths.path_vertvelo_train,
        paths.path_vimd_train,
        paths.path_stream_train,
        paths.pressure_tendency_train,
        # paths.precip_lag1_train,
        # paths.precip_lag2_train,
        # paths.precip_lag3_train,
        f"upstream_predictors/geodiff/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
        f"upstream_predictors/vert_velocity/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
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
        # paths.path_geodiff_test,
        # paths.path_vertvelo_test,
        paths.path_vimd_test,
        paths.path_stream_test,
        paths.pressure_tendency_test,
        # paths.precip_lag1_test,
        # paths.precip_lag2_test,
        # paths.precip_lag3_test,
        "upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        "upstream_predictors/vert_velocity/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test

# Note: possibly remove some of the features for good?

# Version 4: Replace 2 upstream features (based on original or new subset)
def subset_v4(paths, fold) -> tuple[list[str], list[str]]:
    train = [
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
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
        paths.path_vimd_train,
        paths.path_stream_train,
        paths.pressure_tendency_train,
        # paths.precip_lag1_train,
        # paths.precip_lag2_train,
        # paths.precip_lag3_train,
        # f"upstream_predictors/geodiff/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
        # f"upstream_predictors/vert_velocity/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
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
        paths.path_vimd_test,
        paths.path_stream_test,
        paths.pressure_tendency_test,
        # paths.precip_lag1_test,
        # paths.precip_lag2_test,
        # paths.precip_lag3_test,
        # "upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        # "upstream_predictors/vert_velocity/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test


# Version 5: Replace all upstream features (based on original or new subset)
def subset_v5(paths, fold) -> tuple[list[str], list[str]]:
    train = [
        # f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        # f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        # f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
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
        paths.path_geodiff_train,
        paths.path_vertvelo_train,
        paths.path_vimd_train,
        paths.path_stream_train,
        paths.pressure_tendency_train,
        paths.precip_lag1_train,
        paths.precip_lag2_train,
        paths.precip_lag3_train,
        # f"upstream_predictors/geodiff/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
        # f"upstream_predictors/vert_velocity/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
    ]
    test = [
        # "corr_predictors/predictor_test_18_19_1lag.nc",
        # "corr_predictors/predictor_test_18_19_2lag.nc",
        # "corr_predictors/predictor_test_18_19_3lag.nc",
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
        paths.path_geodiff_test,
        paths.path_vertvelo_test,
        paths.path_vimd_test,
        paths.path_stream_test,
        paths.pressure_tendency_test,
        paths.precip_lag1_test,
        paths.precip_lag2_test,
        paths.precip_lag3_test,
        # "upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        # "upstream_predictors/vert_velocity/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test


# Version 6: Replace 2 upstream features and remove all precip corrs
def subset_v6(paths, fold) -> tuple[list[str], list[str]]:
    train = [
        #f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        #f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        #f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
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
        paths.path_geodiff_train,
        paths.path_vertvelo_train,
        paths.path_vimd_train,
        paths.path_stream_train,
        paths.pressure_tendency_train,
        # paths.precip_lag1_train,
        # paths.precip_lag2_train,
        # paths.precip_lag3_train,
        # f"upstream_predictors/geodiff/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
        # f"upstream_predictors/vert_velocity/predictor_train_abs_{fold+10}_{fold+11}_0lag.nc",
    ]
    test = [
        #"corr_predictors/predictor_test_18_19_1lag.nc",
        #"corr_predictors/predictor_test_18_19_2lag.nc",
        #"corr_predictors/predictor_test_18_19_3lag.nc",
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
        paths.path_geodiff_test,
        paths.path_vertvelo_test,
        paths.path_vimd_test,
        paths.path_stream_test,
        paths.pressure_tendency_test,
        # paths.precip_lag1_test,
        # paths.precip_lag2_test,
        # paths.precip_lag3_test,
        # "upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        # "upstream_predictors/vert_velocity/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test


def subset_v7(paths, fold) -> tuple[list[str], list[str]]:
    train = [
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_1lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_2lag.nc",
        f"corr_predictors/predictor_train_{fold + 10}_{fold + 11}_3lag.nc",
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
        paths.path_geodiff_train,
        paths.path_vertvelo_train,
        # paths.path_vimd_train,
        # paths.path_stream_train,
        # paths.pressure_tendency_train,
        # paths.precip_lag1_train,
        # paths.precip_lag2_train,
        # paths.precip_lag3_train,
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
        paths.path_geodiff_test,
        paths.path_vertvelo_test,
        # paths.path_vimd_test,
        # paths.path_stream_test,
        # paths.pressure_tendency_test,
        # paths.precip_lag1_test,
        # paths.precip_lag2_test,
        # paths.precip_lag3_test,
        #"upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        #"upstream_predictors/vvmean/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test


def subset_v7_nocorr(paths, fold) -> tuple[list[str], list[str]]:
    train = [
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
        paths.path_geodiff_train,
        paths.path_vertvelo_train,
        # paths.path_vimd_train,
        # paths.path_stream_train,
        # paths.pressure_tendency_train,
        # paths.precip_lag1_train,
        # paths.precip_lag2_train,
        # paths.precip_lag3_train,
        #f"upstream_predictors/geodiff/predictor_train_abs_{self.fold+10}_{self.fold+11}_0lag.nc",
        #f"upstream_predictors/vvmean/predictor_train_abs_{self.fold+10}_{self.fold+11}_0lag.nc",
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
        paths.path_vo7_test,
        paths.path_sh600_test,
        paths.path_sh925_test,
        paths.path_temp_test,
        paths.path_kindx_test,
        paths.path_sh7_test,
        paths.path_sp_test,
        paths.path_shear_test,
        paths.path_geodiff_test,
        paths.path_vertvelo_test,
        # paths.path_vimd_test,
        # paths.path_stream_test,
        # paths.pressure_tendency_test,
        # paths.precip_lag1_test,
        # paths.precip_lag2_test,
        # paths.precip_lag3_test,
        #"upstream_predictors/geodiff/predictor_test_abs_18_19_0lag.nc",
        #"upstream_predictors/vvmean/predictor_test_abs_18_19_0lag.nc",
    ]

    return train, test
