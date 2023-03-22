from isodisreg import idr
import pandas as pd
import numpy as np


def calculate_crps_idr(val_preds, val_target, train_preds, train_target, mask, lat, lon):
    if mask[lat,lon]:
        idr_per_grid = idr(y=train_target[:,lat,lon], X=pd.DataFrame(train_preds[:,lat,lon]))
        val_dist_pred = idr_per_grid.predict(pd.DataFrame(val_preds[:,lat,lon]))
        crps_per_grid = np.mean(val_dist_pred.crps(val_target[:,lat,lon])) # if seasonal validation, watch out here

        return crps_per_grid
