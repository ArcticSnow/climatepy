'''
Functions to computde synoptic indexes

- 2D Blocking index following the method in Nagavciuc et al.(2022) which is based on Tibaldi and Molteni (1990)
'''
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import fetch_era5 as fe
import geo_utils as gu

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

def convert_era5_to_daily(f_input, f_output, var=['z_anomaly', 'u', 'v']):

    ds = xr.open_dataset(f_input)
    dd = ds.resample(time='D').mean(dim='time')
    dd['z'] = dd.z / 9.80665
    tmp = (dd.z - dd.z.mean(dim=('longitude', 'latitude')))
    dd['z_anomaly'] = tmp - tmp.mean(dim='time')
    te.to_netcdf(dd, f_output, var=var)


def plot_synoptic(map, extent=[], projection=ccrs.PlateCarree(), plot_wind=True):
    '''
    In Construction ...
    '''
    plt.figure(figsize=(12,6))
    p = plt.subplot(1,1, 1,projection=ccrs.PlateCarree())

    p.coastlines('50m', linewidth=1)
    p.set_extent([lon-50, lon+50, lat-30, lat+30], ccrs.PlateCarree())

    dd_coarse.sel(cluster=cluster).centroids.plot.imshow(interpolation='nearest',ax=p, cmap=plt.cm.Spectral_r)
    dd_coarse.sel(cluster=cluster).centroids.plot.contour(colors='k', levels=np.arange(-150, 150, 20), ax=p, linewidths=2, alpha=0.3)
    if plot_wind:
        dd_coarse.isel(time=step).plot.quiver(x='longitude', y='latitude', u='u', v='v', ax=p)


def kmeans_cluster_maps(ds, n_cluster=100, var_clust='z_anomaly', lat_res=5, lon_res=5):
    '''
    Function to cluster a stack of maps (n * 2D Arrays) using kmeans
    Args:
        ds:
        n_cluster:
        var_clust:
        lat_res:
        lon_res:

    Returns:

    '''
    if lat_res is not None and lon_res is not None:
        print(f'Resampling maps to new resolutino (longitude={lon_res} ; latitude={lat_res})')
        dd_coarse = ds.coarsen(longitude=lon_res, latitude=lat_res, boundary='pad').mean()
    else:
        dd_coarse = ds

    # Reshape and Standardize maps
    num_samples, height, width = dd_coarse[var_clust].shape
    image_array = dd_coarse[var_clust].values.reshape(num_samples, height * width)
    scaler = RobustScaler()
    normalized_data = scaler.fit_transform(image_array)

    # Perform K-mean
    kmeans = MiniBatchKMeans(n_clusters=n_cluster, random_state=0)
    cluster_labels = kmeans.fit_predict(normalized_data)

    centroids = scaler.inverse_transform(kmeans.cluster_centers_).reshape(num_clusters, height, width)
    dd_coarse['centroids'] = (('cluster', 'latitude', 'longitude'), centroids)

    return kmeans, scaler, cluster_labels




def blocking_index(era):
    '''
    Function to compute the blocking index according to Nagavciuc et al.(2022) itself based on Tibaldi and Molteni (1990)
    https://www.cpc.ncep.noaa.gov/products/precip/CWlink/blocking/index/index.nh.shtml
    
    Args:
        era: daily 500mb geopotential height from ERA5

    Returns:
        dataset with Northern gradient, sourthern gradient and blocking index

    TODO:
    - add option to fetch ERA5 data
    '''
    # use the 500 mb geopotential
    ds = era.z / 9.81
    ds = gu.convert_longitude(ds)  # convert longitude from 0:360 to -180:180
    dss = ds.to_dataset()
    dss['GHGS'] = (ds - ds.shift(latitude=15)) / 15
    dss['GHGN'] = (ds.shift(latitude=15) - ds) / 15
    dss['BI'] = (dss.GHGS > 0) & (dss.GHGN < -10)

    return dss

