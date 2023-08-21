'''
Functions to compute synoptic indices and patterns
S. Filhol, August 2023

**Indices:**
- blocking
    - 2D blocking index according to Nagaviuc et al.(2022)


**Patterns:**
- unsupervised
    - [x] K-means clustering
    - [ ] CNN feature extraction followed by K-mean clustering
- supervised classification given a catalogue of synoptic patterns (TBI). Some interesting models are the image
recognition models such as used in this [notebook](https://www.kaggle.com/code/hamzamanssor/weather-recognition-using-deep-learning-models) for classifying type of precipitation based on image.


'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, MaxPool2D, Dense

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


def cnn_model():
    '''
    Example of a CNN model for a stack of 2D arrays (ny,nx) shape. This model has not been tested thoroughly.

    Returns:
        tensorflow CNN model object
    '''

    kernel_size = (3,3)
    model = Sequential()
    model.add(Conv2D(input_shape=(ny,nx,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=kernel_size,padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=kernel_size, padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=kernel_size, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dense(32))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='mse')

    # print model summary to console
    model.summary()
    return model


def cnn_kmeans_cluster_map(arr, model=None):
    # IN CONSTRUCTION
    # Convert dataset to numpy array (2D or 3D). Array is organize [var, time, lat, lon]

    # Standardize data and reshape array for CNN model
    scaler = RobustScaler()
    if len(arr.shape) == 4:
        arr_sc = scaler.fit_transform(arr.reshape(2,-1).T)
        arr_tr = arr_sc.T.reshape((2, nt, ny, nx))
        arr_tr = np.transpose(arr_tr, (1,2,3,0))
    elif len(arr.shape) == 3:
        arr_sc = scaler.fit_transform(arr.reshape(1,-1).T)
        arr_tr = arr_sc.T.reshape((nt, ny, nx))
    else:
        print('ERROR: shape of input array not compatible. Must be [n_var, n_time, n_latitude, n_longitude] or [n_time, n_latitude, n_longitude]')
        return

    features = model.predict(arr_tr)

    # Perform K-Means clustering
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features)

    return cluster_labels, features, kmeans






def blocking_index(era):
    '''
    Function to compute the blocking index according to Nagaviuc et al.(2022) itself based on Tibaldi and Molteni (1990)
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

