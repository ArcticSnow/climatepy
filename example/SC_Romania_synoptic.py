from TopoPyScale import topo_export as te
from TopoPyScale import topo_utils as tu
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from climatepy import synoptic
import pickle
from climatepy import fetch_climate as fc

#===============================================================================
#     Fetch Data from ERA5
#===============================================================================

# Location of the Southern Carpathians
lon = 24.6
lat = 45.6
n_clusters = 30

# fetch ERA5 data for the 500mb atmospheric layer
fc.fetch_era5('reanalysis',
                 '1950-09-01',
                 '2023-08-30', './data/hourly/',
                 lat+30,
                 lat-30,
                 lon+50,
                 lon-50,
                 step=1,
                 num_threads=30,
                 surf_plev='plev',
                 plevels=500,
                 varoi=['geopotential', 'u_component_of_wind','v_component_of_wind'])

# Transform hourly data to daily
flist = glob.glob('data/hourly/*nc')
flist.sort()

def fun(file):
    ds = xr.open_dataset(file)
    dd = ds.resample(time = 'D').mean(dim='time')
    dd['z'] = dd.z/9.80665
    tmp = (dd.z - dd.z.mean(dim=('longitude', 'latitude')))
    dd['z_anomaly'] = tmp - tmp.mean(dim='time')
    fout = file.split('/')[-1]
    fout = 'data/' + fout.split('.')[0] + 'daily.nc'
    te.to_netcdf(dd, fout, variables=['z_anomaly', 'u', 'v'])
    print(f'---> Daily file {fout.split("/")[-1]} ready!')

param = zip(flist)
tu.multicore_pooling(fun, param, n_cores=30)   # this is to be ran on a server
print('---> END: Daily file ready!')

# Generate random indices to sample the dataset. Keep 1/3 of the data to train the K-mean clustering
idx = np.random.choice(np.arange(0,26889),int(np.round(26889/3)), replace=False)

# export subsample to a netcdf file
te.to_netcdf(ds.z_anomaly.isel(time=idx).to_dataset(), fname='data/train_z_anomaly.nc')
ds_coarse = ds.coarsen(longitude=1, latitude=1, boundary='pad').mean()
te.to_netcdf(ds_coarse, fname='ds_coarse.nc')

dd = xr.open_dataset('data/train_z_anomaly.nc')
# perform the clustering on a resampled dataset
kmeans, scaler, cluster_labels, dd_coarse = synoptic.kmeans_cluster_maps(dd, n_clusters=n_clusters, lat_res=1, lon_res=1)

# Get stats of clusters
cl, cnt = np.unique(cluster_labels, return_counts=True)

#==============================================
# Store model results to disk
#==============================================
fname = 'kmeans_model.pckl'
with open(fname, 'wb') as file:
    pickle.dump(kmeans, file)

fname = 'kmeans_scaler.pckl'
with open(fname, 'wb') as file:
    pickle.dump(scaler, file)

fname = 'kmeans_cluster_labels.pckl'
with open(fname, 'wb') as file:
    pickle.dump(cluster_labels, file)

te.to_netcdf(dd_coarse, fname='kmeans_ds_train.nc')

# Load model from disk
fname = 'kmeans_model.pckl'
with open(fname, 'rb') as file:
    kmeans = pickle.load(file)

fname = 'kmeans_scaler.pckl'
with open(fname, 'rb') as file:
    scaler = pickle.load(file)

fname = 'kmeans_cluster_labels.pckl'
with open(fname, 'rb') as file:
    cluster_labels = pickle.load(file)

dd_coarse = xr.open_dataset('kmeans_ds_train.nc')

#=====================================================================
#   Plot clusters - Characteristics Synoptic Patterns from K-means
#=====================================================================
import cartopy
import cartopy.crs as ccrs
from matplotlib import patches

arr = dd_coarse.z_anomaly.values
#tmp = dd_coarse[['u','v']].coarsen(latitude=5, longitude=5).mean()
#arr_u = tmp.u.values
#arr_v = tmp.v.values
#lon_u = tmp.longitude.values
#lat_u = tmp.latitude.values

# plot all patterns
for cluster_id in [0,1]:  #np.unique(cluster_labels):
    # Select features belonging to the cluster
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    #cluster_features = np.transpose(arr, (1,2,3,0))[cluster_indices] + z_mean
    map_char = np.mean(arr[cluster_indices, :,:], axis=0)
    u_char = np.mean(arr_u[cluster_indices, :,:], axis=0)
    v_char = np.mean(arr_v[cluster_indices, :,:], axis=0)

    plt.figure(figsize=(12,6))
    p = plt.subplot(1,1, 1,projection=ccrs.PlateCarree())

    p.coastlines('50m', linewidth=1)
    p.set_extent([lon-50, lon+50, lat-30, lat+30], ccrs.PlateCarree())

    im1 = p.imshow(map_char, interpolation='nearest', cmap=plt.cm.Spectral_r, extent=[lon-50, lon+50, lat-30, lat+30], vmin=-150, vmax=150)
    p.contour(map_char, colors='k', levels=np.arange(-200, 200, 20), linewidths=2, alpha=0.3, extent=[lon-50, lon+50, lat-30, lat+30], origin='upper')
    #p.quiver(longitude, latitude, u_char, v_char)

    # add rectangle of the SC of interest
    p.add_patch(patches.Rectangle((22.41, 45.15), 4.14, 0.71, facecolor=(1,0,0,0), edgecolor='r'))
    plt.colorbar(im1)
    p.set_title(f'Pattern {cluster_id+1} of 30\nGeopotential Height Anomaly (m)')

    plt.tight_layout()
    plt.savefig(f'pattern_{str(cluster_id+1).zfill(2)}.png', dpi=150)



#=====================================================================
# Predict features on the entire timeseries of 500mb geopotential
#=====================================================================

# Load model from disk
fname = 'kmeans_model.pckl'
with open(fname, 'rb') as file:
    kmeans = pickle.load(file)

fname = 'kmeans_scaler.pckl'
with open(fname, 'rb') as file:
    scaler = pickle.load(file)

# write code to predict clusters in 10 years chunck.
for years in ['194*daily.nc',
    '195*daily.nc',
    '196*daily.nc',
    '197*daily.nc',
    '198*daily.nc',
    '199*daily.nc',
    '200*daily.nc',
    '201*daily.nc',
    '202*daily.nc'
              ]:
    print(f'---> Applying K-means model to {years}')
    ds = xr.open_mfdataset(f'data/PLEV_{years}')
    df = synoptic.kmeans_predict(ds, kmeans, scaler, var_clust='z_anomaly', lat_res=None, lon_res=None )
    df.to_pickle(f'df_clusters_{years.split("*")[0]}0s.pckl')

flist_df = glob.glob('df_clust*.pckl')

fl = []
for file in flist_df:
    fl.append(pd.read_pickle(file))
df = pd.concat(fl)