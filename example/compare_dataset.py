import xarray as xr
import glob
from TopoPyScale import topo_export as te

# crop E-Obs to SC domain to reduce size
flist = glob.glob('*0e.nc')

for file in flist:
    ds = xr.open_dataset(file)
    ds = ds.sel()

'sea_level_pressure', 'wind_spimport glob
import xarray as xr
from TopoPyScale import topo_export as te
from climatepy import fetch_climate as fu
# fetch data from CDS server
fu.fetch_E_Obs(varoi=['sea_level_pressure',
                      'wind_speed''mean_temperature',
                      'precipitation_amount',
                      'relative_humidity',
                      'surface_shortwave_downwelling_radiation'])

for var in ['tg', 'pp', 'qq', 'fg', 'hu']:
    ds = xr.open_dataset(f'{var}_ens_mean_0.1deg_reg_v27.0e.nc')
    dd = ds.sel(latitude=slice(44.5, 46.5), longitude=slice(21.3, 26.7))
    te.to_netcdf(dd, f'{var}_ens_mean_0.1deg_reg_v27.0e_crop.nc')
    ds, dd = None, None



flist = ['qq_ens_mean_0.1deg_reg_v27.0e_crop.nc',
         'pp_ens_mean_0.1deg_reg_v27.0e_crop.nc',
         'tg_ens_mean_0.1deg_reg_v27.0e_crop.nc',
         'rr_ens_mean_0.1deg_reg_v27.0e_crop.nc',
         'hu_ens_mean_0.1deg_reg_v27.0e_crop.nc']

ds = xr.open_dataset(flist[0])
for n in range(1,5):
    da = xr.open_dataarray(flist[n])
    ds[flist[n].split('_')[0]] = (xr.open_dataarray(flist[n]), da.values)


