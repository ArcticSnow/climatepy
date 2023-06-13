'''
Collection of functions usefull to climatic analysys of timeseries out of TopoPyScale
S. Filhol, March 2023
'''

#---------------------- Defining periods of reference for computing statistics ----------------------
import xarray as xr
import pandas as pd
from pyproj import Transformer

def resample_climate(ds, freq='1D', var_mean=['t', 'u', 'v', 'p', 'SW', 'LW'], var_sum=['tp', 'precip_lapse_rate']):
    '''
    Function to resmaple climate variable.
    Args:
        ds: dataset to be resampled
        freq: frequency at which to resample dataset
        var_mean: variable for which resampling is done by averaging, e.g. temperature
        var_sum: variable for which resampling is cumulative, e.g. precip

    Returns:
        dataset at resampled freq

    '''
    res = None
    if var_mean is not None:
        res = ds[var_mean].resample(time=freq).mean()
    if var_sum is not None:
        if res is not None:
            res = xr.merge([res, ds[var_sum].resample(time=freq).sum()])
        else:
            res = ds[var_sum].resample(time=freq).sum()

    print(f'Dataset resampled to {freq} frequency')
    return res


def read_fsm(file_pattern):
    '''
    Function to load FSM files into a dataset
    Args:
        file_pattern:

    Returns:

    '''

def read_pt_fsm(fname):
    '''
    Function to read FSM outputs with pandas.
    '''
    fsm = pd.read_csv(fname, 
                  delim_whitespace=True, header=None)
    fsm.columns = ['year', 'month', 'day', 'hour', 'albedo', 'runoff', 'snd', 'swe', 't_surface', 't_soil']
    fsm['time'] = pd.to_datetime(fsm.year.astype(str) +'-' +  fsm.month.astype(str) +'-' + fsm.day.astype(str))
    fsm.set_index('time', inplace=True)
    return fsm



def compute_reference_periods(obj, water_month_start=10):
    if isinstance(obj, pd.DataFrame):
        compute_reference_periods_df(obj, water_month_start)
        
    elif isinstance(obj, xr.Dataset):
        compute_reference_periods_ds(obj, water_month_start)
        

def compute_reference_periods_df(df, water_month_start=10, time_column='time'):
    ''' compute
    - water year
     - water month
     - seasons ['DJF', 'MAM', 'JJA', 'SON']
     - day of water year
     '''
    df['water_year'] = df.index.year.where(df.index.month < water_month_start, df.index.year + 1)
    df['water_start'] = df.water_year.apply(lambda x: pd.to_datetime(f'{x-1}-{water_month_start}-01', format='%Y-%m-%d'))
    df['water_doy'] = (df.index - df.water_start).dt.days
    
    return df

def compute_reference_periods_ds(ds, water_year_month_start = 10):
    '''
    Function to derive all reference periods used for climatic analysis relevant to the hydrological cycle
     - water year
     - water month
     - seasons ['DJF', 'MAM', 'JJA', 'SON']
    
    ds - xarray dataset. Must contain time coordinate
    water_year_month_start -  
    '''
    ds['water_year'] = ds.time.dt.year.where(ds.time.dt.month < water_year_month_start, ds.time.dt.year + 1)
    ds['month'] = ds.time.dt.month
    ds['water_month'] = (ds.month - water_year_month_start) % 12 + 1
    ds['season'] = ds.time.dt.season
    # winter_d1 is defined as winter from [Oct to April]
    #ds['season_d1'] = xr.where((ds.month >= 10) |  (ds.month<=4), 'winter', 'summer')


def resampling_meteo(ds, frequency='1D'):
    '''
    Function to resample meteorological variables to given frequency (e.g. daily). Differentiate between cumulative variables (radiations, precipitations) to averaging ones
    '''
    daily = ds[['t', 'ws', 'p']].resample(time=frequency).mean()
    daily[['LW', 'SW','tp']] = ds[['LW', 'SW','tp']].resample(time=frequency).sum()
    
    return daily

def resampling_daily_xclim(ds):
    daily = resampling_meteo(ds, frequency='1D')
    daily['tasmin'] = ds['t'].resample(time='1D').min()
    daily['tasmax'] = ds['t'].resample(time='1D').max()
    
    return daily

def convert_epsg_pts(xs,ys, epsg_src=4326, epsg_tgt=3844):
    """
    Simple function to convert a list fo poitn from one projection to another oen using PyProj
    Args:
        xs (array): 1D array with X-coordinate expressed in the source EPSG
        ys (array): 1D array with Y-coordinate expressed in the source EPSG
        epsg_src (int): source projection EPSG code
        epsg_tgt (int): target projection EPSG code
    Returns: 
        array: Xs 1D arrays of the point coordinates expressed in the target projection
        array: Ys 1D arrays of the point coordinates expressed in the target projection
    """
    print('Convert coordinates from EPSG:{} to EPSG:{}'.format(epsg_src, epsg_tgt))
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:{}".format(epsg_tgt), always_xy=True)
    Xs, Ys = trans.transform(xs, ys)
    return Xs, Ys

def open_dataset_climate(flist, concat_dim='point_id'):
    ds__list = []
    for file in flist:
        ds__list.append(xr.open_dataset(file))
    ds_ = xr.concat(ds__list, dim=concat_dim)
    return ds_
    

