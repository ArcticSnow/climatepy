'''
Collection of functions usefull to climatic analysys of timeseries out of TopoPyScale
S. Filhol, March 2023
'''

#---------------------- Defining periods of reference for computing statistics ----------------------
import xarray as xr
import pandas as pd
from pyproj import Transformer
import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mproc
import pdb

def multicore_pooling(fun, fun_param, n_cores):
    '''
    Function to perform multiprocessing on n_cores
    Args:
        fun (obj): function to distribute
        fun_param zip(list): zip list of functino arguments
        n_core (int): number o cores
    '''
    if n_cores is None:
        n_cores = mproc.cpu_count() - 2
        print(f'WARNING: number of cores to use not provided. By default {n_cores} cores will be used')
    elif n_cores > mproc.cpu_count():
        n_cores = mproc.cpu_count() - 2
        print(f'WARNING: Only {mproc.cpu_count()} cores available on this machine, reducing n_cores to {n_cores} ')

    # make sure it will run on one core at least
    if n_cores == 0:
        n_cores = 1

    pool = Pool(n_cores)
    pool.starmap(fun, fun_param)
    pool.close()
    pool.join()
    pool = None


def multithread_pooling(fun, fun_param, n_threads):
    '''
    Function to perform multiprocessing on n_threads
    Args:
        fun (obj): function to distribute
        fun_param zip(list): zip list of functino arguments
        n_core (int): number of threads
    '''
    tpool = ThreadPool(n_threads)
    tpool.starmap(fun, fun_param)
    tpool.close()
    tpool.join()
    tpool = None

def resample_climate(ds, freq='1D',
                     var_mean=['t', 'u', 'v', 'p', 'SW', 'LW'],
                     var_sum=['tp', 'precip_lapse_rate'],
                     var_min=None,
                     var_max=None):
    '''
    Function to resample climate variable.
    Args:
        ds: dataset to be resampled
        freq: frequency at which to resample dataset
        var_mean: variable for which resampling is done by averaging, e.g. temperature
        var_sum: variable for which resampling is cumulative, e.g. precip

    Returns:
        dataset at resampled freq

    '''
    res_mean = None
    res_min = None
    res_max = None
    res_sum = None

    if var_mean is not None:
        res_mean = ds[var_mean].resample(time=freq).mean()
        res_mean = res_mean.rename(dict(zip(var_mean, [var+'_mean' for var in var_mean])))
    if var_sum is not None:
        res_sum = ds[var_sum].resample(time=freq).sum()
        res_sum = res_sum.rename(dict(zip(var_sum, [var+'_sum' for var in var_sum])))
    if var_min is not None:
        res_min = ds[var_min].resample(time=freq).min()
        res_min= res_min.rename(dict(zip(var_min, [var+'_min' for var in var_min])))
    if var_max is not None:
        res_max = ds[var_max].resample(time=freq).max()
        res_max = res_max.rename(dict(zip(var_max, [var+'_max' for var in var_max])))

    res_list = []
    for r in [res_mean, res_sum, res_min, res_max]:
        if r is not None:
            res_list.append(r)
    res = xr.merge(res_list)

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

def compute_normal(df, varoi='Tair',
                   normal_period=['1991-01-01','2020-12-31'],
                   daily_agg='mean',
                   quantiles=[0.1, 0.25, 0.75, 0.9],
                   groupby_var=['ref_doy', 'ref_year']):
    """
    Function to compute monthly normal at daily resolution

    Args:
        quantiles:
        df (dataframe): Dataframe with timeseires (index must be datetime)
        varoi (str): Variable to compute normal
        normal_period (str): list of start and end dates. Default ['1991-01-01','2020-12-31']
        ref_month_start: Month to start the period on which to return normal timeseries. Default is January
        method (str): daily aggregation method. Default='mean'

    Returns:
        dataseries - daily normals
        dataframe - dataframe with reference periods
        dataframe - dataframe with daily aggregated values for the period of reference

    """

    if daily_agg=='mean':
        df_norm = df[normal_period[0]:normal_period[1]][groupby_var + [varoi]].groupby(groupby_var).mean()[varoi].unstack()
    elif daily_agg=='sum':
        df_norm = df[normal_period[0]:normal_period[1]][groupby_var + [varoi]].groupby(groupby_var).sum()[varoi].unstack()
    elif daily_agg=='max':
        df_norm = df[normal_period[0]:normal_period[1]][groupby_var + [varoi]].groupby(groupby_var).max()[varoi].unstack()
    elif daily_agg=='min':
        df_norm = df[normal_period[0]:normal_period[1]][groupby_var + [varoi]].groupby(groupby_var).min()[varoi].unstack()

    def rolling_monthly_mean(arr):
        da =  pd.DataFrame()
        da['norm'] = np.concatenate([arr,arr,arr])
        # 2. Concatenate three time the yearly time series for then computing the a 31 days rolling mean
        return da.norm.rolling(31, center=True).mean()[366:366+366].reset_index().norm

    # Compute a monthly anomaly statistics  with a daily resolution
    normals=pd.DataFrame()
    normals['mean'] = rolling_monthly_mean(df_norm.mean(axis=1).values)
    normals['min'] = rolling_monthly_mean(df_norm.min(axis=1).values)
    normals['max'] = rolling_monthly_mean(df_norm.max(axis=1).values)
    for q in quantiles:
        normals[f'q{int(q*100)}'] = rolling_monthly_mean(df_norm.quantile(q, axis=1).values)

    return normals, df_norm

def compute_reference_periods(obj, ref_month_start=10, year_offset=0):
    """
    Function to compute reference periods using a reference month. For instance a typical water year would start in September (9)

    Args:
        obj: Dataset or dataframe
        ref_month_start:

    Returns:

    """
    if isinstance(obj, pd.DataFrame):
        return compute_reference_periods_df(obj, ref_month_start, year_offset=year_offset)
        
    elif isinstance(obj, xr.Dataset):
        compute_reference_periods_ds(obj, ref_month_start, year_offset=year_offset)
        

def compute_reference_periods_df(df, ref_month_start=10, year_offset=0):
    """
    Function to derive all reference periods used for climatic analysis in respect to whatever reference mont. Hydrological year ref_month_start=10 or 9

    Args:
        df: xarray dataset. Must contain time coordinate
        ref_month_start: reference month to start the referenc year. For instance an hydrological year starting in October, ref_month_start=10

    Returns:
     - ref_year: start year of the period of refetence
     - period: explicit years covered by the period
     - ref month_start: starting month of the reference year
     - ref_doy: reference day of year
     - seasons ['DJF', 'MAM', 'JJA', 'SON']
    """
    ref_year = df.index.year.where(df.index.month >= ref_month_start, df.index.year - 1)
    df['ref_year'] = ref_year
    if ref_month_start != 1:
        df = df.assign(period= df.ref_year.astype(str) + '-' + (df.ref_year+1).astype(str))
    else:
        df['period'] = df.ref_year.astype(str)
    df['tmp'] = df.ref_year.values.astype(str)
    df['ref_start'] = pd.to_datetime(df.assign(tmp2 = df.tmp + f'-{ref_month_start}-1').tmp2)
    df['ref_doy'] = (df.index - df.ref_start).dt.days + 1
    #df['ref_moy'] = (df.index - df.ref_start).dt.month + 1
    df = df.drop(columns=['tmp'])
    df['season']=((df.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

    
    return df

def compute_reference_periods_ds(ds, ref_month_start = 10):
    """
    Function to derive all reference periods used for climatic analysis in respect to whatever reference mont. Hydrological year ref_month_start=10 or 9

    Args:
        ds: xarray dataset. Must contain time coordinate
        ref_month_start: reference month to start the referenc year. For instance an hydrological year starting in October, ref_month_start=10

    Returns:
     - ref year
     - ref month
     - seasons ['DJF', 'MAM', 'JJA', 'SON']
    """

    ds['ref_year'] = ds.time.dt.year.where(ds.time.dt.month >= ref_month_start, ds.time.dt.year)
    if ref_month_start != 1:
        ds = ds.assign(period= ds.ref_year.astype(str) + '-' + (ds.ref_year+1).astype(str))
    else:
        ds['period'] = ds.ref_year.astype(str)
    ds['month'] = ds.time.dt.month
    ds['ref_month'] = (ds.month - ref_month_start) % 12 + 1
    ds['season'] = ds.time.dt.season

    ds['tmp'] = (('time'),(ds.ref_year.values-1).astype(str))
    ds['ref_start'] = (('time'), pd.to_datetime(ds.tmp.str.cat(f'-{ref_month_start}-1')))
    ds['ref_doy'] = (ds.time - ds.ref_start).dt.days + 1
    ds = ds.drop('tmp')


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



def fill_with_hard_limit(df_or_series,
                         limit: int,
                         fill_method='interpolate',
                         **fill_method_kwargs):
    """
    The fill methods from Pandas such as ``interpolate`` or ``bfill``
    will fill ``limit`` number of NaNs, even if the total number of
    consecutive NaNs is larger than ``limit``. This function instead
    does not fill any data when the number of consecutive NaNs
    is > ``limit``.

    Adapted from: https://stackoverflow.com/a/30538371/11052174

    :param df_or_series: DataFrame or Series to perform interpolation on.
    :param limit: Maximum number of consecutive NaNs to allow. Any occurrences of more consecutive NaNs than ``limit`` will have no
        filling performed.
    :param fill_method: Filling method to use, e.g. 'interpolate', 'bfill', etc.
    :param fill_method_kwargs: Keyword arguments to pass to the fill_method, in addition to the given limit.

    :returns: A filled version of the given df_or_series according
        to the given inputs.

    From: https://stackoverflow.com/a/66373000/1367097
    """
    # Keep things simple, ensure we have a DataFrame.
    try:
        df = df_or_series.to_frame()
    except AttributeError:
        df = df_or_series

    mask = pd.DataFrame(True, index=df.index, columns=df.columns)      # Initialize our mask.
    grp = (df.notnull() != df.shift().notnull()).cumsum()     # Get cumulative sums of consecutive NaNs.
    grp['ones'] = 1    # Add columns of ones.

    # Loop through columns and update the mask.
    for col in df.columns:
        mask.loc[:, col] = (
                (grp.groupby(col)['ones'].transform('count') <= limit)
                | df[col].notnull()
        )

    # Now, interpolate and use the mask to create NaNs for the larger gaps.
    method = getattr(df, fill_method)
    out = method(limit=limit, **fill_method_kwargs)[mask]

    # Be nice to the caller and return a Series if that's what they provided.
    if isinstance(df_or_series, pd.Series):
        # Return a Series.
        return out.loc[:, out.columns[0]]
    return out


def find_periods(df, snow_var, ndays=4, with_snow=True, snd_thresh=0.1, save_to_file=None, water_month_start=9):
    '''

    Args:
        ndays: Maximum number of days separating two periods
        with_snow:      - True: find periods of contiguous snow cover of at least ndays
                        - False: find periods of snow cover where the ground is snow free for a maximum period of ndays
        snd_thresh (float): snow depth threshold above which snow is detected. Unit, same as input data
        water_month_start:

    Returns:

    '''

    de = df[snow_var].copy()

    if with_snow:
        de.loc[de>snd_thresh] = np.nan
        de.loc[de<=snd_thresh] = 1
        a = -1
    else:
        de.loc[de>snd_thresh] = 1
        de.loc[de<=snd_thresh] = np.nan
        a = 1

    de = fill_with_hard_limit(df_or_series=de, limit=ndays, fill_method='interpolate')
    de.loc[np.isnan(de)] = 0
    de = a * de.diff()

    df_periods = pd.DataFrame()
    df_periods['start'] = de.loc[de==1].index
    df_periods['end'] = de.loc[de==-1].index
    df_periods['duration'] = (df_periods.end - df_periods.start)

    df_periods.set_index(df_periods.start, inplace=True)
    compute_reference_periods(df_periods, water_month_start=water_month_start)


    med_list = []
    mean_list = []
    for i, row in df_periods.iterrows():
        med_list.append(df.loc[row.start.strftime('%Y-%m-%d'):row.end.strftime('%Y-%m-%d')][snow_var].median())
        mean_list.append(df.loc[row.start.strftime('%Y-%m-%d'):row.end.strftime('%Y-%m-%d')][snow_var].mean())
    df_periods['snd_median'] = med_list
    df_periods['snd_mean'] = mean_list
    if save_to_file is None:
        return df_periods
    else:
        df_periods.to_pickle(save_to_file)
