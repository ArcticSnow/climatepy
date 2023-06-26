'''
Python Class to work with observation and downscaled timeseries


'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import colors
import numpy as np
import climatepy.climate_utils as cu
import climatepy.isd_NOAA as noa
import xarray as xr
from pathlib import Path
import os

import xarray as xr
from xarrayMannKendall import xarrayMannKendall as xm
from TopoPyScale import topo_sim as ts

class clustered():
    def __init__(self,
                fname_pattern,
                ds_param_file=None,
                 water_month_start=9,
                 var_mean=None,
                 var_sum= None,
                 var_min=None,
                 var_max=None,
                 parallel=False,
                 daily_ds=False):

        self.fname_pattern=fname_pattern
        self.water_month_start = water_month_start
        self.var_mean = var_mean
        self.var_sum = var_sum
        self.var_min = var_min
        self.var_max = var_max

        # open and load dataset
        if ds_param_file is not None:
            self.ds_param = xr.open_dataset(ds_param_file)

        if daily_ds:
            self.daily = xr.open_mfdataset(self.fname_pattern, concat_dim='point_id', combine='nested', parallel=parallel)
        else:
            self.ds = xr.open_mfdataset(self.fname_pattern, concat_dim='point_id', combine='nested', parallel=parallel)
            self.daily = cu.resample_climate(self.ds, freq='1D', var_mean=var_mean, var_sum=var_sum, var_min=var_min, var_max=var_max)

        cu.compute_reference_periods(self.daily, water_month_start=self.water_month_start)

        print('---> Data loaded')

    def compute_FDD(self):
        #algo to compute FDD
        return

    def compute_snow_period(self, snow_var, ndays=4, with_snow=True, snd_thresh=0.1, n_core=4):

        from TopoPyScale import topo_utils as tu

        df = fsm.DJF.to_dataframe()
        df.reset_index(inplace=True)

        df_list = []
        fname = []
        for pt in df.point_id.unique():
            df_list.append(df.snd.loc[df.point_id==pt])
            fname.append(f'')

        n = len(df.point_id.unique())
        fun_param = zip(df_list, [snow_var]*n, [with_snow]*n, [snd_thresh]*n)
        tu.multicore_pooling(cu.find_periods, fun_param, n_core)





    def agg_seasonal(self, var_mean=None, var_sum=None, var_min=None, var_max=None, hydrological_year=True,):

        if hydrological_year is False:
            year_ref = 'year'
        else:
            year_ref = 'water_year'

        if var_sum is None:
            var_sum = self.var_sum
        if var_mean is None:
            var_mean = self.var_mean
        if var_min is None:
            var_min = self.var_min
        if var_max is None:
            var_max = self.var_max

        DJF_list = []
        MAM_list = []
        JJA_list = []
        SON_list = []

        if var_mean is not None:
            DJF_list.append(self.daily.where(self.daily.season=='DJF')[var_mean].groupby(self.daily[year_ref]).mean(dim='time'))
            MAM_list.append(self.daily.where(self.daily.season=='MAM')[var_mean].groupby(self.daily[year_ref]).mean(dim='time'))
            JJA_list.append(self.daily.where(self.daily.season=='JJA')[var_mean].groupby(self.daily[year_ref]).mean(dim='time'))
            SON_list.append(self.daily.where(self.daily.season=='SON')[var_mean].groupby(self.daily[year_ref]).mean(dim='time'))

        if var_sum is not None:
            DJF_list.append(self.daily.where(self.daily.season=='DJF')[var_sum].groupby(self.daily[year_ref]).sum(dim='time'))
            MAM_list.append(self.daily.where(self.daily.season=='MAM')[var_sum].groupby(self.daily[year_ref]).sum(dim='time'))
            JJA_list.append(self.daily.where(self.daily.season=='JJA')[var_sum].groupby(self.daily[year_ref]).sum(dim='time'))
            SON_list.append(self.daily.where(self.daily.season=='SON')[var_sum].groupby(self.daily[year_ref]).sum(dim='time'))

        if var_min is not None:
            DJF_list.append(self.daily.where(self.daily.season=='DJF')[var_min].groupby(self.daily[year_ref]).min(dim='time'))
            MAM_list.append(self.daily.where(self.daily.season=='MAM')[var_min].groupby(self.daily[year_ref]).min(dim='time'))
            JJA_list.append(self.daily.where(self.daily.season=='JJA')[var_min].groupby(self.daily[year_ref]).min(dim='time'))
            SON_list.append(self.daily.where(self.daily.season=='SON')[var_min].groupby(self.daily[year_ref]).min(dim='time'))

        if var_max is not None:
            DJF_list.append(self.daily.where(self.daily.season=='DJF')[var_max].groupby(self.daily[year_ref]).max(dim='time'))
            MAM_list.append(self.daily.where(self.daily.season=='MAM')[var_max].groupby(self.daily[year_ref]).max(dim='time'))
            JJA_list.append(self.daily.where(self.daily.season=='JJA')[var_max].groupby(self.daily[year_ref]).max(dim='time'))
            SON_list.append(self.daily.where(self.daily.season=='SON')[var_max].groupby(self.daily[year_ref]).max(dim='time'))

        self.DJF = xr.merge(DJF_list)
        self.MAM = xr.merge(MAM_list)
        self.JJA = xr.merge(JJA_list)
        self.SON = xr.merge(SON_list)

        DJF_list = None
        MAM_list = None
        JJA_list = None
        SON_list = None

        print('---> Seasonal aggregation completed')


    def agg_annual(self, var_mean=None, var_sum=None, var_min=None, var_max=None, hydrological_year=True, agg='mean'):

        if hydrological_year is False:
            year_ref = 'year'
        else:
            year_ref = 'water_year'

        if var_sum is None:
            var_sum = self.var_sum
        if var_mean is None:
            var_mean = self.var_mean
        if var_min is None:
            var_min = self.var_min
        if var_max is None:
            var_max = self.var_max

        annual_list = []
        if  var_mean is not None:
            annual_list.append(self.daily[var_mean].groupby(self.daily[year_ref]).mean(dim='time'))
        if  var_sum is not None:
            annual_list.append(self.daily[var_sum].groupby(self.daily[year_ref]).sum(dim='time'))
        if  var_min is not None:
            annual_list.append(self.daily[var_min].groupby(self.daily[year_ref]).min(dim='time'))
        if var_max is not None:
            annual_list.append(self.daily[var_max].groupby(self.daily[year_ref]).max(dim='time'))

        self.annual = xr.merge(annual_list)
        annual_list = None
        print('---> Annual aggregation completed')


    def mann_kendall(self, da, rename_dict={'water_year':'time', 'longitude': 'x', 'latitude':'y'}, p_value=0.05):
        MK_class = xm.Mann_Kendall_test(da.rename(rename_dict), dim='time', alpha=p_value, method='theilslopes')
        invert_dict = {v: k for k, v in rename_dict.items()}
        invert_dict.pop('time')
        MK_trends = xr.merge([da, MK_class.compute().rename(invert_dict)])
        return MK_trends

    def map_stat(self, ds, var=None):
        if type(ds) is xr.DataArray:
            ds = ds.to_dataset()

        if len(list(ds.keys()))==1:
            var = list(ds.keys())[0]

        return ds[var].sel(point_id=self.ds_param.cluster_labels)

    def plot_map_stat(self,
                 ds,
                 var=None,
                 ax=None,
                 cmap=plt.cm.RdBu_r,
                 hillshade=True,
                 **kwargs):

        if ax is None:
            fig, ax = plt.subplots(1,1)

        if type(ds) is xr.DataArray:
            ds = ds.to_dataset()

        alpha=1
        if hillshade:
            ls = LightSource(azdeg=315, altdeg=45)
            shade = ls.hillshade(self.ds_param.elevation.values, vert_exag=0.5,
                                 dx=self.ds_param.x.diff('x')[0].values,
                                 dy=self.ds_param.y.diff('y')[0].values,
                                 fraction=1.0)
            ax.imshow(shade,
                       extent=[self.ds_param.x.min(), self.ds_param.x.max(), self.ds_param.y.min(), self.ds_param.y.max()],
                       cmap=plt.cm.gray)
            alpha=0.5

        if len(list(ds.keys()))==1:
            var = list(ds.keys())[0]
        im = ax.imshow(ds[var].sel(point_id=self.ds_param.cluster_labels),
                  extent=[self.ds_param.x.min(), self.ds_param.x.max(), self.ds_param.y.min(), self.ds_param.y.max()],
                  alpha=alpha,
                  cmap=cmap,
                  **kwargs)

        return ax, im









class station():
    def __init__(self, 
                 path_toposcale_project=None, 
                 point_id=0,
                path_noaa=None,
                file_met_office=None,
                 water_month_start=9):
        self.point_id = point_id
        self.path_toposcale_project = path_toposcale_project
        self.path_noaa = path_noaa
        self.file_met_office = file_met_office
        self.water_month_start = water_month_start

        
        if not os.path.isdir(path_toposcale_project):
            raise('ERROR: Path to TopoPyScale project does not exist')
        if not os.path.isdir(path_noaa):
            raise('ERROR: Path to NOAA dataset does not exist')
        
        self.load_meta()
        
        self.obs = observation(path_noaa = self.path_noaa, 
                               stn_id=self.station_id, 
                               file_met_office=self.file_met_office,
                               water_month_start = self.water_month_start,
                               )
        self.down = downscaled(self.path_toposcale_project, point_id)
        self.avail_dataset = {'obs': self.obs.dataset,
                             'down': self.down.dataset}
    
    def load_meta(self):
        df_path = Path(self.path_toposcale_project, 'outputs','df_centroids.pck')
        df = pd.read_pickle(df_path).iloc[self.point_id]
        self.name = df.NUME
        self.latitude = df.latitude
        self.longitude = df.longitude
        self.topography = df
        self.station_id = df.CODST
        print('---> Station metadata loaded')
        
        
    def compute_reference_periods(self, water_month_start=None):
        if water_month_start is not None:
            self.water_month_start = water_month_start
        print('---> Computing reference periods ...')
        if hasattr(self.obs,'noaa'):
            cu.compute_reference_periods(self.obs.noaa, water_month_start=self.water_month_start)

        if hasattr(self.obs,'met_office'):
            cu.compute_reference_periods(self.obs.met_office, water_month_start=self.water_month_start)
        
        if hasattr(self.down,'toposcale'):
            cu.compute_reference_periods(self.down.toposcale, water_month_start=self.water_month_start)
        
        if hasattr(self.down,'fsm'):
            cu.compute_reference_periods(self.down.fsm, water_month_start=self.water_month_start)
        print('Refence periods computed for all available datasets')
        
    def merge_datasets(self, d1, d2, **kwargs):
        # add check that timestep is equivalent for both datasets.
        return pd.merge(d1, d2, how='inner', **kwargs)


class downscaled():
    def __init__(self, path_toposcale_project, point_id):
        
        self.file_fsm_sim = Path(path_toposcale_project, 'fsm_sims', f'sim_FSM_pt_{str(point_id).zfill(2)}.txt')
        self.file_toposcale = Path(path_toposcale_project, 'outputs', 'downscaled', f'down_pt_{str(point_id).zfill(2)}.nc')
        self.dataset = []

        if os.path.isfile(self.file_fsm_sim):
            print('---> FSM simulation file found')
            self.load_fsm()
            self.dataset.append('fsm')
            
              
        if os.path.isfile(self.file_toposcale):
            print('---> TopoPyScale file found')
            self.load_downscaled()
            self.dataset.append('toposcale')

    def load_downscaled(self):
        ds = xr.open_dataset(self.file_toposcale)
        self.toposcale = ds.to_dataframe()

    def load_fsm(self):
        self.fsm = cu.read_pt_fsm(self.file_fsm_sim)
          

class observation():
    
    def __init__(self, file_met_office=None, path_noaa=None, stn_id=None, water_month_start=None):
        import fnmatch
        
        self.file_met_office = file_met_office
        self.dataset = []
        self.water_month_start = water_month_start
        
        if path_noaa is not None:
            for file in os.listdir(path_noaa):
                if fnmatch.fnmatch(file, f'NOAA_{stn_id}*.csv'):
                    self.file_noaa = file
                    self.load_NOAA_data(Path(path_noaa,file))
                    print('---> NOAA file found')
                    self.dataset.append('noaa')
                
        if file_met_office is not None:
            if Path(self.file_met_office).is_file():
                print('---> Met Office Obs. file found')
                self.load_romania_met_office_data()
                self.dataset.append('met_office')
            
        


    def load_romania_met_office_data(self):
        to = pd.read_csv(self.file_met_office)
        to['date'] = pd.to_datetime(to.DAT, format='%d/%m/%Y %H:%M')
        to = to.rename(columns={'GROSZ':'SD'})
        to.set_index(to.date, inplace=True)
        to = to.drop(columns=['COD', 'DENST', 'Unnamed: 0', 'date', 'DAT'])

        #cu.compute_reference_periods(to, water_month_start=self.water_month_start)
        to.SD = to.SD / 100  #convert cm to m
        self.met_office = to

    def load_NOAA_data(self, file):
        
        self.noaa = noa.read_isd_csv(file,
                                     var=['tmp', 'wnd', 'precip', 'rh', 'sd', 'slp'], 
                                     keep_cols=noa.keep_cols_default,
                                     index_timestamp=True)
class seasonal():
    def __init__(self, df, var='t', name=None, water_month_start=9):
        self.df = df
        self.var = var
        self.name = name
        self.window_days = None
        self.window_years = None
        self.water_month_start = water_month_start

        if 'water_year' not in self.df.columns:
            cu.compute_reference_periods(df, water_month_start=self.water_month_start)
        self.t_seasonal_raw = self.df.groupby(['water_year', 'water_doy'])[var].mean().unstack().T

    def compute_anomaly(self, window_days=5, window_years=10, reference_period=None):
        '''
        Function to compute anomaly grid
        Args:
            window_days (int): rolling median along days-direction. Good practice to provide an odd number
            window_years (int): rolling median along years-direction. Good practice to provide an odd number
            reference_period (list): ['1970-09-01', '2000-08-31']

        '''
        if reference_period is None:
            reference_period = [self.df.index[0], self.df.index[-1]]

        self.reference_period = reference_period
        self.window_days = window_days
        self.window_years = window_years

        self.df[f'{self.var}_{self.window_days}D_mean'] = self.df[self.var][self.reference_period[0]: self.reference_period[1]].rolling(self.window_days).mean()
        ref_var = self.df.groupby(['water_year', 'water_doy'])[f'{self.var}_{self.window_days}D_mean'].mean().unstack().T
        yearly_mean = ref_var.mean(axis=1)

        self.t_seasonal_smooth = self.df.groupby(['water_year', 'water_doy'])[f'{self.var}'].mean().unstack().T
        self.t_seasonal_anomaly = self.t_seasonal_smooth.subtract(yearly_mean, axis=0)
        self.t_reference = ref_var


    def plot_freezing_change(self):
        '''Needs to be adapted'''

        fig, ax = plt.subplots(3,1,sharex=True,figsize=(12,8), gridspec_kw={'height_ratios': [1.2, 1.2, 0.5], 'hspace':0.1})
        tu.mean(axis=0).plot(label='mean $T_{air}$', ax=ax[0])
        tu.min(axis=0).plot(label='min $T_{air}$', ax=ax[0])
        tu.max(axis=0).plot(label='max $T_{air}$', ax=ax[0])
        ax[0].set_ylabel('$T_{air}$ [$^{o}C$]')
        ax[1].set_ylabel('Number of days above 0$^{o}C$')
        tu.iloc[-1].plot(c='r', label='2022/2023', ax=ax[0])
        tu.iloc[-2].plot(c='r', label='2021/2022', ax=ax[0], alpha=0.3)

        ((tu>=0).sum(axis=0)/tu.count(axis=0)).rolling(7, center=True).mean().plot(ax=ax[1], alpha=0.5, linestyle=':', label='1974-2023')
        ((tu.iloc[:31]>=0).sum(axis=0)/tu.iloc[:31].count(axis=0)).rolling(7, center=True).mean().plot(ax=ax[1], label='1974-2004')
        ((tu.iloc[31:]>=0).sum(axis=0)/tu.iloc[31:].count(axis=0)).rolling(7, center=True).mean().plot(ax=ax[1], label='2005-2023')

        ax[1].set_ylabel('$P_{T>0}$')
        ax[1].set_xticks(ti.water_doy.values, labels=ti.index.strftime('%b'))
        ax[0].legend(loc='lower right')
        ax[1].legend()
        ax[1].set_yticks([0,0.25, 0.5, 0.75, 1], ['0','25%','50%','75%', '100%'])

        first = ((tu.iloc[:31]>=0).sum(axis=0)/tu.iloc[:31].count(axis=0)).rolling(7, center=True).mean()
        second = ((tu.iloc[31:]>=0).sum(axis=0)/tu.iloc[31:].count(axis=0)).rolling(7, center=True).mean()

        (second - first).rolling(15, center=True).mean().plot(ax=ax[2])
        ax[2].set_ylabel('Diff $P_{T>0}$')
        ax[2].set_yticks([0,0.1,0.2, 0.3], ['0', '10%', '20%', '30%'])
        ax[2].set_xlabel(' ')


    def plot_seasonal(self, series='t_seasonal', vmin=-20, vmax=10, vcenter=None, cmap=plt.cm.RdBu_r, window_years=None, axis=None):
        if vcenter is None:
            vcenter = (vmax-vmin)/2 + vmin
        if axis is None:
            fig, ax = plt.subplots(1,1)
        else:
            ax = axis

        cnorm = colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)

        if window_years is not None:
            self.window_years = window_years

        if self.window_years is not None:
            tu = getattr(self, series).rolling(self.window_years, center=True, axis=1).mean()
        else:
            tu = getattr(self, series)

        # Create reference period to tick labels
        ti = pd.DataFrame()
        ti['date'] = pd.to_datetime(pd.date_range(self.df.index[0], self.df.index[-1] + pd.offsets.MonthBegin(),
                      freq='M', inclusive='both') + pd.Timedelta('1D'))
        ti.set_index(ti.date, inplace=True)
        cu.compute_reference_periods(ti, water_month_start=self.water_month_start)

        im = ax.imshow(tu,
                   aspect='auto',cmap=cmap, interpolation='nearest', norm=cnorm,
                   extent=[self.df.index[0].year-0.5, self.df.index[-1].year+0.5, 366,0])
        plt.colorbar(im)
        ax.set_yticks(ti.water_doy.values, labels=ti.index.strftime('%b'))





class snow():
    def __init__(self, df, snow_var, name=None, water_month_start=9, reference_period=None):
        self.df = df
        self.snow_var = snow_var
        self.name = name
        self.water_month_start = water_month_start
        if reference_period is None:
            self.reference_period = [self.df.index[0], self.df.index[-1]]
        else:
            self.reference_period = reference_period

        if 'water_year' not in self.df.columns:
            cu.compute_reference_periods(self.df, water_month_start=self.water_month_start)
    
    def plot_snow_depths(self, df=None, ylim=[0,400], axis=None, c_median='r'):
        if axis is None:
            fig, ax = plt.subplots(1,1)
        else:
            ax = axis

        if df is None:
            df = self.df

        ti = pd.DataFrame()
        ti['date'] = pd.to_datetime(pd.date_range(self.df.index[0], self.df.index[-1] + pd.offsets.MonthBegin(),
                      freq='M', inclusive='both') + pd.Timedelta('1D'))
        ti.set_index(ti.date, inplace=True)
        cu.compute_reference_periods(ti, water_month_start=self.water_month_start)

        year_list = df.water_year.unique()
        for i, year in enumerate(year_list):
            if i==0:
                ax.plot(df.loc[df.water_year==year].water_doy, df.loc[df.water_year==year][self.snow_var], alpha=0.2, c='k',
                        label=f'{year_list.min()}-{year_list.max()} Snow depths')
            else:
                ax.plot(df.loc[df.water_year==year].water_doy, df.loc[df.water_year==year][self.snow_var], alpha=0.2, c='k')

        df.groupby(df.water_doy).median()[self.snow_var].plot(c=c_median, label=f'{year_list.min()}-{year_list.max()} Median snow depth', ax=ax)
        ax.set_ylim(ylim)
        ax.legend()
        ax.set_xlabel('Days from September 1')
        ax.set_ylabel('Snow depth [cm]')
        ax.set_xticks(ti.water_doy.values, labels=ti.index.strftime('%b'))
        
    def plot_year_vs_month(self, axis=None, cmap=plt.cm.viridis, **kwargs):
        if axis is None:
            fig, ax = plt.subplots(1,1)
        else:
            ax = axis

        ti = pd.DataFrame()
        ti['date'] = pd.to_datetime(pd.date_range(self.df.index[0], self.df.index[-1] + pd.offsets.MonthBegin(),
                      freq='M', inclusive='both') + pd.Timedelta('1D'))
        ti.set_index(ti.date, inplace=True)
        cu.compute_reference_periods(ti, water_month_start=self.water_month_start)

        mon = self.df.groupby([self.df.water_doy, self.df.water_year]).mean()
        year_list = self.df.water_year.unique()
        im = ax.imshow(mon[self.snow_var].unstack(),
                       aspect='auto',
                       cmap=cmap,
                       interpolation='nearest',
                       extent=[self.df.index[0].year-0.5, self.df.index[-1].year+0.5, 366,0], **kwargs)

        plt.colorbar(im)
        ax.set_yticks(ti.water_doy.values, labels=ti.index.strftime('%b'))
        ax.set_xlabel('Hydrological year')
        ax.set_title(f'Daily Mean Snow Depth at {self.name}')



    # Methods to compute snow season length
    def fill_with_hard_limit(self,
                             df_or_series,
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


    def find_periods(self, ndays=4, with_snow=True, snd_thresh=0.1):
        '''

        Args:
            ndays: Maximum number of days separating two periods
            with_snow:      - True: find periods of contiguous snow cover of at least ndays
                            - False: find periods of snow cover where the ground is snow free for a maximum period of ndays
            snd_thresh (float): snow depth threshold above which snow is detected. Unit, same as input data
            water_month_start:

        Returns:

        '''

        de = self.df[self.snow_var].copy()

        if with_snow:
            de.loc[de>snd_thresh] = np.nan
            de.loc[de<=snd_thresh] = 1
            a = -1
        else:
            de.loc[de>snd_thresh] = 1
            de.loc[de<=snd_thresh] = np.nan
            a = 1

        de = self.fill_with_hard_limit(df_or_series=de, limit=ndays, fill_method='interpolate')
        de.loc[np.isnan(de)] = 0
        de = a * de.diff()

        df_periods = pd.DataFrame()
        df_periods['start'] = de.loc[de==1].index
        df_periods['end'] = de.loc[de==-1].index
        df_periods['duration'] = (df_periods.end - df_periods.start)

        df_periods.set_index(df_periods.start, inplace=True)
        cu.compute_reference_periods(df_periods, water_month_start=self.water_month_start)


        med_list = []
        for i, row in df_periods.iterrows():
            med_list.append(self.df.loc[row.start.strftime('%Y-%m-%d'):row.end.strftime('%Y-%m-%d')].median())
        df_periods['snd_median'] = med_list
        return df_periods
        
        
        