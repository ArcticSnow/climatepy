'''
Python Class to work with observation and downscaled timeseries


'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import climate_utils as cu
import xarray as xr
from pathlib import Path
import os

class station():
    def __init__(self, 
                 path_toposcale_project=None, 
                 point_id=0,
                path_noaa=None,
                file_met_office=None):
        self.point_id = point_id
        self.path_toposcale_project = path_toposcale_project
        self.path_noaa = path_noaa
        self.file_met_office = file_met_office
        
        if not os.path.isdir(path_toposcale_project):
            raise('ERROR: Path to TopoPyScale project does not exist')
        if not os.path.isdir(path_noaa):
            raise('ERROR: Path to NOAA dataset does not exist')
        
        self.load_meta()
        
        self.obs = observation(path_noaa = self.path_noaa, 
                               stn_id=self.station_id, 
                               file_met_office=self.file_met_office)
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
        
        
    def compute_reference_periods(self, water_month_start=9):
        if hasattr(omu.obs,'noaa'):
            cu.compute_reference_periods(self.obs.noaa, water_month_start=water_month_start)

        if hasattr(omu.obs,'met_office'):
            cu.compute_reference_periods(self.obs.met_office, water_month_start=water_month_start)
        
        if hasattr(omu.down,'toposcale'):
            cu.compute_reference_periods(self.down.toposcale, water_month_start=water_month_start)
        
        if hasattr(omu.down,'fsm'):
            cu.compute_reference_periods(self.down.fsm, water_month_start=water_month_start)
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
    
    def __init__(self, file_met_office=None, path_noaa=None, stn_id=None):
        import fnmatch
        
        self.file_met_office = file_met_office
        self.dataset = []
        
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

        cu.compute_reference_periods(to, water_month_start=9)
        to.SD = to.SD / 100  #convert cm to m
        self.met_office = to

    def load_NOAA_data(self, file):
        import isd_NOAA as noa
        
        self.noaa = noa.read_isd_csv(file, 
                                     var=['tmp', 'wnd', 'precip', 'rh', 'sd', 'slp'], 
                                     keep_cols=noa.keep_cols_default, 
                                     index_timestamp=True)
class seasonal():
    def __init__(self, df, var='t', name=None):
        self.df = df
        self.var = var
        self.name = name

        if 'water_year' not in df.columns:
            compute_reference_periods(ti)

        self.t_seasonal = self.df[var].groupby(['water_year', 'water_doy']).value.mean().unstack()

    def compute_anomaly(self, window_days=5, window_years=10, reference_period=None):
        if reference_period is None:
            reference_period = [self.df.iloc[0].index, self.df.iloc[-1].index]

        self.window_days = window_days
        self.window_years = window_years
        self.t_seasonl_anomaly = (self.t_seasonal.rolling(window_days, center=True, axis=1).mean().rolling(window_years,center=True, axis=0).mean() - self.t_seasonal.rolling(window_days, center=True, axis=1).mean().rolling(window_years,center=True, axis=0).mean().mean(axis=0)).T


    def plot_seasonal(self, series='t_seasonal', vmin=-20, vmax=10, vcenter=None, cmap=plt.cm.RdBu_r):
        if vcenter is None:
            vcenter = (vmax-vmin)/2 + vmin

        cnorm = colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        tu = getattr(self, series)
        plt.imshow(tu,
                   aspect='auto',cmap=cmap, interpolation='nearest', norm=cnorm,
                   extent=[self.df.iloc[0].index.year-0.5, self.df.iloc[-1].index.year+0.5, 366,0])
        plt.colorbar()
        plt.yticks(ti.water_doy.values, labels=ti.index.strftime('%b'))




class snow():
    def __init__(self, df, snow_var, name=None):
        self.df = df
        self.snow_var = snow_var
        self.name = name
    
    def plot_snow_depths(self, ylim=[0,400]):
        year_list = self.df.water_year.unique()
        for i, year in enumerate(year_list):
            if i==0:
                plt.plot(self.df.loc[self.df.water_year==year].water_doy, self.df.loc[self.df.water_year==self.df][self.snow_var], alpha=0.2, c='k',
                        label=f'{year_list.min()}-{year_list.max()} Snow depths')
            else:
                plt.plot(self.df.loc[self.df.water_year==year].water_doy, self.df.loc[df.water_year==year][self.snow_var], alpha=0.2, c='k')

        self.df.groupby(self.df.water_doy).median()[self.snow_var].plot(c='r', label=f'{year_list.min()}-{year_list.max()} Median snow depth')
        plt.ylim(ylim[0],ylim[1])
        plt.legend()
        plt.xlabel('Days from September 1')
        plt.ylabel('Snow depth [cm]')
        
    def plot_year_vs_month(self):
        mon = self.df.groupby([self.df.water_doy, self.df.water_year]).median()
        year_list = self.df.water_year.unique()
        plt.imshow(mon[self.snow_var].unstack(), aspect='auto', interpolation='nearest', extent=[year_list.min()-0.5,year_list.max()+0.5,366,0])

        plt.colorbar()
        plt.ylabel('Days from September 1')
        plt.xlabel('Hydrological year')
        plt.title(f'Daily Median Snow Depth at {self.name}')


    # Methods to compute snow season length
    def fill_with_hard_limit(
        df_or_series, limit: int,
        fill_method='interpolate',
        **fill_method_kwargs):
        """
        The fill methods from Pandas such as ``interpolate`` or ``bfill``
        will fill ``limit`` number of NaNs, even if the total number of
        consecutive NaNs is larger than ``limit``. This function instead
        does not fill any data when the number of consecutive NaNs
        is > ``limit``.

        Adapted from: https://stackoverflow.com/a/30538371/11052174

        :param df_or_series: DataFrame or Series to perform interpolation
            on.
        :param limit: Maximum number of consecutive NaNs to allow. Any
            occurrences of more consecutive NaNs than ``limit`` will have no
            filling performed.
        :param fill_method: Filling method to use, e.g. 'interpolate',
            'bfill', etc.
        :param fill_method_kwargs: Keyword arguments to pass to the
            fill_method, in addition to the given limit.

        :returns: A filled version of the given df_or_series according
            to the given inputs.

        From: https://stackoverflow.com/a/66373000/1367097
        """

        # Keep things simple, ensure we have a DataFrame.
        try:
            df = df_or_series.to_frame()
        except AttributeError:
            df = df_or_series

        # Initialize our mask.
        mask = pd.DataFrame(True, index=df.index, columns=df.columns)

        # Get cumulative sums of consecutive NaNs.
        grp = (df.notnull() != df.shift().notnull()).cumsum()

        # Add columns of ones.
        grp['ones'] = 1

        # Loop through columns and update the mask.
        for col in df.columns:

            mask.loc[:, col] = (
                    (grp.groupby(col)['ones'].transform('count') <= limit)
                    | df[col].notnull()
            )

        # Now, interpolate and use the mask to create NaNs for the larger
        # gaps.
        method = getattr(df, fill_method)
        out = method(limit=limit, **fill_method_kwargs)[mask]

        # Be nice to the caller and return a Series if that's what they
        # provided.
        if isinstance(df_or_series, pd.Series):
            # Return a Series.
            return out.loc[:, out.columns[0]]

        return out


    def find_periods_ndays(self, ndays=10, with_snow=True, snd_thresh=0, water_month_start=9):
        '''
        Function to compute start and end of snow periods. 

        ndays (int): number of days defining period
        with_snow (bool):   - True: find periods of contiguous snow cover of at least ndays
                            - False: find periods of snow cover where the ground is snow free for a maximum period of ndays

        '''
        de = self.df.copy()

        if with_snow:
            de.loc[de>snd_thresh] = np.nan
            de.loc[de<=snd_thresh] = 1
            a = -1
        else:
            de.loc[de>snd_thresh] = 1
            de.loc[de<=snd_thresh] = np.nan
            a = 1

        de = fill_with_hard_limit(de, limit=ndays, fill_method='interpolate')
        de.loc[np.isnan(de)] = 0
        de = a * de.diff()

        df_periods = pd.DataFrame()
        df_periods['start'] = de.loc[de==1].index
        df_periods['end'] = de.loc[de==-1].index
        df_periods['duration'] = (df_periods.end - df_periods.start)

        df_periods.set_index(df_periods.start, inplace=True)
        cu.compute_reference_periods(df_periods, water_month_start=9)


        med_list = []
        for i, row in df_periods.iterrows():
            med_list.append(df.loc[row.start.strftime('%Y-%m-%d'):row.end.strftime('%Y-%m-%d')].median())
        df_periods['snd_median'] = med_list
        return df_periods
        
        
        