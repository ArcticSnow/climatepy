'''
Python Code to handle ISD NOAA data format
Simon Filhol, February 2021

TODO:
    - add support exception in case a column is available. 
    - write functiont to doanload data from NOAA server


Official data description: ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-format-document.pdf


**Variable description**
1. WND
    - Direction (1-360, 999=NaN), scaling factor = 1
    - Direction Quality. Keep only if equal to 1
    - type code {'A':'Abridged Beaufort', 'B':'Beaufort','C':'Calm', 'H':'5-Minute Average Speed', 'N':'Normal','R':'60-Minute Average Speed', 'Q':'Squal','T':'180 Minute Average Speed', 'V':Variable','9':'Missing'}
    - speed (0-900), scaling factor = 10, units= m.s-1, 9999=NaN
    - speed quality, keep 1 or 6, NaN all other
2. TMP
    - Air temperature (-0932 - 0618), scaling factor = 10, units = degC, 99999=NaN
    - air temp quality, keep 1 and 5
3. DEW
    - dew temperature (-0982 - 0368), scaling factor = 10, units = degC, 99999=NaN
    - dew temp quality, keep 1 and 5,
4. SLP
    - Sea Level Pressure relaitve to sea level (08600-10900), scaling factor = 10, units = hPa, 99999=NaN
    - sea level pressure quality: keep 1 and 5
5. AA1, AA2, AA3, AA4
    - Liquid Precipitation
6. AJ1
    - Snowdepth
7. MA1
    - air pressure (not compensated)
8. MW1
    - Manual weather observation (interesting to cross check conditions)


Nice blog post on the ISD NOAA data format:
https://www.visualcrossing.com/resources/documentation/weather-data/how-we-process-integrated-surface-database-historical-weather-data/
'''
import pandas as pd
import numpy as np


def RHfromDEW_Wanielista(Tair, Tdew):
    '''
    Formula to copute Relative humidity from air temperature and dew temperature.
    Tair and Tdew in degree C
    returns: RH in %
    
    Reference: Martin Wanielista, Robert Kersten and Ron Eaglin. 1997. Hydrology Water Quantity and Quality 
    Control. John Wiley & Sons. 2nd ed. 
    '''
    RH = 100*((112-0.1*Tair+Tdew)/(112+0.9*Tair))**8
    return RH

keep_cols_default = ['STATION', 'DATE', 'SOURCE', 'LATITUDE', 'LONGITUDE', 'ELEVATION',
       'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL']

def read_isd_csv(file, var=['tmp', 'wnd', 'precip', 'rh', 'sd', 'slp'], keep_cols=keep_cols_default, index_timestamp=True):
    '''
    Function to parse ISD data from NOAA saved as CSV file into Pandas DataFrame. Index is replaced by timestamp
    
    :param file: path of file to read
    :param var: variable to parse. Currently supported: tmp, wnd, precip, rh, sd, slp 
    :param drop_cols: names of columns to drop from the dataframe returned
    :return: Pandas dataframe
    '''
    
    df = pd.read_csv(file, parse_dates=['DATE'])
    if index_timestamp:
        df.set_index(df.DATE, inplace=True)
    cols = df.columns
    # unpack airtemperature
    if 'tmp' in var and 'tmp' in cols.str.lower():
        df[['tmp', 'tmp_q']] = df.TMP.str.split(',', n=1, expand=True).astype(int)
        df.tmp.loc[df.tmp == 9999] = np.nan
        df.tmp = df.tmp/10
        
        keep_cols.extend(('tmp','tmp_q'))
        keep_cols = list(set(keep_cols))

    #unpack wind observation
    if 'wnd' in var and 'wnd' in cols.str.lower():
        df[['wd', 'wd_q', 'w_code', 'ws', 'ws_q']] = df.WND.str.split(',', expand=True)
        df[['wd', 'wd_q', 'ws', 'ws_q']] = df[['wd', 'wd_q', 'ws', 'ws_q']].astype(int)
        df.wd.loc[df.wd==999] = np.nan
        df.ws.loc[df.ws==9999] = np.nan
        df.ws = df.ws / 10
        keep_cols.extend(('wd', 'wd_q', 'w_code', 'ws', 'ws_q'))
        keep_cols = list(set(keep_cols))

    # unpack DEW to RH
    if 'rh' in var and 'dew' in cols.str.lower() and 'tmp' in cols.str.lower():
        df[['dew', 'dew_q']]=df.DEW.str.split(',', n=1, expand=True).astype(int)
        df.dew.loc[df.dew == 9999] = np.nan
        df.dew = df.dew/10
        df['rh'] = RHfromDEW_Wanielista(df.tmp, df.dew)
        keep_cols.extend(('dew', 'dew_q'))
        keep_cols = list(set(keep_cols))

    # unpack sea level pressure
    if 'slp' in var and 'slp' in cols.str.lower():
        df[['slp', 'slp_q']]=df.SLP.str.split(',', n=1, expand=True).astype(int)
        df.slp.loc[df.slp == 99999] = np.nan
        df.slp = df.slp/10
        keep_cols.extend(('slp', 'slp_q'))
        keep_cols = list(set(keep_cols))

    # unpack snow depth
    if 'sd' in var and 'aj1' in cols.str.lower():
        df[['sd','sd_code','sd_q','swe','swe_code','swe_q']] = df.AJ1.str.split(',',  expand=True)
        df[['sd', 'swe']] = df[['sd', 'swe']].astype(float)
        df.sd.loc[df.sd==9999] = np.nan
        df.swe.loc[df.swe==9999] = np.nan
        df[['swe']] = df[['swe']]/10  # convert to mm unit
        df[['sd']] = df[['sd']]*10  # convert to mm unit
        keep_cols.extend(('sd','sd_code','sd_q','swe','swe_code','swe_q'))
        keep_cols = list(set(keep_cols))

    # unpack precipitation columns
    if 'precip' in var and 'AA1' in cols and 'AA2' in cols and 'AA3' in cols:
        df[['precip_period_1','precip_1','precip_code_1','precip_q_1']] = df.AA1.str.split(',',  expand=True)
        df[['precip_period_2','precip_2','precip_code_2','precip_q_2']] = df.AA2.str.split(',',  expand=True)
        df[['precip_period_3','precip_3','precip_code_3','precip_q_3']] = df.AA2.str.split(',',  expand=True)
        df[['precip_period_1','precip_1','precip_period_2','precip_2','precip_period_3','precip_3']] = df[['precip_period_1','precip_1','precip_period_2','precip_2','precip_period_3','precip_3']].astype(float)
        df.precip_period_1.loc[df.precip_period_1 == 99] = np.nan
        df.precip_period_2.loc[df.precip_period_2 == 99] = np.nan
        df.precip_period_3.loc[df.precip_period_3 == 99] = np.nan
        df.precip_1.loc[df.precip_1 == 9999] = np.nan
        df.precip_2.loc[df.precip_2 == 9999] = np.nan
        df.precip_3.loc[df.precip_3 == 9999] = np.nan
        keep_cols.extend(('precip_period_1','precip_1','precip_code_1','precip_q_1',
                         'precip_period_2','precip_2','precip_code_2','precip_q_2',
                         'precip_period_3','precip_3','precip_code_3','precip_q_3'))
        keep_cols = list(set(keep_cols))

    # drop columns
    if 'all' in keep_cols:
        return df
    else:
        return df[keep_cols]