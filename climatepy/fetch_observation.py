"""
Functionalities to fetch weather observation data from public repositories
S. Filhol, Aug 2023

Some functions are modified version of TopoPyScale original implementation.
WARNING: CDS and other service require to have a user token

Currently supported data repositories:
- [x] WMO weather observation network 1755 to 2020. Available through CDS server
- [x] Norwegian Meteorological observation database
"""


import requests, os, glob
import pandas as pd
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
from climatepy import climate_utils as cu
def get_metno_obs(sources, voi, start_date, end_date, client_id=os.getenv('FROST_API_CLIENTID')):
    """
    Function to download observation data from MetNo FROST API (Norwegian Meteorological institute)

    Args
        sources (list): station code, e.g. 'SN25830'
        voi (list): variables to download
        start_date (str): starting date
        end_date (str): ending date
        client_id (str): FROST_API_CLIENTID

    Return:
        dataframe: all data combined together

    List of variable: https://frost.met.no/element table
    Find out about stations: https://seklima.met.no/
	WARNING: Download max one year at the time to not reach max limit of data to download.

    TODO:
    - [ ] convert df to xarray dataset with stn_id, time as coordinates
    """
    endpoint = 'https://frost.met.no/observations/v0.jsonld'

    df_out = pd.DataFrame()
    for source in sources:
        print('.............................................')
        print('---> Retreiving data for {}'.format(source))
        for var in voi:
            parameters = {
                'sources': source,
                'elements': var,
                'referencetime': start_date + '/' + end_date,
            }

            # Issue an HTTP GET request
            r = requests.get(endpoint, parameters, auth=(client_id,''))
            json = r.json()
            # Check if the request worked, print out any errors
            if r.status_code == 200:
                data = json['data']
                print('-> got VOI: {}'.format(var))
                df = pd.DataFrame()
                for i in range(len(data)):
                    row = pd.DataFrame(data[i]['observations']) # [:1] # raw data = [:1]; corrected = [1:]
                    row['referenceTime'] = data[i]['referenceTime']
                    row['sourceId'] = data[i]['sourceId']
                    df = df.append(row)

                df = df.reset_index()
                df = df[['elementId', 'value', 'qualityCode', 'referenceTime', 'sourceId', 'unit']].copy()
                df['referenceTime'] = pd.to_datetime(df['referenceTime'])
                df_out = df_out.append(df)
            else:
                print('..... Error with variable {}! Returned status code {}'.format(var, r.status_code))
                print('..... Reason: %s' % json['error']['reason'])
                print('---> {} for {} skipped'.format(var, source))

    return df_out


def fetch_WMO_insitu_observations(years,
                                  months,
                                  bbox,
                                  target_path='.',
                                  product='daily',
                                  var=['accumulated_precipitation', 'air_temperature', 'fresh_snow',
                     'snow_depth', 'snow_water_equivalent', 'wind_from_direction', 'wind_speed'],
                                  n_threads=20):
    """
    Function to download WMO in-situ data from land surface in-situ observations from Copernicus.
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-surface-land?tab=overview

    Args:
        year (str or list): year(s) to download
        month (str or list): month(s) to download
        bbox (list): bonding box in lat-lon [lat_south, lon_west, lat_north, lon_east]
        target (str): filename
        var (list): list of variable to download. available:
        n_threads (int): number of threads to download from CDS server

    Returns:
        Store to disk the dataset as zip file

    TODO:
        - [x] test function
        - [ ] check if can download large extent at once or need multiple requests?
        - [x] save data in individual files for each stations. store either as csv or netcdf (better)
        - [ ] write file reader, and data compression into netcdf. Maybe as part of another function
    """
    import cdsapi
    import zipfile

    try:
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
    except:
        print(f'ERROR: path "{target_path}" does not exist')
        return False

    if product == 'sub_daily':
        var_avail = ['air_pressure', 'air_pressure_at_sea_level', 'air_temperature',
                     'dew_point_temperature', 'wind_from_direction', 'wind_speed']
        for v in var:
            if v not in var_avail:
                print(f'WARNING: fecthing {v} from WMO on CDS server not available for {product}')
                return

    elif product == 'daily':
        var_avail = ['accumulated_precipitation', 'air_temperature', 'fresh_snow',
                     'snow_depth', 'snow_water_equivalent', 'wind_from_direction', 'wind_speed']
        for v in var:
            if v not in var_avail:
                print(f'WARNING: fecthing {v} from WMO on CDS server not available for {product}')
                return

    elif product == 'monthly':
        var_avail = ['accumulated_precipitation', 'air_temperature']
        for v in var:
            if v not in var_avail:
                print(f'WARNING: fecthing {v} from WMO on CDS server not available for {product}')
                return
    else:
        print(f'WARNING: {product} not available')

    def cds_retreive(product, var, year, month, bbox, target_path):
        fname = f'download_{year}_{month}.zip'
        c = cdsapi.Client()
        c.retrieve(
            'insitu-observations-surface-land',
            {
                'time_aggregation': product,
                'variable': var,
                'usage_restrictions': 'restricted',
                'data_quality': 'passed',
                'year': year,
                'month': month,
                'day': ['01', '02', '03',
                     '04', '05', '06',
                     '07', '08', '09',
                     '10', '11', '12',
                     '13', '14', '15',
                     '16', '17', '18',
                     '19', '20', '21',
                     '22', '23', '24',
                     '25', '26', '27',
                     '28', '29', '30',
                     '31'
                     ],
                'area': bbox,
                'format': 'zip',
            },
            target_path + os.sep + fname)
        try:
            with zipfile.ZipFile(target_path + os.sep + fname) as z:
                z.extractall(path=target_path)
                print('---> Observation extracted')
                os.remove(target_path + os.sep + fname)
                print(f'\t File {fname} unzipped')
                return
        except:
            print(f'ERROR: Invalid target path\n\t target path used: {target_path}')

    year_list = []
    month_list = []
    for year in years:
        for month in months:
            year_list.append(year)
            month_list.append(month)
    i = len(year_list)

    param = zip([product]*i, [var]*i, year_list, month_list, [bbox]*i, [target_path]*i)
    cu.multithread_pooling(cds_retreive, fun_param=param, n_threads=n_threads)
    print('---> download completed')

