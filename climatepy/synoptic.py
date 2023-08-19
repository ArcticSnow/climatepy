'''
Functions to computde synoptic indexes

- 2D Blocking index following the method in Nagavciuc et al.(2022) which is based on Tibaldi and Molteni (1990)
'''


def convert_longitude(ds, lon_name='longitude'):

    # Adjust lon values to make sure they are within (-180, 180)
    ds['_longitude_adjusted'] = xr.where(
        ds[lon_name] > 180,
        ds[lon_name] - 360,
        ds[lon_name])

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    ds = (
        ds
        .swap_dims({lon_name: '_longitude_adjusted'})
        .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
        .drop(lon_name))

    ds = ds.rename({'_longitude_adjusted': lon_name})
    return ds


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
    ds = convert_longitude(ds)  # convert longitude from 0:360 to -180:180
    dss = ds.to_dataset()
    dss['GHGS'] = (ds - ds.shift(latitude=15)) / 15
    dss['GHGN'] = (ds.shift(latitude=15) - ds) / 15
    dss['BI'] = (dss.GHGS > 0) & (dss.GHGN < -10)

    return dss

