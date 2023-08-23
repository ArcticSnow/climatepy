import pandas as pd
from datetime import datetime
import cdsapi, os, sys
from dateutil.relativedelta import *
import glob
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime, timedelta

def fetch_era5(product, startDate, endDate, eraDir, latN, latS, lonE, lonW, step, num_threads=10, surf_plev='surf',
				  plevels=None, realtime=False, varoi=None):
	""" Sets up era5 surface retrieval.
	* Creates list of year/month pairs to iterate through.
	* MARS retrievals are most efficient when subset by time.
	* Identifies preexisting downloads if restarted.
	* Calls api using parallel function.

	Args:
		product: "reanalysis" (HRES) or "ensemble_members" (EDA)
		startDate:
		endDate:
		eraDir: directory to write output
		latN: north latitude of bbox
		latS: south latitude of bbox
		lonE: easterly lon of bbox
		lonW: westerly lon of bbox
		step: timestep to use: 1, 3, 6
		num_threads: number of threads to use for downloading data
		surf_plev: download surface single level or pressure level product: 'surf' or 'plev'
		plevels: list of pressure level to download for plev option
		realtime: download realtime mode
		varoi: list of variable to download

	Returns:
		Monthly era surface files stored in disk.

	"""
	print('\n')
	print('---> Loading ERA5 {} climate forcing'.format(surf_plev))
	bbox = [str(latN), str(lonW), str(latS), str(lonE)]
	time_step_dict = {'1H': ['00:00', '01:00', '02:00',
							 '03:00', '04:00', '05:00',
							 '06:00', '07:00', '08:00',
							 '09:00', '10:00', '11:00',
							 '12:00', '13:00', '14:00',
							 '15:00', '16:00', '17:00',
							 '18:00', '19:00', '20:00',
							 '21:00', '22:00', '23:00'],
					'3H': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
					'6H': ['00:00', '06:00', '12:00', '18:00']}

	df = pd.DataFrame()
	# date_range will make sure to include the month of the latest date (endDate) provided
	df['dates'] = pd.date_range(startDate, pd.Timestamp(endDate)-pd.offsets.Day()+pd.offsets.MonthEnd(), freq='M', inclusive='both')
	df['month'] = df.dates.dt.month
	df['year'] = df.dates.dt.year
	if surf_plev == 'surf':
		#df['dataset'] = df.dates.apply(lambda x: 'reanalysis-era5-single-levels' if x.year >= 1979 else 'reanalysis-era5-single-levels-preliminary-back-extension')
		df['dataset'] = 'reanalysis-era5-single-levels'
		df['target_file'] = df.dates.apply(lambda x: eraDir + "SURF_%04d%02d.nc" % (x.year, x.month))
	elif surf_plev == 'plev':
		#df['dataset'] = df.dates.apply(lambda x: 'reanalysis-era5-pressure-levels' if x.year >= 1979 else 'reanalysis-era5-pressure-levels-preliminary-back-extension')
		df['dataset'] = 'reanalysis-era5-pressure-levels'
		df['target_file'] = df.dates.apply(lambda x: eraDir + "PLEV_%04d%02d.nc" % (x.year, x.month))
		loc_list = []
		loc_list.extend([plevels]*df.shape[0])
		df['plevels'] = loc_list
	else:
		sys.exit('ERROR: surf_plev can only be surf or plev')
	df['file_exist'] = 0
	df.file_exist = df.target_file.apply(lambda x: os.path.isfile(x)*1)
	df['step'] = step
	df['time_steps'] = df.step.apply(lambda x: time_step_dict.get(x))
	df['bbox'] = df.step.apply(lambda x: bbox)
	df['product_type'] = product
	if varoi is None:
		if surf_plev == 'surf':
			df['varoi'] = df.step.apply(['geopotential', '2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
                             'surface_solar_radiation_downwards','surface_pressure',
                             'Total precipitation', '2m_temperature', 'TOA incident solar radiation',
                             'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux'])
		elif surf_plev == 'plev':
			df['varoi'] = df.step.apply(['geopotential', 'temperature', 'u_component_of_wind','v_component_of_wind', 'relative_humidity', 'specific_humidity'])
	else:
		df['varoi'] = df.step.apply(lambda x: varoi)


	print("Start = ", df.dates[0].strftime('%Y-%b'))
	print("End = ", df.dates[len(df.dates) - 1].strftime('%Y-%b'))

	if df.file_exist.sum() > 0:
		print("ECWMF {} data found:".format(surf_plev.upper()))
		print(df.target_file.loc[df.file_exist == 1].apply(lambda x: x.split('/')[-1]))

	if (df.file_exist == 0).sum() > 0:
		print("Downloading {} from ECWMF:".format(surf_plev.upper()))
		print(df.target_file.loc[df.file_exist == 0].apply(lambda x: x.split('/')[-1]))

	download = df.loc[df.file_exist == 0]
	if download.shape[0] > 0:
		# ans = input('---> Download ERA5 {} data? (y/n)'.format(surf_plev.upper()))
		# if (ans.lower() == 'y') or (ans == '1'):
		if surf_plev == 'surf':
			pool = ThreadPool(num_threads)
			pool.starmap(era5_request_surf,
						 zip(list(download.dataset),
							 list(download.year),
							 list(download.month),
							 list(download.bbox),
							 list(download.target_file),
							 list(download.product_type),
							 list(download.time_steps),
							 list(download.varoi))
						 )
			pool.close()
			pool.join()
		elif surf_plev == 'plev':
			pool = ThreadPool(num_threads)
			pool.starmap(era5_request_plev, zip(list(download.dataset),
												list(download.year),
												list(download.month),
												list(download.bbox),
												list(download.target_file),
												list(download.product_type),
												list(download.time_steps),
												list(download.plevels),
												list(download.varoi)))
			pool.close()
			pool.join()
		else:
			sys.exit('ERROR: surf_plev can only be surf or plev')
		#else:
		#	sys.exit('ERROR: Some forcing files are missing given the date range provided\n ---> or implement a method to modify start/end date of project to file available')

	if realtime:
		if surf_plev == 'surf':
			# redownload current month to catch missing days in realtime mode.
			era5_realtime_surf(eraDir, df.dataset[0], df.bbox[0], df.product_type[0])

		if surf_plev == 'plev':
			# redownload current month to catch missing days in realtime mode.
			era5_realtime_plev(eraDir, df.dataset[0], df.bbox[0], df.product_type[0], df.plevels[0])

def era5_request_surf(dataset,
                      year,
                      month,
                      bbox,
                      target,
                      product,
                      time,
                      varoi=['geopotential', '2m_dewpoint_temperature', 'surface_thermal_radiation_downwards',
                             'surface_solar_radiation_downwards','surface_pressure',
                             'Total precipitation', '2m_temperature', 'TOA incident solar radiation',
                             'friction_velocity', 'instantaneous_moisture_flux', 'instantaneous_surface_sensible_heat_flux']):
	"""CDS surface api call

	Args:
		dataset (str): copernicus dataset (era5)
		year (str or list): year of interest
		month (str or list): month of interest
		bbox (list): bonding box in lat-lon
		target (str): filename
		product (str): type of model run. defaul: reanalysis
		time (str or list): hours for which to download data

	Returns:
		Store to disk dataset as indicated

	"""
	c = cdsapi.Client()
	c.retrieve(
		dataset,
		{'variable': varoi,
		 'product_type': product,
		 "area": bbox,
		 'year': year,
		 'month': '%02d'%(month),
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
		 'time': time,
		 'grid': [0.25, 0.25],
		 'format': 'netcdf'
		 },
		target)
	print(f'--> {target} downloaded')




def era5_request_plev(dataset,
                      year,
                      month,
                      bbox,
                      target,
                      product,
                      time,
                      plevels,
                      varoi=['geopotential', 'temperature', 'u_component_of_wind','v_component_of_wind', 'relative_humidity', 'specific_humidity']):
	"""CDS plevel api call

	Args:
		dataset (str): copernicus dataset (era5)
		year (str or list): year of interest
		month (str or list): month of interest
		bbox (list): bonding box in lat-lon
		target (str): filename
		product (str): type of model run. defaul: reanalysis
		time (str or list): hours to query
		plevels (str or list): pressure levels to query

	Returns:
		Store to disk dataset as indicated

	"""
	c = cdsapi.Client()
	c.retrieve(
		dataset,
		{
			'product_type': product,
			'format': 'netcdf',
			"area": bbox,
			'variable': varoi,
			'pressure_level': plevels,
			'year': year,
			'month': '%02d'%(month),
			'day': [
				'01', '02', '03',
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
			'time': time,
			'grid': [0.25, 0.25],
		},
		target)
	print(f'--> {target} downloaded')


def fetch_E_Obs(target='e_obs.zip',
				product='ensemble_mean',
				varoi=['mean_temperature', 'precipitation_amount', 'relative_humidity','surface_shortwave_downwelling_radiation'],
				grid_res='0.1deg',
				format='zip',
				version='27.0e',
				period='full_period'):
	'''
	Function to fetch E-Obs data from CDS webserver. E-Obs data are referenced under Cornes et al. (2018).
	E-Obs are spatially interpolated in-situ observation data using statistical methods over the European continent. Gridding avaialbel at 0.1 and 0.25 degree.

	Data and options to download available at: https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-gridded-observations-europe?tab=form

	Args:
		target (str): filename of download zip or tar.gz file
		product (str): 'ensemble_mean', 'ensemble_spread', 'elevation'. Elevation
		varoi (list str): variable of interest to download
		grid_res (str): grid resolution. '0.1deg' or '0.25deg' avaiable
		format (str): file format to download data. 'zip' or 'tgz'
		version (str): version of the dataset. latest (as of Aug 23, 2023: '27.0e')
		period (str): period of interest. 'full_period' or specific periods '1950_1964','1965_1979','1980_1994','1995_2010','2011_2018','2012_2020'

	'''
	# handle potential incompatibilities in request parameters
	if product == 'elevation':
		varoi = 'land_surface_elevation'
	if format == 'tg' and target.split('.')[-1] == 'zip':
		target = target.split('.')[:-1] + '.tg'

	c = cdsapi.Client()
	c.retrieve(
		'insitu-gridded-observations-europe',
		{
			'format': format,
			'product_type': product,
			'variable': varoi,
			'grid_resolution': grid_res,
			'version': version,
			'period': period,
		},
		target)

	print(f'--> {target} downloaded')