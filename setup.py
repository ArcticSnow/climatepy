#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name='climatepy',
    version='0.0.1',
    description='Collection of tools to perform timeseries analysis on climate data (Observation and Downscaled)',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/ArcticSnow/climatepy',
    download_url = 'https://github.com/ArcticSnow/climatepy',
    project_urls={
        'Source':'https://github.com/ArcticSnow/climatepy',
    },
    # Author details
    author=['Simon Filhol'],
    author_email='simon.filhol@geo.uio.no',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Atmospheric Science',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.9',
    ],

    # What does your project relate to?
    keywords=['climate', 'meteorology', 'xarray'],
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['xarray[complete]',
                        'matplotlib',
                        'seaborn',
                        'scikit-learn',
                        'pandas',
                        'numpy',
                        'netcdf4',
                        'h5netcdf',
                        'rasterio',
                        'pyproj',
                      'tensorflow',
                      'xarrayMannKendall'],
    include_package_data=True
)
