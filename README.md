# climatepy
Collection of tools to perform timeseries analysis, synoptic analysis, and plots on climate data (Observation and Downscaled). This builds partly
upon TopoPyScale outputs as well as a number of official data format from NOAA, WMO, MetNo, et al. or land surface model outputs (*e.g.* FSM)
S. Filhol, May 2023

**This toolbox is under construction and currently designed for specific projects** 

## Installation

```bash
git clone https://github.com/ArcticSnow/climatepy.git

# install in development mode
pip install -e climatepy
```

## TODO
- [ ] write function to compute FDD and TDD
- [ ] write function to compute snow free season (sd_thresh_free (e.g. 10cm), sd_thresh_snow_onset (e.g. 80cm)).
- [ ] write function to infer synoptic patterns using unsupervised techniques
[ ] write function to infer synoptic patterns from known patterns (dates with known of similar recognizable patterns)

## Long term ideas and Resources:
integrate possibility to download forecast model (i.e. GFS):
- NAM regional model (e.g. Alaska) by NOAA: https://www.nco.ncep.noaa.gov/pmb/products/nam/
- GFS model (NOAA) see the Python lirbary: https://github.com/jagoosw/getgfs