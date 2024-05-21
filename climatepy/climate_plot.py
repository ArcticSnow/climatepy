"""
Collection of functions for plotting timeseries with climatic point of view
S. Filhol, May 2024


in construction !!!!!!!!!
"""
from matplotlib import colors

def plot_rolling_mean(df, ax, var='tair', n_days=5, vmin=-20, vmax=5):

    cnorm = colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    ti = pd.DataFrame()
    ti['date'] = pd.to_datetime(pd.date_range("2022-01-01", pd.Timestamp("2022-12-31")+pd.offsets.MonthBegin(),
                  freq='M', inclusive='both')+pd.Timedelta('1D'), utc=False)
    ti.set_index(ti.date, inplace=True)
    cu.compute_reference_periods(ti, water_month_start=1)

    #plt.imshow(df_daily.rolling(5, axis=1, center=True).mean().rolling(10, axis=0, center=True).mean(), norm=cnorm, cmap=plt.cm.RdBu_r, aspect='auto', interpolation='nearest', extent=[1957.5,2022.5, 365.5,0.5])
    ax.imshow(df_daily.rolling(7, axis=1, center=True).mean(), norm=cnorm, cmap=plt.cm.RdBu_r, aspect='auto', interpolation='nearest', extent=[1957.5,2022.5, 365.5,0.5])

    ax.colorbar()
    ax.set_yticks(ti.water_doy.values, labels=ti.index.strftime('%b'))
    ax.set_title(f'{n_days} Days Mean Air Temperature  [$^{o}C$]')














