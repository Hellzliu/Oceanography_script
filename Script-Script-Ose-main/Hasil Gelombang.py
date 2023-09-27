import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset, num2date
import xarray as xr
from datetime import datetime
import pandas as pd
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
import math
from sklearn import datasets, linear_model
from tabulate import tabulate
#%%
#Data DELFT
data = Dataset(r'wavm-wave-0.1.nc','r')
lon = data.variables['x'][:]
lonlist = lon[100] #system koordinat di delft mengosongkan longitude daratan
lat = data.variables['y'][:]
latlist = lat[:,4] #system koordinat di delft mengosongkan latitude daratan

data2 = Dataset(r'wavm-wave-0.02.nc','r')
lon2 = data2.variables['x'][:]
lonlist2 = lon2[240] #system koordinat di delft mengosongkan longitude daratan
lat2 = data2.variables['y'][:]
latlist2 = lat2[:,525] #system koordinat di delft mengosongkan latitude daratan

time1 = []

def parameter(a, x, y, u, v, kind):
    swh = a.variables['hsign'][:]
    tp = a.variables['period'][:]
    mdr = a.variables['dir'][:]
    swh[swh==0]=['nan']
    tp[tp==0]=['nan']
    mdr[mdr==0]=['nan']
    time = a.variables['time'][:]
    time_unit = a.variables['time'].getncattr('units')
    local_time = num2date(time, units=time_unit, calendar='gregorian')


    #Export Koordinat
    list_lon = [120.5,123,124.5]
    list_lat = [-9.5,-10,-8.55]
    
    app_lon1 = []
    app_lat1 = []
    
    for a in range(len(list_lon)) :
        ext_lon = list_lon[a]
        ext_lat = list_lat[a]
        sq_dif_lon = (x - ext_lon)**2
        sq_dif_lat = (y - ext_lat)**2
        index_lon = sq_dif_lon.argmin()
        app_lon1.append(index_lon)
        index_lat = sq_dif_lat.argmin()
        app_lat1.append(index_lat)
    if kind == 'swh' :
        return swh[:,app_lat1[u],app_lon1[v]]
    elif kind == 'tp':
        return tp[:,app_lat1[u],app_lon1[v]]
    elif kind == 'mdr':
        return mdr[:,app_lat1[u],app_lon1[v]]
    else :
        return 'error pls being specified at your kind'
x = parameter(data, lonlist, latlist, 0, 0, 'tp')  

#%%
swh = data.variables['hsign'][:]
swh[swh==0]=['nan']
list_lon = [120.5,123,124.5]
list_lat = [-9.5,-10,-8.55]
app_lon1 = []
app_lat1 = []
for b in range(len(list_lon)) :
    ext_lon = list_lon[b]
    ext_lat = list_lat[b]
    sq_dif_lon = (lonlist - ext_lon)**2
    sq_dif_lat = (latlist - ext_lat)**2
    index_lon = sq_dif_lon.argmin()
    app_lon1.append(index_lon)
    index_lat = sq_dif_lat.argmin()
    app_lat1.append(index_lat)
#%%
#iter time
def timed(a) :
    time = a.variables['time'][:]
    time_unit = a.variables['time'].getncattr('units')
    local_time = num2date(time, units=time_unit, calendar='gregorian')
    return local_time

ext_time1 = np.arange(0, len(timed(data)))
timex1 = []
    
for x in ext_time1:
    iter_local_time=str(timed(data)[x])
    timex1.append(iter_local_time)
time_list1 = pd.to_datetime(pd.Series(timex1), format='%Y-%m-%d %H:%M:%S')                                 
#%%
#Tinggi Gelombang
markers_on = [25, 229, 385]
fig, ax = plt.subplots(3,1,figsize=(12, 6))
fig.tight_layout(pad=3.0)
ax[0].format_xdata = mdates.DateFormatter('%m-%d')
ax[0].plot(time_list1, parameter(data, lonlist, latlist, 0, 0)  , label='0.1')
ax[0].plot(time_list1, parameter(data2, lonlist2, latlist2, 0, 0)  , label='0.02')
ax[0].plot(time_list1, parameter(data, lonlist, latlist, 0, 0),'vb', markevery=markers_on)
ax[0].legend(loc = 5)
ax[0].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave at : Station A'))

ax[1].format_xdata = mdates.DateFormatter('%m-%d')
ax[1].plot(time_list1, parameter(data, lonlist, latlist, 1, 1)  , label='0.1')
ax[1].plot(time_list1, parameter(data2, lonlist2, latlist2, 1, 1)  , label='0.02')
ax[1].plot(time_list1, parameter(data, lonlist, latlist, 1, 1),'vb', markevery=markers_on)
ax[1].legend(loc = 5)
ax[1].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave at : Station B'))

ax[2].format_xdata = mdates.DateFormatter('%m-%d')
ax[2].plot(time_list1, parameter(data, lonlist, latlist, 2, 2)  , label='0.1')
ax[2].plot(time_list1, parameter(data2, lonlist2, latlist2, 2, 2)  , label='0.02')
ax[2].plot(time_list1, parameter(data, lonlist, latlist, 2, 2),'vb', markevery=markers_on)
ax[2].legend(loc = 5)
ax[2].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave at : Station C'))
#%%
#shortern version of Tinggi Gelombang
markers_on = [25, 229, 385]
fig, ax = plt.subplots(3,1,figsize=(12, 6))
fig.tight_layout(pad=3.0)

for i, station in enumerate(['A', 'B', 'C']):
    ax[i].format_xdata = mdates.DateFormatter('%m-%d')
    ax[i].plot(time_list1, parameter(data, lonlist, latlist, i, i, 'swh'), label='0.1')
    ax[i].plot(time_list1, parameter(data2, lonlist2, latlist2, i, i, 'swh'), label='0.02')
    ax[i].plot(time_list1, parameter(data, lonlist, latlist, i, i, 'swh'),'vb', markevery=markers_on)
    ax[i].legend(loc = 5)
    ax[i].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave at : Station {}'.format(station)))

#%%
# shortern version of Period Gelombang
markers_on = [25, 229, 385]
fig, ax = plt.subplots(3,1,figsize=(12, 6))
fig.tight_layout(pad=3.0)

for i, station in enumerate(['A', 'B', 'C']):
    ax[i].format_xdata = mdates.DateFormatter('%m-%d')
    ax[i].plot(time_list1, parameter(data, lonlist, latlist, i, i, 'tp'), label='0.1')
    ax[i].plot(time_list1, parameter(data2, lonlist2, latlist2, i, i, 'tp'), label='0.02')
    ax[i].plot(time_list1, parameter(data, lonlist, latlist, i, i, 'tp'),'vb', markevery=markers_on)
    ax[i].legend(loc = 5)
    ax[i].set(xlabel="Date", ylabel="Wave Period (s)", title=('Wave Period at : Station {}'.format(station)))
    #%%
def mean(a):
    min1 = '{:.3f}'.format(np.min(a))
    max1 = '{:.3f}'.format(np.max(a))
    mean = '{:.3f}'.format(np.mean(a))
    return print('min','max','mean'),print(min1, max1, mean)

x1 = parameter(data, lonlist, latlist, 0, 0) #0.1 stasiun 1
x2 = parameter(data, lonlist, latlist, 1, 1) ## stasiun 2
x3 = parameter(data, lonlist, latlist, 2, 2) ## stasiun 3
    
y1 = parameter(data2, lonlist2, latlist2, 0, 0) #0.02 stasiun 1
y2 = parameter(data2, lonlist2, latlist2, 1, 1) ## stasiun 2
y3 = parameter(data2, lonlist2, latlist2, 2, 2) ## stasiun 3
   
#%%
#Periode Gelombang
fig, ax2 = plt.subplots(3,1,figsize=(12, 6))
fig.tight_layout(pad=3.0)
ax2[0].format_xdata = mdates.DateFormatter('%m-%d')
ax2[0].plot(time_list1, tp[:,app_lat1[0],app_lon1[0]], label='Delft3d')
ax2[0].legend(loc = 5)
ax2[0].set(xlabel="Date", ylabel="Periode (s)", title=('Wave Period in :'+str(ext_lon)+','+str(ext_lat)))

ax2[1].format_xdata = mdates.DateFormatter('%m-%d')
ax2[1].plot(time_list1, tp[:,app_lat1[1],app_lon1[1]], label='Delft3d')
ax2[1].legend(loc = 5)
ax2[1].set(xlabel="Date", ylabel="Periode (s)", title=('Wave Period in :'+str(ext_lon)+','+str(ext_lat)))

ax2[2].format_xdata = mdates.DateFormatter('%m-%d')
ax2[2].plot(time_list1, tp[:,app_lat1[2],app_lon1[2]], label='Delft3d')
ax2[2].legend(loc = 5)
ax2[2].set(xlabel="Date", ylabel="Periode (s)", title=('Wave Period in :'+str(ext_lon)+','+str(ext_lat)))


#%%
#Arah Gelombang
md1 = (-1*mdr[:,app_lat1[0],app_lon1[0]]+270)%360
md2 = (-1*mdr[:,app_lat1[1],app_lon1[1]]+270)%360
md3 = (-1*mdr[:,app_lat1[2],app_lon1[2]]+270)%360

fig, ax3 = plt.subplots(3,1,figsize=(12, 6))
fig.tight_layout(pad=3)
ax3[0].format_xdata = mdates.DateFormatter('%m-%d')
ax3[0].plot(time_list1, md1 , label='Delft3d')
ax3[0].legend(loc = 5)
ax3[0].set(xlabel="Date", ylabel="Direction (degree)", title=('Wave direciton in :'+str(ext_lon)+','+str(ext_lat)))

ax3[1].format_xdata = mdates.DateFormatter('%m-%d')
ax3[1].plot(time_list1, md2 , label='Delft3d')
ax3[1].legend(loc = 5)
ax3[1].set(xlabel="Date", ylabel="Direction (degree)", title=('Wave direciton in :'+str(ext_lon)+','+str(ext_lat)))

ax3[2].format_xdata = mdates.DateFormatter('%m-%d')
ax3[2].plot(time_list1, md3, label='Delft3d')
ax3[2].legend(loc = 5)
ax3[2].set(xlabel="Date", ylabel="Direction (degree)", title=('Wave direciton in :'+str(ext_lon)+','+str(ext_lat)))
