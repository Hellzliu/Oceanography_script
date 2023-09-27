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
data = Dataset(r'wavm-wave-0.02.nc','r')
lon = data.variables['x'][:]
lonlist = lon[63] #system koordinat di delft mengosongkan longitude daratan
lat = data.variables['y'][:]
latlist = lat[:,523] #system koordinat di delft mengosongkan latitude daratan
swh = data.variables['hsign'][:]
tp = data.variables['period'][:]
mdr = data.variables['dir'][:]
swh[swh==0]=['nan']
tp[tp==0]=['nan']
mdr[mdr==0]=['nan']
time = data.variables['time'][:]
time_unit = data.variables['time'].getncattr('units')
local_time = num2date(time, units=time_unit, calendar='gregorian')
#%%
#Data CMEMS
root = Dataset(r'VerifCMEMS.nc','r')
lon2 = root.variables['longitude'][:]
lat2 = root.variables['latitude'][:]
time2 = root.variables['time'][:]   
swh2 = root.variables['VHM0']
tp2 = root.variables['VTM02']
mdr2 = root.variables['VMDR']
time_unit2 = root.variables['time'].getncattr('units')
local_time2 = num2date(time2, units=time_unit2, calendar='gregorian')

#%%
#Data ECMWF
root2 = Dataset(r'Era5Wave.nc','r')
lon3 = root2.variables['longitude'][:]
lat3 = root2.variables['latitude'][:]
time3 = root2.variables['time'][:]   
swh3 = root2.variables['swh'][:]
tp3 = root2.variables['mwp'][:]
mdr3 = root2.variables['mwd'][:]
time_unit3 = root2.variables['time'].getncattr('units')
local_time3 = num2date(time3, units=time_unit3, calendar='gregorian')
#%%
#Export Koordinat
list_lon = [120.5,123,124.5]
list_lat = [-9.5,-10,-8.55]

app_lon1 = []
app_lat1 = []
app_lon2 = []
app_lat2 = []
app_lon3 = []
app_lat3 = []
for a in range(len(list_lon)) :
    ext_lon = list_lon[a]
    ext_lat = list_lat[a]
    sq_dif_lon = (lonlist - ext_lon)**2
    sq_dif_lat = (latlist - ext_lat)**2
    sq_dif_lon2 = (lon2 - ext_lon)**2
    sq_dif_lat2 = (lat2 - ext_lat)**2
    sq_dif_lon3 = (lon3 - ext_lon)**2
    sq_dif_lat3 = (lat3 - ext_lat)**2
    index_lon = sq_dif_lon.argmin()
    app_lon1.append(index_lon)
    index_lat = sq_dif_lat.argmin()
    app_lat1.append(index_lat)
    index_lon2 = sq_dif_lon2.argmin()
    app_lon2.append(index_lon2)
    index_lat2 = sq_dif_lat2.argmin()
    app_lat2.append(index_lat2)
    index_lon3 = sq_dif_lon3.argmin()
    app_lon3.append(index_lon3)
    index_lat3 = sq_dif_lat3.argmin()
    app_lat3.append(index_lat3)
#%%
#iter time
ext_time1 = np.arange(0, len(local_time))
ext_time2 = np.arange(209, 386)
ext_time3 = np.arange(624, 1153)
timex1 = []
timex2 = []
timex3 = []

for x in ext_time1:
        iter_local_time=str(local_time[x])
        timex1.append(iter_local_time)
        
for y in ext_time2:
        iter_local_time2=str(local_time2[y])
        timex2.append(iter_local_time2)
        
for z in ext_time3:
        iter_local_time3=str(local_time3[z])
        timex3.append(iter_local_time3)
time_list1 = pd.to_datetime(pd.Series(timex1), format='%Y-%m-%d %H:%M:%S')
time_list2 = pd.to_datetime(pd.Series(timex2), format='%Y-%m-%d %H:%M:%S')
time_list3 = pd.to_datetime(pd.Series(timex3), format='%Y-%m-%d %H:%M:%S')
#%% wave
maxDelft1 = max(swh[:,app_lat1[0],app_lon1[0]])
minDelft1 = min(swh[:,app_lat1[0],app_lon1[0]])
avgDelft1 = sum(swh[:,app_lat1[0],app_lon1[0]])/len(swh[:,app_lat1[0],app_lon1[0]])
maxDelft2 = max(swh[:,app_lat1[1],app_lon1[1]])
minDelft2 = min(swh[:,app_lat1[1],app_lon1[1]])
avgDelft2 = sum(swh[:,app_lat1[1],app_lon1[1]])/len(swh[:,app_lat1[1],app_lon1[1]])
maxDelft3 = max(swh[:,app_lat1[2],app_lon1[2]])
minDelft3 = min(swh[:,app_lat1[2],app_lon1[2]])
avgDelft3 = sum(swh[:,app_lat1[2],app_lon1[2]])/len(swh[:,app_lat1[2],app_lon1[2]])

maxCMEMS1 = max(swh2[:,app_lat2[0],app_lon2[0]])
minCMEMS1 = min(swh2[:,app_lat2[0],app_lon2[0]])
avgCMEMS1 = sum(swh2[:,app_lat2[0],app_lon2[0]])/len(swh2[:,app_lat2[0],app_lon2[0]])
maxCMEMS2 = max(swh2[:,app_lat2[1],app_lon2[1]])
minCMEMS2 = min(swh2[:,app_lat2[1],app_lon2[1]])
avgCMEMS2 = sum(swh2[:,app_lat2[1],app_lon2[1]])/len(swh2[:,app_lat2[1],app_lon2[1]])
maxCMEMS3 = max(swh2[:,app_lat2[2],app_lon2[2]])
minCMEMS3 = min(swh2[:,app_lat2[2],app_lon2[2]])
avgCMEMS3 = sum(swh2[:,app_lat2[2],app_lon2[2]])/len(swh2[:,app_lat2[2],app_lon2[2]])

maxERA1 = max(swh3[:,app_lat3[0],app_lon3[0]])
minERA1 = min(swh3[:,app_lat3[0],app_lon3[0]])
avgERA1 = sum(swh3[:,app_lat3[0],app_lon3[0]])/len(swh3[:,app_lat3[0],app_lon3[0]])
maxERA2 = max(swh3[:,app_lat3[1],app_lon3[1]])
minERA2 = min(swh3[:,app_lat3[1],app_lon3[1]])
avgERA2 = sum(swh3[:,app_lat3[1],app_lon3[1]])/len(swh3[:,app_lat3[1],app_lon3[1]])
maxERA3 = max(swh3[:,app_lat3[2],app_lon3[2]])
minERA3 = min(swh3[:,app_lat3[2],app_lon3[2]])
avgERA3 = sum(swh3[:,app_lat3[2],app_lon3[2]])/len(swh3[:,app_lat3[2],app_lon3[2]])

print(tabulate([['Delft 3D 1',str(maxDelft1),str(minDelft1),str(avgDelft1)], 
                ['Delft 3D 2',str(maxDelft2),str(minDelft2),str(avgDelft2)], 
                ['Delft 3D 3',str(maxDelft3),str(minDelft3),str(avgDelft3)],
                ['CMEMS 1',str(maxCMEMS1),str(minCMEMS1),str(avgCMEMS1)],
                ['CMEMS 2',str(maxCMEMS2),str(minCMEMS2),str(avgCMEMS2)],
                ['CMEMS 3',str(maxCMEMS3),str(minCMEMS3),str(avgCMEMS3)],
                ['ERA 1',str(maxERA1),str(minERA1),str(avgERA1)],
                ['ERA 2',str(maxERA2),str(minERA2),str(avgERA2)],
                ['ERA 3',str(maxERA3),str(minERA3),str(avgERA3)]],
               headers = ['swh','max','min','avg']))
#%% Period
maxDelft1 = max(tp[:,app_lat1[0],app_lon1[0]])
minDelft1 = min(tp[:,app_lat1[0],app_lon1[0]])
avgDelft1 = sum(tp[:,app_lat1[0],app_lon1[0]])/len(tp[:,app_lat1[0],app_lon1[0]])
maxDelft2 = max(tp[:,app_lat1[1],app_lon1[1]])
minDelft2 = min(tp[:,app_lat1[1],app_lon1[1]])
avgDelft2 = sum(tp[:,app_lat1[1],app_lon1[1]])/len(tp[:,app_lat1[1],app_lon1[1]])
maxDelft3 = max(tp[:,app_lat1[2],app_lon1[2]])
minDelft3 = min(tp[:,app_lat1[2],app_lon1[2]])
avgDelft3 = sum(tp[:,app_lat1[2],app_lon1[2]])/len(tp[:,app_lat1[2],app_lon1[2]])

maxCMEMS1 = max(tp2[:,app_lat2[0],app_lon2[0]])
minCMEMS1 = min(tp2[:,app_lat2[0],app_lon2[0]])
avgCMEMS1 = sum(tp2[:,app_lat2[0],app_lon2[0]])/len(tp2[:,app_lat2[0],app_lon2[0]])
maxCMEMS2 = max(tp2[:,app_lat2[1],app_lon2[1]])
minCMEMS2 = min(tp2[:,app_lat2[1],app_lon2[1]])
avgCMEMS2 = sum(tp2[:,app_lat2[1],app_lon2[1]])/len(tp2[:,app_lat2[1],app_lon2[1]])
maxCMEMS3 = max(tp2[:,app_lat2[2],app_lon2[2]])
minCMEMS3 = min(tp2[:,app_lat2[2],app_lon2[2]])
avgCMEMS3 = sum(tp2[:,app_lat2[2],app_lon2[2]])/len(tp2[:,app_lat2[2],app_lon2[2]])

maxERA1 = max(tp3[:,app_lat3[0],app_lon3[0]])
minERA1 = min(tp3[:,app_lat3[0],app_lon3[0]])
avgERA1 = sum(tp3[:,app_lat3[0],app_lon3[0]])/len(tp3[:,app_lat3[0],app_lon3[0]])
maxERA2 = max(tp3[:,app_lat3[1],app_lon3[1]])
minERA2 = min(tp3[:,app_lat3[1],app_lon3[1]])
avgERA2 = sum(tp3[:,app_lat3[1],app_lon3[1]])/len(tp3[:,app_lat3[1],app_lon3[1]])
maxERA3 = max(tp3[:,app_lat3[2],app_lon3[2]])
minERA3 = min(tp3[:,app_lat3[2],app_lon3[2]])
avgERA3 = sum(tp3[:,app_lat3[2],app_lon3[2]])/len(tp3[:,app_lat3[2],app_lon3[2]])

print(tabulate([['Delft 3D 1',str(maxDelft1),str(minDelft1),str(avgDelft1)], 
                ['Delft 3D 2',str(maxDelft2),str(minDelft2),str(avgDelft2)], 
                ['Delft 3D 3',str(maxDelft3),str(minDelft3),str(avgDelft3)],
                ['CMEMS 1',str(maxCMEMS1),str(minCMEMS1),str(avgCMEMS1)],
                ['CMEMS 2',str(maxCMEMS2),str(minCMEMS2),str(avgCMEMS2)],
                ['CMEMS 3',str(maxCMEMS3),str(minCMEMS3),str(avgCMEMS3)],
                ['ERA 1',str(maxERA1),str(minERA1),str(avgERA1)],
                ['ERA 2',str(maxERA2),str(minERA2),str(avgERA2)],
                ['ERA 3',str(maxERA3),str(minERA3),str(avgERA3)]],
               headers = ['period','max','min','avg']))

#%%
#Tinggi Gelombang
fig, ax = plt.subplots(3,1,figsize=(12, 6))
fig.tight_layout(pad=3.0)
ax[0].format_xdata = mdates.DateFormatter('%m-%d')
ax[0].plot(time_list1, swh[:,app_lat1[0],app_lon1[0]], label='Delft3d')
#ax[0].scatter(time_list2, swh2[ext_time2,app_lat2[0],app_lon2[0]], label = 'CMEMS', c= 'orange')
#ax[0].scatter(time_list3[::3], swh3[ext_time3[::3],app_lat3[0],app_lon3[0]], label = 'ECMWF', c= 'green')
ax[0].legend(loc = 5)
ax[0].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave at : Station A'))#+str(ext_lon)+','+str(ext_lat)))

ax[1].format_xdata = mdates.DateFormatter('%m-%d')
ax[1].plot(time_list1, swh[:,app_lat1[1],app_lon1[1]], label='Delft3d')
#ax[1].scatter(time_list2, swh2[ext_time2,app_lat2[1],app_lon2[1]], label = 'CMEMS', c= 'orange')
#ax[1].scatter(time_list3[::3], swh3[ext_time3[::3],app_lat3[1],app_lon3[1]], label = 'ECMWF', c= 'green')
ax[1].legend(loc = 5)
ax[1].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave at : Station B'))
#ax[1].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave in :'+str(ext_lon)+','+str(ext_lat)))

ax[2].format_xdata = mdates.DateFormatter('%m-%d')
ax[2].plot(time_list1, swh[:,app_lat1[2],app_lon1[2]], label='Delft3d')
#ax[2].scatter(time_list2, swh2[ext_time2,app_lat2[2],app_lon2[2]], label = 'CMEMS', c= 'orange')
#ax[2].scatter(time_list3[::3], swh3[ext_time3[::3],app_lat3[2],app_lon3[2]], label = 'ECMWF', c= 'green')
ax[2].legend(loc = 5)
ax[2].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave at : Station C'))
#ax[2].set(xlabel="Date", ylabel="Sign Wave (m)", title=('Sign Wave in :'+str(ext_lon)+','+str(ext_lat)))
#%%
#Periode Gelombang
fig, ax2 = plt.subplots(3,1,figsize=(12, 6))
fig.tight_layout(pad=3.0)
ax2[0].format_xdata = mdates.DateFormatter('%m-%d')
ax2[0].plot(time_list1, tp[:,app_lat1[0],app_lon1[0]], label='Delft3d')
#ax2[0].scatter(time_list2, tp2[ext_time2,app_lat2[0],app_lon2[0]], label = 'CMEMS', c= 'orange')
#ax2[0].scatter(time_list3[::3], tp3[ext_time3[::3],app_lat3[0],app_lon3[0]], label = 'ECMWF', c= 'green')
ax2[0].legend(loc = 5)
ax2[0].set(xlabel="Date", ylabel="Periode (s)", title=('Wave Period at : Station A'))#+str(ext_lon)+','+str(ext_lat)))

ax2[1].format_xdata = mdates.DateFormatter('%m-%d')
ax2[1].plot(time_list1, tp[:,app_lat1[1],app_lon1[1]], label='Delft3d')
#ax2[1].scatter(time_list2, tp2[ext_time2,app_lat2[1],app_lon2[1]], label = 'CMEMS', c= 'orange')
#ax2[1].scatter(time_list3[::3], tp3[ext_time3[::3],app_lat3[1],app_lon3[1]], label = 'ECMWF', c= 'green')
ax2[1].legend(loc = 5)
ax2[1].set(xlabel="Date", ylabel="Periode (s)", title=('Wave Period at : Station B'))#+str(ext_lon)+','+str(ext_lat)))

ax2[2].format_xdata = mdates.DateFormatter('%m-%d')
ax2[2].plot(time_list1, tp[:,app_lat1[2],app_lon1[2]], label='Delft3d')
#ax2[2].scatter(time_list2, tp2[ext_time2,app_lat2[2],app_lon2[2]], label = 'CMEMS', c= 'orange')
#ax2[2].scatter(time_list3[::3], tp3[ext_time3[::3],app_lat3[2],app_lon3[2]], label = 'ECMWF', c= 'green')
ax2[2].legend(loc = 5)
ax2[2].set(xlabel="Date", ylabel="Periode (s)", title=('Wave Period at : Station C'))#+str(ext_lon)+','+str(ext_lat)))


#%%
#Arah Gelombang
md1 = (-1*mdr[:,app_lat1[0],app_lon1[0]]+270)%360
md2 = (-1*mdr[:,app_lat1[1],app_lon1[1]]+270)%360
md3 = (-1*mdr[:,app_lat1[2],app_lon1[2]]+270)%360

fig, ax3 = plt.subplots(3,1,figsize=(12, 6))
fig.tight_layout(pad=3)
ax3[0].format_xdata = mdates.DateFormatter('%m-%d')
ax3[0].plot(time_list1, md1 , label='Delft3d')
#ax3[0].scatter(time_list2, mdr2[ext_time2,app_lat2[0],app_lon2[0]], label = 'CMEMS', c= 'orange')
#ax3[0].scatter(time_list3[::3], mdr3[ext_time3[::3],app_lat3[0],app_lon3[0]] , label = 'ECMWF', c= 'green')
ax3[0].legend(loc = 5)
ax3[0].set(xlabel="Date", ylabel="Direction (degree)", title=('Wave direciton at : Station A'))#+str(ext_lon)+','+str(ext_lat)))

ax3[1].format_xdata = mdates.DateFormatter('%m-%d')
ax3[1].plot(time_list1, md2 , label='Delft3d')
#ax3[1].scatter(time_list2, mdr2[ext_time2,app_lat2[1],app_lon2[1]], label = 'CMEMS', c= 'orange')
#ax3[1].scatter(time_list3[::3], mdr3[ext_time3[::3],app_lat3[1],app_lon3[1]], label = 'ECMWF', c= 'green')
ax3[1].legend(loc = 5)
ax3[1].set(xlabel="Date", ylabel="Direction (degree)", title=('Wave direciton at : Station B'))#+str(ext_lon)+','+str(ext_lat)))

ax3[2].format_xdata = mdates.DateFormatter('%m-%d')
ax3[2].plot(time_list1, md3, label='Delft3d')
#ax3[2].scatter(time_list2, mdr2[ext_time2,app_lat2[2],app_lon2[2]], label = 'CMEMS', c= 'orange')
#ax3[2].scatter(time_list3[::3], mdr3[ext_time3[::3],app_lat3[2],app_lon3[2]], label = 'ECMWF', c= 'green')
ax3[2].legend(loc = 5)
ax3[2].set(xlabel="Date", ylabel="Direction (degree)", title=('Wave direciton at : Station C'))#+str(ext_lon)+','+str(ext_lat)))
#%% Direction
md3[150] = md3[149]
avgDelft1 = sum(md1/len(md1))
avgDelft2 = sum(md2/len(md2))
avgDelft3 = sum(md3/len(md3))

avgCMEMS1 = sum(mdr2[:,app_lat2[0],app_lon2[0]])/len(mdr2[:,app_lat2[0],app_lon2[0]])
avgCMEMS2 = sum(mdr2[:,app_lat2[1],app_lon2[1]])/len(mdr2[:,app_lat2[1],app_lon2[1]])
avgCMEMS3 = sum(mdr2[:,app_lat2[2],app_lon2[2]])/len(mdr2[:,app_lat2[2],app_lon2[2]])

avgERA1 = sum(mdr3[:,app_lat3[0],app_lon3[0]])/len(mdr3[:,app_lat3[0],app_lon3[0]])
avgERA2 = sum(mdr3[:,app_lat3[1],app_lon3[1]])/len(mdr3[:,app_lat3[1],app_lon3[1]])
avgERA3 = sum(mdr3[:,app_lat3[2],app_lon3[2]])/len(mdr3[:,app_lat3[2],app_lon3[2]])

print(tabulate([['Delft 3D 1',str(avgDelft1)], 
                ['Delft 3D 2',str(avgDelft2)], 
                ['Delft 3D 3',str(avgDelft3)],
                ['CMEMS 1',str(avgCMEMS1)],
                ['CMEMS 2',str(avgCMEMS2)],
                ['CMEMS 3',str(avgCMEMS3)],
                ['ERA 1',str(avgERA1)],
                ['ERA 2',str(avgERA2)],
                ['ERA 3',str(avgERA3)]],
                   headers = ['direction','avg']))