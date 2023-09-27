from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from netCDF4 import Dataset, num2date
import math
#%%
ncfile = Dataset(r'C:\Helwis Liufandy\Tugas Oil Spill\Windrose\DataAngin.nc')
# Create wind speed and direction variables
lon = ncfile.variables['lon'][:]
lat = ncfile.variables['lat'][:]
time = ncfile.variables['time']
u = ncfile.variables['northward_wind']
v = ncfile.variables['eastward_wind']
time_unit = ncfile.variables['time'].getncattr('units')
time_cal = ncfile.variables['time'].getncattr('calendar') 
local_time = num2date(time, units=time_unit, calendar=time_cal)

#%%
ext_lon = 105.770874
ext_lat = -6.001265

sq_dif_lon = (lon - ext_lon)**2
sq_dif_lat = (lat - ext_lat)**2

#minimum value
index_lon = sq_dif_lon.argmin()
index_lat = sq_dif_lat.argmin()
#%%
speed=[]
direction = []
for i in range(0,len(local_time)):
    eastwrd = u[i,index_lat,index_lon]
    northwrd = v[i,index_lat,index_lon]
    print(eastwrd,northwrd)
    ws = math.sqrt(eastwrd**2+northwrd**2)
    speed.append(ws)
    wd3 = np.mod(180+np.rad2deg(np.arctan2(eastwrd, northwrd)),360)
    direction.append(wd3)


#%%

#%%

ax = WindroseAxes.from_ax()
ax.bar(direction, speed, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()

