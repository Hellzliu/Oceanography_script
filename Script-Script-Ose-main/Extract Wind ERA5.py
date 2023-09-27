from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
data = Dataset(r'E:/Data/Proyek/Atepoki/Lama/Mesh Bulat/Mike/Angin/Angin1.nc')
lon = data.variables['longitude']
lat = data.variables['latitude']
time = data.variables['time']
u = data.variables['u10'][:,0,0,0]
v = data.variables['v10'][:,0,0,0]
time_unit = data.variables['time'].getncattr('units')
time_cal = data.variables['time'].getncattr('calendar') 
local_time = num2date(time, units=time_unit, calendar=time_cal)

x = u**2+v**2
speed = np.sqrt(x)
direction = np.mod(180+np.rad2deg(np.arctan2(u, v)),360)

d = {'Time':local_time,'Speed': speed, 'direction': direction}
df = pd.DataFrame(data = d, index = None )
