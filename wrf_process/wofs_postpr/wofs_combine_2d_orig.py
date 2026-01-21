import glob
import datetime
import sys
import xarray as xr

# Set mem_num based on the $mem_num variable in runscript
init_str = str(sys.argv[1])
outstep = int(sys.argv[2]) # output timestep in minutes
savedir = sys.argv[3] # directory to save to 

init_year = init_str[0:4]
init_month = init_str[4:6]
init_day = init_str[6:8]
init_hour = init_str[9:11]
init_min = init_str[11:13]

# Check time range of WoFS run and determine number of output files
if int(init_min) == 0:
    range_mins = 360
elif int(init_min) == 30:
    range_mins = 180
frames =  int((range_mins/outstep) + 1)

for i in range(frames):
    
    # Define a datetime object to easily calculate timesteps with timedelta
    init_dt = datetime.datetime(int(init_year), int(init_month), int(init_day), int(init_hour), int(init_min))
    out_dt = init_dt + (i*datetime.timedelta(minutes=outstep))
    
    # Extract output time strings from output datetime object
    out_year = ('%02d' % out_dt.year)
    out_month = ('%02d' % out_dt.month)
    out_day = ('%02d' % out_dt.day)
    out_hour = ('%02d' % out_dt.hour)
    out_min = ('%02d' % out_dt.minute)
   
    print('Made it to glob')

    # Find file corresponding to the output time for each member
    files = sorted(glob.glob(f'/lustre/scratch/wfaletti/wofs/wofs{init_year}'
                      f'{init_month}{init_day}_{init_hour}{init_min}/**/'
                      f'wrfout_d01_{out_year}-{out_month}-{out_day}_{out_hour}:{out_min}:00_mem*.nc', 
                    recursive=True))
    
    print(files)

    # Open files and concatenate along new 'Member' dimension - ties all members into one ds at a given time
    ds = xr.open_mfdataset(files, concat_dim='Member', combine='nested', compat='override', coords='all')

    # Save newly-concatenated dataset to file
    outname = ('wofs_i' + str(init_year + init_month + init_day + init_hour + init_min) + 
           '_v' + str(out_year + out_month + out_day + out_hour + out_min) + '.nc')
    
    ds.to_netcdf(str(outname))
