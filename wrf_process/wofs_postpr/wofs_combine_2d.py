import glob
import datetime
import sys
import xarray as xr
import wofs_cbook as wcalc
import numpy as np

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


    # Calculate and write probabilities
        # Calculate
            # total
    uh_swt_02 = ds['UH_SWT02'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_swt_02_stats = wcalc.calc_ens_products(uh_swt_02, 3, 5, 9, kernel, 30)

    uh_swt_25 = ds['UH_SWT25'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_swt_25_stats = wcalc.calc_ens_products(uh_swt_25, 3, 5, 9, kernel, 60)

    uh_swt_25_wrf = ds['UH_SWT25_WRF'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_swt_25_wrf_stats = wcalc.calc_ens_products(uh_swt_25_wrf, 3, 5, 9, kernel, 60)

    cref_swt = ds['CREF_SWT'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    cref_swt_stats = wcalc.calc_ens_products(cref_swt, 3, 5, 9, kernel, 40)

            # hourly
    uh_hrswt_02 = ds['UH_HRSWT02'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_hrswt_02_stats = wcalc.calc_ens_products(uh_hrswt_02, 3, 5, 9, kernel, 30)

    uh_hrswt_25 = ds['UH_HRSWT25'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_hrswt_25_stats = wcalc.calc_ens_products(uh_hrswt_25, 3, 5, 9, kernel, 60)

    uh_hrswt_25_wrf = ds['UH_HRSWT25_WRF'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_hrswt_25_wrf_stats = wcalc.calc_ens_products(uh_hrswt_25_wrf, 3, 5, 9, kernel, 60)

    cref_hrswt = ds['CREF_HRSWT'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    cref_hrswt_stats = wcalc.calc_ens_products(cref_hrswt, 3, 5, 9, kernel, 40)

        # 30-min
    uh_30swt_02 = ds['UH_30SWT02'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_30swt_02_stats = wcalc.calc_ens_products(uh_30swt_02, 3, 5, 9, kernel, 30)

    uh_30swt_25 = ds['UH_30SWT25'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_30swt_25_stats = wcalc.calc_ens_products(uh_30swt_25, 3, 5, 9, kernel, 60)

    uh_30swt_25_wrf = ds['UH_30SWT25_WRF'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_30swt_25_wrf_stats = wcalc.calc_ens_products(uh_30swt_25_wrf, 3, 5, 9, kernel, 60)

    cref_30swt = ds['CREF_30SWT'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    cref_30swt_stats = wcalc.calc_ens_products(cref_30swt, 3, 5, 9, kernel, 40)

    # 15-min
    uh_15swt_02 = ds['UH_15SWT02'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_15swt_02_stats = wcalc.calc_ens_products(uh_15swt_02, 3, 5, 9, kernel, 30)

    uh_15swt_25 = ds['UH_15SWT25'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_15swt_25_stats = wcalc.calc_ens_products(uh_15swt_25, 3, 5, 9, kernel, 60)

    uh_15swt_25_wrf = ds['UH_15SWT25_WRF'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    uh_15swt_25_wrf_stats = wcalc.calc_ens_products(uh_15swt_25_wrf, 3, 5, 9, kernel, 60)

    cref_15swt = ds['CREF_15SWT'][:,0,:,:].values
    kernel = wcalc.gauss_kern(size=3)
    cref_15swt_stats = wcalc.calc_ens_products(cref_15swt, 3, 5, 9, kernel, 40)

        # Write

            # total
                # 0-2 km UH
    ds['UH_PROB_02_9KM'] = (['Time', 'south_north', 'west_east'],  np.expand_dims(uh_swt_02_stats[2], 0))
    ds['UH_PROB_02_9KM'].attrs['units'] = 'none'
    ds['UH_PROB_02_9KM'].attrs['description'] = 'Prob of 0-2km UH >30m2/s2 within 9km'

    ds['UH_PROB_02_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_swt_02_stats[3], 0))
    ds['UH_PROB_02_15KM'].attrs['units'] = 'none'
    ds['UH_PROB_02_15KM'].attrs['description'] = 'Prob of 0-2km UH >30m2/s2 within 15km'

    ds['UH_PROB_02_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_swt_02_stats[4], 0))
    ds['UH_PROB_02_27KM'].attrs['units'] = 'none'
    ds['UH_PROB_02_27KM'].attrs['description'] = 'Prob of 0-2km UH >30m2/s2 within 27km'
                # 2-5 km UH
    ds['UH_PROB_25_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_swt_25_stats[2], 0))
    ds['UH_PROB_25_9KM'].attrs['units'] = 'none'
    ds['UH_PROB_25_9KM'].attrs['description'] = 'Prob of 2-5km UH >60m2/s2 within 9km'

    ds['UH_PROB_25_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_swt_25_stats[3], 0))
    ds['UH_PROB_25_15KM'].attrs['units'] = 'none'
    ds['UH_PROB_25_15KM'].attrs['description'] = 'Prob of 2-5km UH >60m2/s2 within 15km'

    ds['UH_PROB_25_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_swt_25_stats[4], 0))
    ds['UH_PROB_25_27KM'].attrs['units'] = 'none'
    ds['UH_PROB_25_27KM'].attrs['description'] = 'Prob of 2-5km UH >60m2/s2 within 27km'
                # WRF-native 2-5 km UH
    ds['UH_PROB_25_WRF_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_swt_25_wrf_stats[2], 0))
    ds['UH_PROB_25_WRF_9KM'].attrs['units'] = 'none'
    ds['UH_PROB_25_WRF_9KM'].attrs['description'] = 'Prob of WRF 2-5km UH >60m2/s2 within 9km'

    ds['UH_PROB_25_WRF_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_swt_25_wrf_stats[3], 0))
    ds['UH_PROB_25_WRF_15KM'].attrs['units'] = 'none'
    ds['UH_PROB_25_WRF_15KM'].attrs['description'] = 'Prob of WRF 2-5km UH >60m2/s2 within 15km'

    ds['UH_PROB_25_WRF_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_swt_25_wrf_stats[4], 0))
    ds['UH_PROB_25_WRF_27KM'].attrs['units'] = 'none'
    ds['UH_PROB_25_WRF_27KM'].attrs['description'] = 'Prob of WRF 2-5km UH >60m2/s2 within 27km'
                # Composite reflectivity
    ds['CREF_PROB_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_swt_stats[2], 0))
    ds['CREF_PROB_9KM'].attrs['units'] = 'none'
    ds['CREF_PROB_9KM'].attrs['description'] = 'Prob of composite reflectivity >40dBZ within 9km'

    ds['CREF_PROB_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_swt_stats[3], 0))
    ds['CREF_PROB_15KM'].attrs['units'] = 'none'
    ds['CREF_PROB_15KM'].attrs['description'] = 'Prob of composite reflectivity >40dBZ within 15km'

    ds['CREF_PROB_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_swt_stats[4], 0))
    ds['CREF_PROB_27KM'].attrs['units'] = 'none'
    ds['CREF_PROB_27KM'].attrs['description'] = 'Prob of composite reflectivity >40dBZ within 27km'

            # hourly
                # 0-2 km UH
    ds['UH_HRPROB_02_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_02_stats[2], 0))
    ds['UH_HRPROB_02_9KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_02_9KM'].attrs['description'] = 'Hourly prob of 0-2km UH >30m2/s2 within 9km'

    ds['UH_HRPROB_02_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_02_stats[3], 0))
    ds['UH_HRPROB_02_15KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_02_15KM'].attrs['description'] = 'Hourly prob of 0-2km UH >30m2/s2 within 15km'

    ds['UH_HRPROB_02_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_02_stats[4], 0))
    ds['UH_HRPROB_02_27KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_02_27KM'].attrs['description'] = 'Hourly prob of 0-2km UH >30m2/s2 within 27km'
                # 2-5 km UH
    ds['UH_HRPROB_25_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_25_stats[2], 0))
    ds['UH_HRPROB_25_9KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_25_9KM'].attrs['description'] = 'Hourly prob of 2-5km UH >60m2/s2 within 9km'

    ds['UH_HRPROB_25_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_25_stats[3], 0))
    ds['UH_HRPROB_25_15KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_25_15KM'].attrs['description'] = 'Hourly prob of 2-5km UH >60m2/s2 within 15km'

    ds['UH_HRPROB_25_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_25_stats[4], 0))
    ds['UH_HRPROB_25_27KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_25_27KM'].attrs['description'] = 'Hourly prob of 2-5km UH >60m2/s2 within 27km'
                # WRF-native 2-5 km UH
    ds['UH_HRPROB_25_WRF_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_25_wrf_stats[2], 0))
    ds['UH_HRPROB_25_WRF_9KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_25_WRF_9KM'].attrs['description'] = 'Hourly prob of WRF 2-5km UH >60m2/s2 within 9km'

    ds['UH_HRPROB_25_WRF_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_25_wrf_stats[3], 0))
    ds['UH_HRPROB_25_WRF_15KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_25_WRF_15KM'].attrs['description'] = 'Hourly prob of WRF 2-5km UH >60m2/s2 within 15km'

    ds['UH_HRPROB_25_WRF_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_hrswt_25_wrf_stats[4], 0))
    ds['UH_HRPROB_25_WRF_27KM'].attrs['units'] = 'none'
    ds['UH_HRPROB_25_WRF_27KM'].attrs['description'] = 'Hourly prob of WRF 2-5km UH >60m2/s2 within 27km'
                # Composite reflectivity
    ds['CREF_HRPROB_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_hrswt_stats[2], 0))
    ds['CREF_HRPROB_9KM'].attrs['units'] = 'none'
    ds['CREF_HRPROB_9KM'].attrs['description'] = 'Hourly prob of composite reflectivity >40dBZ within 9km'

    ds['CREF_HRPROB_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_hrswt_stats[3], 0))
    ds['CREF_HRPROB_15KM'].attrs['units'] = 'none'
    ds['CREF_HRPROB_15KM'].attrs['description'] = 'Hourly prob of composite reflectivity >40dBZ within 15km'

    ds['CREF_HRPROB_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_hrswt_stats[4], 0))
    ds['CREF_HRPROB_27KM'].attrs['units'] = 'none'
    ds['CREF_HRPROB_27KM'].attrs['description'] = 'Hourly prob of composite reflectivity >40dBZ within 27km'

        # 30-min
                # 0-2 km UH
    ds['UH_30PROB_02_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_02_stats[2], 0))
    ds['UH_30PROB_02_9KM'].attrs['units'] = 'none'
    ds['UH_30PROB_02_9KM'].attrs['description'] = '30-min prob of 0-2km UH >30m2/s2 within 9km'

    ds['UH_30PROB_02_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_02_stats[3], 0))
    ds['UH_30PROB_02_15KM'].attrs['units'] = 'none'
    ds['UH_30PROB_02_15KM'].attrs['description'] = '30-min prob of 0-2km UH >30m2/s2 within 15km'

    ds['UH_30PROB_02_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_02_stats[4], 0))
    ds['UH_30PROB_02_27KM'].attrs['units'] = 'none'
    ds['UH_30PROB_02_27KM'].attrs['description'] = '30-min prob of 0-2km UH >30m2/s2 within 27km'

            # 2-5 km UH
    ds['UH_30PROB_25_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_25_stats[2], 0))
    ds['UH_30PROB_25_9KM'].attrs['units'] = 'none'
    ds['UH_30PROB_25_9KM'].attrs['description'] = '30-min prob of 2-5km UH >60m2/s2 within 9km'

    ds['UH_30PROB_25_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_25_stats[3], 0))
    ds['UH_30PROB_25_15KM'].attrs['units'] = 'none'
    ds['UH_30PROB_25_15KM'].attrs['description'] = '30-min prob of 2-5km UH >60m2/s2 within 15km'

    ds['UH_30PROB_25_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_25_stats[4], 0))
    ds['UH_30PROB_25_27KM'].attrs['units'] = 'none'
    ds['UH_30PROB_25_27KM'].attrs['description'] = '30-min prob of 2-5km UH >60m2/s2 within 27km'
                # WRF-native 2-5 km UH
    ds['UH_30PROB_25_WRF_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_25_wrf_stats[2], 0))
    ds['UH_30PROB_25_WRF_9KM'].attrs['units'] = 'none'
    ds['UH_30PROB_25_WRF_9KM'].attrs['description'] = '30-min prob of WRF 2-5km UH >60m2/s2 within 9km'

    ds['UH_30PROB_25_WRF_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_25_wrf_stats[3], 0))
    ds['UH_30PROB_25_WRF_15KM'].attrs['units'] = 'none'
    ds['UH_30PROB_25_WRF_15KM'].attrs['description'] = '30-min prob of WRF 2-5km UH >60m2/s2 within 15km'

    ds['UH_30PROB_25_WRF_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_30swt_25_wrf_stats[4], 0))
    ds['UH_30PROB_25_WRF_27KM'].attrs['units'] = 'none'
    ds['UH_30PROB_25_WRF_27KM'].attrs['description'] = '30-min prob of WRF 2-5km UH >60m2/s2 within 27km'
                # Composite reflectivity
    ds['CREF_30PROB_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_30swt_stats[2], 0))
    ds['CREF_30PROB_9KM'].attrs['units'] = 'none'
    ds['CREF_30PROB_9KM'].attrs['description'] = '30-min prob of composite reflectivity >40dBZ within 9km'

    ds['CREF_30PROB_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_30swt_stats[3], 0))
    ds['CREF_30PROB_15KM'].attrs['units'] = 'none'
    ds['CREF_30PROB_15KM'].attrs['description'] = '30-min prob of composite reflectivity >40dBZ within 15km'

    ds['CREF_30PROB_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_30swt_stats[4], 0))
    ds['CREF_30PROB_27KM'].attrs['units'] = 'none'
    ds['CREF_30PROB_27KM'].attrs['description'] = '30-min prob of composite reflectivity >40dBZ within 27km'

        # 15-min
                # 0-2 km UH
    ds['UH_15PROB_02_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_02_stats[2], 0))
    ds['UH_15PROB_02_9KM'].attrs['units'] = 'none'
    ds['UH_15PROB_02_9KM'].attrs['description'] = '15-min prob of 0-2km UH >30m2/s2 within 9km'

    ds['UH_15PROB_02_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_02_stats[3], 0))
    ds['UH_15PROB_02_15KM'].attrs['units'] = 'none'
    ds['UH_15PROB_02_15KM'].attrs['description'] = '15-min prob of 0-2km UH >30m2/s2 within 15km'

    ds['UH_15PROB_02_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_02_stats[4], 0))
    ds['UH_15PROB_02_27KM'].attrs['units'] = 'none'
    ds['UH_15PROB_02_27KM'].attrs['description'] = '15-min prob of 0-2km UH >30m2/s2 within 27km'
                 # 2-5 km UH
    ds['UH_15PROB_25_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_25_stats[2], 0))
    ds['UH_15PROB_25_9KM'].attrs['units'] = 'none'
    ds['UH_15PROB_25_9KM'].attrs['description'] = '15-min prob of 2-5km UH >60m2/s2 within 9km'

    ds['UH_15PROB_25_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_25_stats[3], 0))
    ds['UH_15PROB_25_15KM'].attrs['units'] = 'none'
    ds['UH_15PROB_25_15KM'].attrs['description'] = '15-min prob of 2-5km UH >60m2/s2 within 15km'

    ds['UH_15PROB_25_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_25_stats[4], 0))
    ds['UH_15PROB_25_27KM'].attrs['units'] = 'none'
    ds['UH_15PROB_25_27KM'].attrs['description'] = '15-min prob of 2-5km UH >60m2/s2 within 27km'
                # WRF-native 2-5 km UH
    ds['UH_15PROB_25_WRF_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_25_wrf_stats[2], 0))
    ds['UH_15PROB_25_WRF_9KM'].attrs['units'] = 'none'
    ds['UH_15PROB_25_WRF_9KM'].attrs['description'] = '15-min prob of WRF 2-5km UH >60m2/s2 within 9km'

    ds['UH_15PROB_25_WRF_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_25_wrf_stats[3], 0))
    ds['UH_15PROB_25_WRF_15KM'].attrs['units'] = 'none'
    ds['UH_15PROB_25_WRF_15KM'].attrs['description'] = '15-min prob of WRF 2-5km UH >60m2/s2 within 15km'

    ds['UH_15PROB_25_WRF_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_15swt_25_wrf_stats[4], 0))
    ds['UH_15PROB_25_WRF_27KM'].attrs['units'] = 'none'
    ds['UH_15PROB_25_WRF_27KM'].attrs['description'] = '15-min prob of WRF 2-5km UH >60m2/s2 within 27km'
                # Composite reflectivity
    ds['CREF_15PROB_9KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_15swt_stats[2], 0))
    ds['CREF_15PROB_9KM'].attrs['units'] = 'none'
    ds['CREF_15PROB_9KM'].attrs['description'] = '15-min prob of composite reflectivity >40dBZ within 9km'

    ds['CREF_15PROB_15KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_15swt_stats[3], 0))
    ds['CREF_15PROB_15KM'].attrs['units'] = 'none'
    ds['CREF_15PROB_15KM'].attrs['description'] = '15-min prob of composite reflectivity >40dBZ within 15km'

    ds['CREF_15PROB_27KM'] = (['Time', 'south_north', 'west_east'], np.expand_dims(cref_15swt_stats[4], 0))
    ds['CREF_15PROB_27KM'].attrs['units'] = 'none'
    ds['CREF_15PROB_27KM'].attrs['description'] = '15-min prob of composite reflectivity >40dBZ within 27km'


    # Save newly-concatenated dataset to file
    outname = ('wofs_i' + str(init_year + init_month + init_day + init_hour + init_min) + 
           '_v' + str(out_year + out_month + out_day + out_hour + out_min) + '.nc')
    
    ds.to_netcdf(str(outname))
