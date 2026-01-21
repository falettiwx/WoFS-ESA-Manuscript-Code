import xarray as xr
import glob
import datetime
import sys
import metpy.calc as mpcalc
from metpy.units import units

# Set mem_num based on the $mem_num variable in runscript
init_str = sys.argv[1]
outstep = int(sys.argv[2]) # output timestep in minutes
savedir = sys.argv[3] # directory to save to 

init_year = init_str[0:4]
init_month = init_str[4:6]
init_day = init_str[6:8]
init_hour = init_str[8:10]
init_min = init_str[10:12]

# Check time range of WoFS run and determine number of output files
if int(init_min) == 0:
    range_mins = 360
elif int(init_min) == 30:
    range_mins = 180
frames = int((range_mins/outstep) + 1)

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
                      f'wrfout_d01_{out_year}-{out_month}-{out_day}_{out_hour}:{out_min}:00_mem*', 
                    recursive=True))
    print(files)

    # Open files and concatenate along new 'Member' dimension - ties all members into one ds at a given time
    ds = xr.open_mfdataset(files, concat_dim='Member', combine='nested', compat='override', coords='all',
               drop_variables=['ACSWUPT', 'ACSWUPTC', 'ACSWDNT', 'ACSWDNTC', 'ACSWUPB', 'ACSWUPBC', 
                                      'ACSWDNB', 'ACSWDNBC', 'ACLWUPT', 'ACLWUPTC', 'ACLWDNT', 'ACLWDNTC', 
                                      'ACLWUPB', 'ACLWUPBC', 'ACLWDNB', 'ACLWDNBC', 'SWUPT', 'SWUPTC', 
                                      'SWDNT', 'SWDNTC', 'SWUPB', 'SWUPBC', 'SWDNB', 'SWDNBC', 'LWUPT', 'LWUPTC', 
                                      'LWDNT', 'LWDNTC', 'LWUPB', 'LWUPBC', 'LWDNB', 'LWDNBC', 'Times', 'EL_PBL',
                                       'QKE','TKE_PBL','LU_INDEX','ZNU','ZNW','ZS','DZS','VAR_SSO','HFX_FORCE',
                                       'LH_FORCE','TSK_FORCE','HFX_FORCE_TEND','LH_FORCE_TEND','TSK_FORCE_TEND','FNM','FNP','RDNW',
                                        'RDN','DNW','DN','CFN','CFN1','SHDMAX','SHDMIN','SNOALB','TSLB','SMOIS','SH2O',
                                       'SEAICE','XICEM','SFROFF','UDROFF','IVGTYP','ISLTYP','VEGFRA','GRDFLX','ACGRDFLX',
                                       'ACRUNOFF','ACSNOM','SNOW','SNOWH','CANWAT','SSTSK','COSZEN','RHOSNF','SNOWFALLAC',
                                       'LAI','VAR','MAPFAC_M','MAPFAC_U','MAPFAC_V','MAPFAC_MX','MAPFAC_MY','MAPFAC_UX',
                                       'MAPFAC_UY','MAPFAC_VX','MAPFAC_VY','MF_VX_INV','F','E','SINALPHA','COSALPHA',
                                        'ALBEDO','CLAT','ALBBCK','EMISS','NOAHRES','TMN','XLAND','ACKLX','ACLHF','SOILT1',
                                        'SNOWC','SST','SST_INPUT','NEST_POS', 'SWDOWN', 'GLW', 'SWNORM', 'OLR',
                                        'UST', 'HFX', 'QFX', 'LH', 'ACHFX', 'QNCCN', 'QNDROP', 'QNRAIN', 'QNICE',
                                        'QNSNOW', 'QNGRAUPEL', 'QNHAIL', 'QVGRAUPEL', 'QVHAIL', 'RAINC', 'RAINSH',
                                       'RAINNC', 'SNOWNC', 'GRAUPELNC', 'HAILNC', 'RDX', 'RDY', 'RESM', 'ZETATOP', 'CF1', 'CF2', 'CF3',
                                       'ITIMESTEP', 'GRPL_MAX', 'HAIL_MAXK1', 'PREC_ACC_C', 'PREC_ACC_NC', 'SNOW_ACC_NC',
                                       'ISEEDARR_SPPT', 'ISEEDARR_SKEBS', 'ISEEDARR_RAND_PERTURB', 'ISEEDARRAY_SPP_CONV', 
                                       'ISEEDARRAY_SPP_PBL', 'ISEEDARRAY_SPP_LSM', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 
                                       'QGRAUP', 'QHAIL', 'CLDFRA', 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F',
                                       'THM','MU','MUB','THIS_IS_AN_IDEAL_RUN','P_HYD','TSK','T00','P00','TLP','TISO','TLP_STRAT',
                                       'P_STRAT','MAX_MSTFX','MAX_MSTFY','SAVE_TOPO_FROM_REAL','SR','PCB','PC'])
    

    # Calculate new variables
    height_asl = ((ds['PH'] + ds['PHB'])/9.81).values
    pres = (ds['P'] + ds['PB']).values
    theta = ds['T'].values + 300
    temp = mpcalc.temperature_from_potential_temperature(pres*units.pascal, theta*units.kelvin).magnitude
    vpres = mpcalc.vapor_pressure(pres*units.pascal, ds['QVAPOR'].values)
    td = mpcalc.dewpoint(vpres).magnitude
    theta_e = mpcalc.equivalent_potential_temperature(pres*units.pascal, temp*units.kelvin, td*units.degC).magnitude

    temp_2m = ds['T2'].values
    pres_2m = ds['PSFC'].values
    vpres_2m = mpcalc.vapor_pressure(pres_2m*units.pascal, ds['Q2'].values)
    td_2m = mpcalc.dewpoint(vpres_2m).magnitude
    theta_e_2m = mpcalc.equivalent_potential_temperature(pres_2m*units.pascal, temp_2m*units.kelvin, td_2m*units.degC).magnitude
    
    # Drop variables unneeded after calculations, then add calculated variables to dataset
    data = ds.drop(['PH', 'PHB', 'P', 'PB', 'T', 'QVAPOR', 'Q2', 'HGT'])

    ds['P'] = (['Member', 'Time', 'bottom_top', 'south_north', 'west_east'], pres)
    ds['P'].attrs['units'] = 'Pascal'
    ds['P'].attrs['description'] = 'Barometric pressure at each gridpoint'
    
    ds['Z_MSL'] = (['Member', 'Time', 'bottom_top_stag', 'south_north', 'west_east'], height_asl)
    ds['Z_MSL'].attrs['units'] = 'm'
    ds['Z_MSL'].attrs['description'] = 'Elevation of each gridpoint above sea level'
    
    ds['T'] = (['Member', 'Time', 'bottom_top', 'south_north', 'west_east'], temp)
    ds['T'].attrs['units'] = 'K'
    ds['T'].attrs['description'] = 'Temperature at each gridpoint'
    
    ds['TD'] = (['Member', 'Time', 'bottom_top', 'south_north', 'west_east'], td)
    ds['TD'].attrs['units'] = 'K'
    ds['TD'].attrs['description'] = 'Dewpoint temperature at each gridpoint'
    
    ds['THE'] = (['Member', 'Time', 'bottom_top', 'south_north', 'west_east'], theta_e)
    ds['THE'].attrs['units'] = 'K'
    ds['THE'].attrs['description'] = 'Theta-e at each gridpoint'
    
    ds['T2'] = (['Member', 'Time', 'south_north', 'west_east'], temp_2m)
    ds['T2'].attrs['units'] = 'K'
    ds['T2'].attrs['description'] = '2-meter temperature'
    
    ds['TD2'] = (['Member', 'Time', 'south_north', 'west_east'], temp_2m)
    ds['TD2'].attrs['units'] = 'K'
    ds['TD2'].attrs['description'] = '2-meter dewpoint temperature'
    
    ds['THE2'] = (['Member', 'Time', 'south_north', 'west_east'], theta_e_2m)
    ds['THE2'].attrs['units'] = 'K'
    ds['THE2'].attrs['description'] = '2-meter theta-e'


    # Save newly-concatenated dataset to new file
    outname = ('wofs_i' + str(init_year + init_month + init_day + init_hour + init_min) + 
           '_v' + str(out_year + out_month + out_day + out_hour + out_min) + '.nc')

    ds.to_netcdf(str(outname))
