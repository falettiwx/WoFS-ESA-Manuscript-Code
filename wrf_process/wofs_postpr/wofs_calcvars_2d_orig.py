import numpy as np
import xarray as xr
import glob
import sys
from metpy.interpolate import interpolate_1d, log_interpolate_1d
from metpy.units import units
import metpy.calc as mpcalc
import wofs_cbook as wcalc

dirs = sys.argv[1:37] # previously defined as indirs which fed into commented out 'dirs' variable below
print(dirs)
#dirs = sorted(glob.glob(indirs))
print(dirs)
print('len(dirs)',len(dirs))
print('dirs[0]',dirs[0])
for i in range(len(dirs)):
    mem_str = ('%02d' % (i+1))
    
    files = sorted(glob.glob(dirs[i]+'/wrfout_*'))
    print(files)    
    for j in range(len(files)):
        ds=xr.open_dataset(files[j], 
                    drop_variables=[
                        'ACSWUPT', 'ACSWUPTC', 'ACSWDNT', 'ACSWDNTC', 'ACSWUPB', 'ACSWUPBC', 
                         'ACSWDNB', 'ACSWDNBC', 'ACLWUPT', 'ACLWUPTC', 'ACLWDNT', 'ACLWDNTC', 
                         'ACLWUPB', 'ACLWUPBC', 'ACLWDNB', 'ACLWDNBC', 'SWUPT', 'SWUPTC', 
                         'SWDNT', 'SWDNTC', 'SWUPB', 'SWUPBC', 'SWDNB', 'SWDNBC', 'LWUPT', 'LWUPTC', 
                         'LWDNT', 'LWDNTC', 'LWUPB', 'LWUPBC', 'LWDNB', 'LWDNBC', 'Times', 'EL_PBL',
                         'QKE','TKE_PBL','LU_INDEX','ZNU','ZNW','ZS','DZS','VAR_SSO','HFX_FORCE','LH_FORCE',
                        'TSK_FORCE','HFX_FORCE_TEND','LH_FORCE_TEND','TSK_FORCE_TEND','FNM','FNP','RDNW',
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
                           'THM', 'MU', 'MUB', 'THIS_IS_AN_IDEAL_RUN', 'P_HYD', 'TSK', 'T00', 'P00','TLP',
                        'TISO','TLP_STRAT','P_STRAT','MAX_MSTFX','MAX_MSTFY','SAVE_TOPO_FROM_REAL','SR','PCB','PC'
                                ]
                )
        #ds.load()

        # Define state variables
        p = (ds['P'] + ds['PB'])[0].values/100 #convert to hPa
        th = ds['T'][0].values + 300
        qv = ds['QVAPOR'][0].values
        u = ds['U'][0].values
        v = ds['V'][0].values
        t = wcalc.calc_t(th, p)
        td = wcalc.calc_td(t, p, qv)
        th_e, t_star = wcalc.calc_the_bolt(p, t, qv)

        t_2m = ds['T2'][0].values
        psfc = ds['PSFC'][0].values/100 #convert to hPa
        qv_2m = ds['Q2'][0].values
        td_2m = wcalc.calc_td(t_2m, psfc, qv_2m)
        theta_e_2m, t_star_2m = wcalc.calc_the_bolt(psfc, t_2m, qv_2m)
        rh = wcalc.relative_humidity(t, p, qv)
        rh_2m = wcalc.relative_humidity(t_2m, psfc, qv_2m)

        # Define simulated reflectivity
        dbz = ds['REFL_10CM'][0].values

        # Define and destagger U and V winds
        u_stag = ds['U'][0].values
        v_stag = ds['V'][0].values
        u = 0.5*(u_stag[:,:,:-1]+u_stag[:,:,1:])
        v = 0.5*(v_stag[:,:-1,:]+v_stag[:,1:,:])

        # Destagger height and vertical velocity, calculate vertical grid spacing
        w_stag = ds['W'][0].values
        z_stag = ((ds['PH'][0] + ds['PHB'][0])/9.81).values - ds['HGT'][0].values
        w = 0.5*(w_stag[:-1,:,:]+w_stag[1:,:,:])
        z = 0.5*(z_stag[:-1,:,:]+z_stag[1:,:,:])
        dz = z_stag[1:,:,:] - z_stag[:-1,:,:]
        
        
        # calculate CAPE and CIN

            # calculate 100 mb mixed layer
        mltemp = wcalc.calc_mixed_layer(t, psfc, p, dz, level=100) # ml temp
        mlthe = wcalc.calc_mixed_layer(th_e, psfc, p, dz, level=100) # ml theta e
        mlpres = wcalc.calc_mixed_layer(p, psfc, p, dz, level=100) # ml pressure
        mldew = wcalc.calc_mixed_layer(td, psfc, p, dz, level=100) # ml dewpoint
                # plug in variables in to calculate ml and sb parcel profiles
        t_mlparc = wcalc.calc_parcel_dj(p, mlthe, mltemp, mlpres)
        t_sbparc = wcalc.calc_parcel_dj(p, theta_e_2m, t_2m, psfc)
                # calculate ml and sb lcl for cape calculations
        mllcl_p, mllcl_t = mpcalc.lcl(np.ma.getdata(mlpres)*units.hPa, np.ma.getdata(mltemp)*units.kelvin, np.ma.getdata(mldew)*units.kelvin)
        sblcl_p, sblcl_t = mpcalc.lcl(psfc*units.hPa, t_2m*units.kelvin, td_2m*units.kelvin)
                # calculate ml and sbcape
        mlcape = wcalc.calc_cape(t, t_mlparc, p, mllcl_p.magnitude, dz)
        sbcape = wcalc.calc_cape(t, t_sbparc, p, sblcl_p.magnitude, dz)
                # calculate ml and sb cin
        mlcin = wcalc.calc_cin(t, t_mlparc, p, mllcl_p.magnitude, dz)
        sbcin = wcalc.calc_cin(t, t_sbparc, p, sblcl_p.magnitude, dz)
    
            # calculate SRH
        brm_u, brm_v = wcalc.calc_bunkers(p, z, dz, u, v)[0:2]
        srh_1 = wcalc.calc_srh(z, u, v, dz, 0, 1000, brm_u, brm_v)
        srh_3 = wcalc.calc_srh(z, u, v, dz, 0, 3000, brm_u, brm_v)
    
            # calculate 0-1km and 0-6 km shear
        bwd_u1, bwd_v1 = wcalc.calc_wind_shear(u, z, 0, 1000), wcalc.calc_wind_shear(v, z, 0, 1000)
        bwd_u6, bwd_v6 = wcalc.calc_wind_shear(u, z, 0, 6000), wcalc.calc_wind_shear(v, z, 0, 6000)

            # calculate vertical vorticity for layer-average vorticity, UH, and Okubo-Weiss variables
        vort = mpcalc.vorticity(u*units.meter_per_second, v*units.meter_per_second, dx=3000*units.meter, dy=3000*units.meter).magnitude
                # calculate 0-2 and 2-5 km vertical vorticitiy
        vort02 = wcalc.calc_avg_vort(vort, z, dz, 0, 2000)
        vort25 = wcalc.calc_avg_vort(vort, z, dz, 2000, 5000)
                # calculate 0-2 and 2-5 km UH
        uh_02 = wcalc.calc_uh(w, vort, z, dz, 0, 2000)
        uh_25 = wcalc.calc_uh(w, vort, z, dz, 2000, 5000)
                # calculate okubo-weiss
        #owp_02 = wcalc.calc_okubo_weiss(vort, u_stag, v_stag, 3000, 0, 2000)
        #owp_25 = wcalc.calc_okubo_weiss(vort, u_stag, v_stag, 3000, 2000, 5000)
                # calculate max 0-1 km updraft
        llupdr = wcalc.calc_low_level_updraft(w, z).values
                # calculate echo tops
        etop = wcalc.calc_echo_tops(dbz, z) # calculate in kft
                # calculate 1-km reflectivity
        dbz_1km = interpolate_1d(1000*units.meter, z*units.meter, dbz, axis=0)
        
    # Interpolate desired variables to mandatory pressure levels

        t850, td850, q850, rh850, u850, v850 = log_interpolate_1d(850*units.hPa, p*units.hPa, t*units.kelvin, td*units.kelvin,
                                            qv, rh, u*units.meter_per_second, v*units.meter_per_second, axis=0)
        t700, td700, q700, rh700, u700, v700 = log_interpolate_1d(700*units.hPa, p*units.hPa, t*units.kelvin, td*units.kelvin,
                                            qv, rh, u*units.meter_per_second, v*units.meter_per_second, axis=0)
        t500, td500, q500, rh500, u500, v500 = log_interpolate_1d(500*units.hPa, p*units.hPa, t*units.kelvin, td*units.kelvin,
                                            qv, rh, u*units.meter_per_second, v*units.meter_per_second, axis=0)
        #t300, td300, q300, rh300, u300, v300 = log_interpolate_1d(300*units.hpa, p*units.hPa, t*units.kelvin, td*units.kelvin,
                                            #qv, rh, u*units.meter_per_second, v*units.meter_per_second, axis=0)
        
        
        # Remove and replace with calculated variables
        ds = ds.drop(['PH', 'PHB', 'P', 'PB', 'T', 'QVAPOR', 'Q2', 'HGT', 
                      'U', 'V', 'W', 'TH2', 'PSFC', 'P_TOP'])
        
        ds['TD2'] = (['Time', 'south_north', 'west_east'], np.expand_dims(t_2m, 0))
        ds['TD2'].attrs['units'] = 'Kelvin'
        ds['TD2'].attrs['description'] = '2-meter dewpoint temperature'
    
        ds['THE2'] = (['Time', 'south_north', 'west_east'], np.expand_dims(theta_e_2m, 0))
        ds['THE2'].attrs['units'] = 'K'
        ds['THE2'].attrs['description'] = '2-meter theta-e'

        ds['RH2'] = (['Time', 'south_north', 'west_east'], np.expand_dims(rh_2m, 0))
        ds['RH2'].attrs['units'] = 'K'
        ds['RH2'].attrs['description'] = '2-meter relative humidity'

        ds['SBCAPE'] = (['Time', 'south_north', 'west_east'], np.expand_dims(sbcape, 0))
        ds['SBCAPE'].attrs['units'] = 'j/kg'
        ds['SBCAPE'].attrs['description'] = 'Surface-based CAPE'

        ds['MLCAPE'] = (['Time', 'south_north', 'west_east'], np.expand_dims(mlcape, 0))
        ds['MLCAPE'].attrs['units'] = 'j/kg'
        ds['MLCAPE'].attrs['description'] = '100-mb mixed-layer CAPE'

        ds['SBCIN'] = (['Time', 'south_north', 'west_east'], np.expand_dims(sbcin, 0))
        ds['SBCIN'].attrs['units'] = 'j/kg'
        ds['SBCIN'].attrs['description'] = 'Surface-based CIN'

        ds['MLCIN'] = (['Time', 'south_north', 'west_east'], np.expand_dims(mlcin, 0))
        ds['MLCIN'].attrs['units'] = 'j/kg'
        ds['MLCIN'].attrs['description'] = '100-mb mixed-layer CIN'

        ds['SRH1'] = (['Time', 'south_north', 'west_east'], np.expand_dims(srh_1, 0))
        ds['SRH1'].attrs['units'] = 'm2/s2'
        ds['SRH1'].attrs['description'] = '0-1 km SRH'
    
        ds['SRH3'] = (['Time', 'south_north', 'west_east'], np.expand_dims(srh_3, 0))
        ds['SRH3'].attrs['units'] = 'm2/s2'
        ds['SRH3'].attrs['description'] = '0-3 km SRH'

        ds['SHEAR_U1'] = (['Time', 'south_north', 'west_east'], np.expand_dims(bwd_u1, 0))
        ds['SHEAR_U1'].attrs['units'] = 'm/s'
        ds['SHEAR_U1'].attrs['description'] = '0-1 km U bulk wind difference'

        ds['SHEAR_V1'] = (['Time', 'south_north', 'west_east'], np.expand_dims(bwd_v1, 0))
        ds['SHEAR_V1'].attrs['units'] = 'm/s'
        ds['SHEAR_V1'].attrs['description'] = '0-1 km V bulk wind difference'
    
        ds['SHEAR_U6'] = (['Time', 'south_north', 'west_east'], np.expand_dims(bwd_u6, 0))
        ds['SHEAR_U6'].attrs['units'] = 'm/s'
        ds['SHEAR_U6'].attrs['description'] = '0-6 km U bulk wind difference'

        ds['SHEAR_V6'] = (['Time', 'south_north', 'west_east'], np.expand_dims(bwd_v6, 0))
        ds['SHEAR_V6'].attrs['units'] = 'm/s'
        ds['SHEAR_V6'].attrs['description'] = '0-6 km V bulk wind difference'
    
        ds['WZ_02'] = (['Time', 'south_north', 'west_east'], np.expand_dims(vort02, 0))
        ds['WZ_02'].attrs['units'] = 's^-1'
        ds['WZ_02'].attrs['description'] = '0-2 km average vertical vorticity'

        ds['WZ_25'] = (['Time', 'south_north', 'west_east'], np.expand_dims(vort25, 0))
        ds['WZ_25'].attrs['units'] = 's^-1'
        ds['WZ_25'].attrs['description'] = '2-5 km average vertical vorticity'
    
        ds['UH_02'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_02, 0))
        ds['UH_02'].attrs['units'] = 'm2/s2'
        ds['UH_02'].attrs['description'] = '0-2 km updraft helicity'

        ds['UH_25'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_25, 0))
        ds['UH_25'].attrs['units'] = 'm2/s2'
        ds['UH_25'].attrs['description'] = '2-5 km updraft helicity'
    
        #ds['OWP_02'] = (['Time', 'south_north', 'west_east'], np.expand_dims(owp_02, 0))
        #ds['OWP_02'].attrs['units'] = 's^-2'
        #ds['OWP_02'].attrs['description'] = '0-2 km Okubo-Weiss parameter'
    
        #ds['OWP_25'] = (['Time', 'south_north', 'west_east'], np.expand_dims(owp_25, 0))
        #ds['OWP_25'].attrs['units'] = 's^-2'
        #ds['OWP_25'].attrs['description'] = '2-5 km Okubo-Weiss parameter'

        ds['UH_02'] = (['Time', 'south_north', 'west_east'], np.expand_dims(uh_02, 0))
        ds['UH_02'].attrs['units'] = 'm2/s2'
        ds['UH_02'].attrs['description'] = '0-2 km updraft helicity'
    
        ds['LLUP'] = (['Time', 'south_north', 'west_east'], np.expand_dims(llupdr, 0))
        ds['LLUP'].attrs['units'] = 'm/s'
        ds['LLUP'].attrs['description'] = 'Maximum 0-1 km updraft'
        
        ds['ET'] = (['Time', 'south_north', 'west_east'], np.expand_dims(etop, 0))
        ds['ET'].attrs['units'] = 'kft'
        ds['ET'].attrs['description'] = 'Echo top height)'
    
        ds['REFL_1km'] = (['Time', 'south_north', 'west_east'], dbz_1km)
        ds['REFL_1km'].attrs['units'] = 'dBZ'
        ds['REFL_1km'].attrs['description'] = '1-km AGL reflectivity' 
        
        ds['T500'] = (['Time', 'south_north', 'west_east'], t500.magnitude)
        ds['T500'].attrs['units'] = 'K'
        ds['T500'].attrs['description'] = '500-hPa temperature' 
        
        ds['TD500'] = (['Time', 'south_north', 'west_east'], td500.magnitude)
        ds['TD500'].attrs['units'] = 'K'
        ds['TD500'].attrs['description'] = '500-hPa dewpoint temperature' 
        
        ds['Q500'] = (['Time', 'south_north', 'west_east'], q500)
        ds['Q500'].attrs['units'] = 'K'
        ds['Q500'].attrs['description'] = '500-hPa water vapor mixing ratio' 
        
        #ds['RH500'] = (['Time', 'south_north', 'west_east'], rh500)
        #ds['RH500'].attrs['units'] = ''
        #ds['RH500'].attrs['description'] = '500-hPa relative humidity' 
        
        ds['U500'] = (['Time', 'south_north', 'west_east'], u500.magnitude)
        ds['U500'].attrs['units'] = 'm/s'
        ds['U500'].attrs['description'] = '500-hPa U-wind' 
        
        ds['V500'] = (['Time', 'south_north', 'west_east'], v500.magnitude)
        ds['V500'].attrs['units'] = 'm/s'
        ds['V500'].attrs['description'] = '500-hPa V-wind' 
        
        ds['T700'] = (['Time', 'south_north', 'west_east'], t700.magnitude)
        ds['T700'].attrs['units'] = 'K'
        ds['T700'].attrs['description'] = '700-hPa temperature' 
        
        ds['TD700'] = (['Time', 'south_north', 'west_east'], td700.magnitude)
        ds['TD700'].attrs['units'] = 'K'
        ds['TD700'].attrs['description'] = '700-hPa dewpoint temperature' 
        
        ds['Q700'] = (['Time', 'south_north', 'west_east'], q700)
        ds['Q700'].attrs['units'] = 'K'
        ds['Q700'].attrs['description'] = '700-hPa water vapor mixing ratio' 
        
        #ds['RH700'] = (['Time', 'south_north', 'west_east'], rh700)
        #ds['RH700'].attrs['units'] = ''
        #ds['RH700'].attrs['description'] = '500-hPa relative humidity' 
        
        ds['U700'] = (['Time', 'south_north', 'west_east'], u700.magnitude)
        ds['U700'].attrs['units'] = 'm/s'
        ds['U700'].attrs['description'] = '700-hPa U-wind' 
        
        ds['V700'] = (['Time', 'south_north', 'west_east'], v700.magnitude)
        ds['V700'].attrs['units'] = 'm/s'
        ds['V700'].attrs['description'] = '700-hPa V-wind' 
        
        ds['T850'] = (['Time', 'south_north', 'west_east'], t850.magnitude)
        ds['T850'].attrs['units'] = 'K'
        ds['T850'].attrs['description'] = '850-hPa temperature' 
        
        ds['TD850'] = (['Time', 'south_north', 'west_east'], td850.magnitude)
        ds['TD850'].attrs['units'] = 'K'
        ds['TD850'].attrs['description'] = '850-hPa dewpoint temperature' 
        
        ds['Q850'] = (['Time', 'south_north', 'west_east'], q850)
        ds['Q850'].attrs['units'] = 'K'
        ds['Q850'].attrs['description'] = '850-hPa water vapor mixing ratio' 
        
        #ds['RH850'] = (['Time', 'south_north', 'west_east'], rh850)
        #ds['RH850'].attrs['units'] = ''
        #ds['RH850'].attrs['description'] = '850-hPa relative humidity' 
        
        ds['U850'] = (['Time', 'south_north', 'west_east'], u850.magnitude)
        ds['U850'].attrs['units'] = 'm/s'
        ds['U850'].attrs['description'] = '850-hPa U-wind' 
        
        ds['V850'] = (['Time', 'south_north', 'west_east'], v850.magnitude)
        ds['V850'].attrs['units'] = 'm/s'
        ds['V850'].attrs['description'] = '850-hPa V-wind' 

        # Calculate swaths
            # if files is empty list (meaning this is first timestep), copy first timestep output as swath
        if [s + '.nc' for s in files[:j]] == []:
            uh_swt_02 = np.expand_dims(uh_02, 0)
            uh_swt_25 = np.expand_dims(uh_25, 0)
            uh_swt_wrf25 = ds['UP_HELI_MAX'].values
            wz_swt_02 = np.expand_dims(vort02, 0)
            wz_swt_25 = np.expand_dims(vort25, 0)
            dbz_swt = ds['REFD_MAX'].values
            
            #if not first file, open all previous files and take maximum value of all times
        else:
                # open all files already processed (these will contain '.nc' extension)
            ds_swt = xr.open_mfdataset([s + '.nc' for s in files[:j]], concat_dim='Time', 
                             combine='nested', compat='override', coords='all', 
                                drop_variables= ['UH_SWT02','UH_SWT25','UH_SWT25_WRF',
                                                'WZ_SWT02', 'WZ_SWT25', 'CREF_SWT'])

                # append current dataset to previous datasets by variables to calculate swaths for
            swt_vars = ['UP_HELI_MAX','UH_25','UH_02','WZ_02','WZ_25','REFD_MAX']
            ds_swt = xr.concat([ds[swt_vars], ds_swt[swt_vars]], dim='Time')
                # calculate swaths
            uh_swt_02 = np.expand_dims(np.amax(ds_swt['UH_02'].values, 0), 0)
            uh_swt_25 = np.expand_dims(np.amax(ds_swt['UH_25'].values, 0), 0)
            uh_swt_wrf25 = np.expand_dims(np.amax(ds_swt['UP_HELI_MAX'].values, 0), 0)
            wz_swt_02 = np.expand_dims(np.amax(ds_swt['WZ_02'].values, 0), 0)
            wz_swt_25 = np.expand_dims(np.amax(ds_swt['WZ_25'].values, 0), 0)
            dbz_swt = np.expand_dims(np.amax(ds_swt['REFD_MAX'].values, 0), 0)
            print(ds_swt['UH_02'].values[0, 0, 0])

        ds['UH_SWT02'] = (['Time', 'south_north', 'west_east'], uh_swt_02)
        ds['UH_SWT02'].attrs['units'] = 'm2/s2'
        ds['UH_SWT02'].attrs['description'] = '0-2 km UH swaths' 
        
        ds['UH_SWT25'] = (['Time', 'south_north', 'west_east'], uh_swt_25)
        ds['UH_SWT25'].attrs['units'] = 'm2/s2'
        ds['UH_SWT25'].attrs['description'] = '2-5 km UH swaths' 
        
        ds['UH_SWT25_WRF'] = (['Time', 'south_north', 'west_east'], uh_swt_wrf25)
        ds['UH_SWT25_WRF'].attrs['units'] = 'm2/s2'
        ds['UH_SWT25_WRF'].attrs['description'] = '2-5 km UH swaths from raw WRF UH output' 
        
        ds['WZ_SWT02'] = (['Time', 'south_north', 'west_east'], wz_swt_02)
        ds['WZ_SWT02'].attrs['units'] = 'm2/s2'
        ds['WZ_SWT02'].attrs['description'] = '0-2 km vertical vorticity swaths'
        
        ds['WZ_SWT25'] = (['Time', 'south_north', 'west_east'], wz_swt_25)
        ds['WZ_SWT25'].attrs['units'] = 'm2/s2'
        ds['WZ_SWT25'].attrs['description'] = '2-5 km vertical vorticity swaths'
        
        ds['CREF_SWT'] = (['Time', 'south_north', 'west_east'], dbz_swt)
        ds['CREF_SWT'].attrs['units'] = 'm2/s2'
        ds['CREF_SWT'].attrs['description'] = 'Maximum composite reflectivity swaths'
        
        ds.close()
        ds.to_netcdf(files[j]+'.nc')
        print('File created')


