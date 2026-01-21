# ESA_sensitivity
#
# Chris Weiss - 6/10/21
# Modified by Billy Faletti for 3D calculations - 7/7/22
# Modified by Billy Faletti for bash scripting - 7/10/22
#
# Program produces a netcdf file with sensitivity calculations.  Needs a response and state variable file, which are
# defined at the top.

import math
from scipy import *
from scipy import spatial
from scipy import stats
import numpy as np
import sys
import os
import netCDF4
from optparse import OptionParser
import glob
import datetime

indir = sys.argv[1]
outdir = sys.argv[2]
ob_variance=1
num_members = 36    #needed for degrees of freedom on p-statistic


#Read in files for processing
#
resp_dirs = []
state_dirs = []
member_dirs_temp = os.listdir(indir)
for d, dir in enumerate(member_dirs_temp):
    if (dir[0:4] == 'resp'):  # grabs only directories (within main dir) that begin with "mem"
        resp_dirs.append(dir)
for d, dir in enumerate(member_dirs_temp):
    if (dir[0:5] == 'state'):  # grabs only directories (within main dir) that begin with "mem"
        state_dirs.append(dir)
print(resp_dirs)
for r in range(0,size(resp_dirs)):
    for s in range(0,size(state_dirs)):
        try:
            respin = netCDF4.Dataset(indir + '/' + resp_dirs[r], "r")
            print("Opening response file %s \n" % indir + '/' + resp_dirs[r])
        except:
            print("%s does not exist! \n" %indir + '/' + resp_dirs[r])
            sys.exit(1)
        try:
            statein = netCDF4.Dataset(indir + '/' + state_dirs[s], "r")
            print("Opening state file %s \n" % indir + '/' + state_dirs[s])
        except:
            print("%s does not exist! \n" %indir + '/' + state_dirs[s])
            sys.exit(1)

        ne = statein.dimensions['NE'].size
        nx = statein.dimensions['NX'].size
        ny = statein.dimensions['NY'].size
        if len(statein.dimensions) > 3:
            nz = statein.dimensions['NZ'].size
        statevar = statein.getncattr('STATE_VAR')
        respvar = respin.getncattr('RESP_VAR')
        method = respin.getncattr('METHOD')
        rf_center_lon = respin.getncattr('RF_CENTER_LON')
        rf_center_lat = respin.getncattr('RF_CENTER_LAT')
        rf_xl = respin.getncattr('RF_X_LEFT')
        rf_xr = respin.getncattr('RF_X_RIGHT')
        rf_yb = respin.getncattr('RF_Y_BOTTOM')
        rf_yt = respin.getncattr('RF_Y_TOP')
        resptime = respin.getncattr('RESP_TIME')
        statetime = statein.getncattr('STATE_TIME')

        if len(statein.dimensions) <= 3:
            sens = np.zeros((ny,nx))
            stdsens = np.zeros((ny,nx))
            cov = np.zeros((ny,nx))
            #targ = np.zeros((ny,nx))
            #tstat = np.zeros((ny,nx))
            pstat = np.zeros((ny,nx))
            ts = []
            
            state_var = statein.variables[statevar][:]
            
        if len(statein.dimensions) > 3:
            sens = np.zeros((nz,ny,nx))
            stdsens = np.zeros((nz,ny,nx))
            cov = np.zeros((nz,ny,nx))
            #targ = np.zeros((nz,ny,nx))
            #tstat = np.zeros((nz,ny,nx))
            pstat = np.zeros((nz,ny,nx))
            ts = []
            
            state_var = statein.variables[statevar][:,:nz]
            
        mean_state_var = np.mean(state_var, axis=0)
        var_state_var = np.var(state_var, axis=0, ddof=1)
        resp_var = respin.variables[respvar]

        print('Starting covariance calculations')
        print(datetime.datetime.now())
        for i in range(nx):
            if (i%50==0):
                print("Row "+str(i)+" of "+str(nx))
            for j in range(ny):
                if len(statein.dimensions) <= 3:
                    cov[j,i] = np.cov(state_var[:,j,i],resp_var,ddof=1)[0,-1]
                if len(statein.dimensions) > 3:
                    for k in range(nz):
                        cov[k,j,i] = np.cov(state_var[:,k,j,i],resp_var,ddof=1)[0,-1]

        print('Finished covariance calculations')
        print(datetime.datetime.now())
    
        sens = cov/var_state_var
        stdsens = sens*np.sqrt(var_state_var)
        targ = cov*cov/(var_state_var+ob_variance)
        sens = sens+.00001

        for mem in range(ne):
            ts.append((resp_var[mem] - np.mean(resp_var) - sens*(state_var[mem]-np.mean(state_var,axis=0)))**2)
        ts = np.sum(np.array(ts),axis=0)
        tstat = np.array(sens / np.sqrt(ts/(ne-2)/var_state_var/(ne-1)))
        pstat = np.array((stats.t.sf(np.abs(tstat),(num_members-1))*2)*sens/abs(sens))    



        ### Create output file
        #
        xray = np.zeros((ny,nx))
        yray = np.zeros((ny,nx))

        x = statein.variables['XLONG'][0]
        y = statein.variables['XLAT'][0]

        xray[:,:] = x
        yray[:,:] = y
        
        output_path = outdir+ "/" + "stats_"+respvar.replace('_','-')+"_"+method+"_"+resptime[-8:]+"_"+statevar.replace('_','-')+"_"+statetime[-8:]+".nc"
        
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            fout = netCDF4.Dataset(output_path, "w")
            fout.set_auto_mask(False)
        except:
            print("Could not create %s!\n" % output_path)

        fout.createDimension('NY', ny)
        fout.createDimension('NX', nx)
        if len(statein.dimensions) > 3:
             fout.createDimension('NZ', nz)
        #fout.createDimension('NE', ne)

        header = {'sens':'Sensitivity','stdsens':'Standardized Sensitivity','cov':'Covariance','targ':'Targeting','tstat':'T-Statistic','pstat':'P-value'}

        for title in list(header.keys()):
            if len(statein.dimensions) <= 3:
                outvar = fout.createVariable(title,'f4',('NY','NX',))
            if len(statein.dimensions) > 3:
                outvar = fout.createVariable(title,'f4',('NZ','NY','NX',))
            outvar.long_name = header[title]+" of "+ respvar + " to " + statevar
        lat = fout.createVariable('XLAT','f4',('NY','NX'))
        lon = fout.createVariable('XLONG','f4',('NY','NX'))
        #resp_var_p = fout.createVariable('resp_var','f4',('NE',))
        #state_var_p = fout.createVariable('state_var','f4',('NE','NY','NX',))
        
        fout.variables['sens'][:] = sens
        fout.variables['stdsens'][:] = stdsens
        #fout.variables['targ'][:,:] = targ
        #fout.variables['cov'][:,:] = cov
        #fout.variables['tstat'][:,:] = tstat
        fout.variables['pstat'][:] = pstat
        fout.variables['XLAT'][:] = yray
        fout.variables['XLONG'][:] = xray
        #fout.variables['resp_var'][:] = np.array(resp_var)
        #fout.variables['state_var'][:] = np.array(state_var)

        setattr(fout,'RF_CENTER_LON',rf_center_lon)
        setattr(fout,'RF_CENTER_LAT',rf_center_lat)
        setattr(fout,'RF_X_LEFT',rf_xl)
        setattr(fout,'RF_X_RIGHT',rf_xr)
        setattr(fout,'RF_Y_BOTTOM',rf_yb)
        setattr(fout,'RF_Y_TOP',rf_yt)
        setattr(fout,'RESP_VAR', respvar)
        setattr(fout,'STATE_TIME', statetime)
        setattr(fout,'STATE_VAR', statevar)
        setattr(fout,'METHOD', method)

        respin.close()
        statein.close()
        del respin
        del statein
        fout.close()
        del fout
        print('File created')

print('Sensitivity operation complete')
