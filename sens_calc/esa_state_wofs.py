# ESA_state
# 
# Chris Weiss - 6/9/21
# Modified by Billy Faletti for WoFS - 10/7/21
# Modified by Billy Faletti for bash scripting - 7/10/22
#
# Program produces a netcdf file with a 1x(#member) array of state values
#
# All imports, functions, etc.
#
import math
from scipy import *
from scipy import spatial
import numpy as np
import sys
import os
import netCDF4
from optparse import OptionParser
import datetime as dt
import glob

bytime=0 #State file of all ensemble members for a time (what would be used to generate a time series of state)
bymember=1 #State file of all times for a member (what would be used for sensitivity)
mem=36
indir = sys.argv[1] # directory to pull state data from
outdir = sys.argv[2] # data to output state files to
statevarlist =  os.getenv('statevar').split(',')# list of variables 
dx=3000  #grid resolution (m)
dy=3000  #grid resolution (m)
nx=300  #extent in # gridpoints; grabbed from metadata
ny=300
minx=-999  #index number on desired bounds on state array. -999 means full domain
maxx=-999
miny=-999
maxy=-999
if bymember==1:   #Automatically set scope to 36 members or 15 output times
    ne=36
elif bytime==1:
    ne=15
    
    
for statevar in statevarlist:
    #Read in files for processing
    #
    if bymember==1:
        filenames = []
        member_dirs_temp = os.listdir(indir)
        for d, dir in enumerate(member_dirs_temp):
            if (dir[0:4] == 'wofs'):  # grabs only directories (within main dir) that begin with "mem"
                filenames.append(dir)
        files = []
        for n in range(0, len(filenames)):
            temp_dir = os.path.join(indir, filenames[n])
            temp_files = glob.glob(temp_dir)
            files.append(temp_files)
    elif bytime==1:
        member_dirs = []
        member_dirs_temp = os.listdir(indir + 'wrfprd_mem' + f"{mem:04}")    
        for d, dir in enumerate(member_dirs_temp):
            if (dir[0:5] == 'wrf2d'):  # grabs only directories (within main dir) that begin with "mem"
                member_dirs.append(dir)
        files = []
        tfiles = []    
        for n in range(0, len(member_dirs)):
    #        temp_dir = os.path.join(indir, 'wrfprd_mem' + f"{mem:04}", member_dirs[n])
            temp_dir = glob.glob(indir + 'wrfprd_mem' + f"{mem:04}" + '/' + member_dirs[n])
            files.append(temp_dir) 
            
    ### Grab Files for Time Indicated ###
    
    files.sort()  #should have sorted directory paths to each ensemble file to be processed
    
    #Process files to generate state variable

    if (minx==-999):
        minx=0
        miny=0
        maxx=nx
        maxy=ny

    for f, infile in enumerate(files):
        statetime = (infile[0][-15:-11] + '-' + infile[0][-11:-9] + '-' + infile[0][-9:-7] + 
             '_' + infile[0][-7:-5] + '_' + infile[0][-5:-3] + '_00')
        try:  #open cm1out file
            fin = netCDF4.Dataset(infile[0], "r")
            print("Opening %s \n" % infile)
        except:
            print("%s does not exist! \n" %infile)
            sys.exit(1)

        if (bymember==1):
            outname = "state_" + statevar + "_" + statetime + ".nc"
            output_path = outdir + '/' + outname
        elif (bytime==1):
            outname = "state_" + statevar + "_mem" + str(mem) + ".nc"
            output_path = outdir + '/' +  outname
        # REMOVE THIS FOR LOOP IF SOMETHING IS WRONG 
        # THIS IS ONLY DIFFERENCE IN REFORMING ARRAY STEP OF THIS SCRIPT FROM ORIGINAL
        
        if len(fin.variables[statevar].shape) == 5:
            lvls = fin.variables[statevar].shape[2]
            
            stateray = np.zeros((ne,lvls,(maxy-miny),(maxx-minx)))
            xray = np.zeros((ne,(maxy-miny),(maxx-minx)))
            yray = np.zeros((ne,(maxy-miny),(maxx-minx)))
            zray = np.zeros((ne,(maxy-miny),(maxx-minx)))
            dim = 5
            
        if len(fin.variables[statevar].shape) == 4:
            lvls = 1
            
            stateray = np.zeros((ne,(maxy-miny),(maxx-minx)))
            xray = np.zeros((ne,(maxy-miny),(maxx-minx)))
            yray = np.zeros((ne,(maxy-miny),(maxx-minx)))
            zray = np.zeros((ne,(maxy-miny),(maxx-minx)))
            dim = 4
            
        x = fin.variables['XLONG'][0,0,miny:maxy,minx:maxx]
        y = fin.variables['XLAT'][0,0,miny:maxy,minx:maxx]
        
        print('Iterating levels...')
        for zh in range(lvls):
            print(zh)
            if dim == 5:
                z = fin.variables['Z_MSL'][0,0,0,miny:maxy,minx:maxx] # add a z level before completing this code
            
            for g in range(ne):
                if dim == 5:
                    state = fin.variables[statevar][g, 0, zh, miny:maxy,minx:maxx] # CHANGE G TO ZERO IF INCORRECT
                    stateray[g,zh:,:] = state # CHANGE THESE FOUR G'S TO ZERO IF INCORRECT
                    xray[g,:,:] = x
                    yray[g,:,:] = y
                    zray[g,:,:] = z
                if dim == 4:
                    state = fin.variables[statevar][g, 0, miny:maxy,minx:maxx] # CHANGE G TO ZERO IF INCORRECT
                    stateray[g,:,:] = state # CHANGE THESE FOUR G'S TO ZERO IF INCORRECT
                    xray[g,:,:] = x
                    yray[g,:,:] = y
                    #zray[g,:,:] = z
        # Create state file
        #
        print('Creating nc file ',output_path)
        try:
            fout = netCDF4.Dataset(output_path, "w")
        except:
            print("Could not create %s!\n" % output_path)
        fout.createDimension('NE', ne)
        if dim == 5:
            fout.createDimension('NZ', lvls)
        fout.createDimension('NY', (maxy-miny))
        fout.createDimension('NX', (maxx-minx))
        setattr(fout,'DX',dx)
        setattr(fout,'DY',dy)
        setattr(fout,'STATE_TIME', statetime)
        setattr(fout,'MEMBER', mem)
        setattr(fout,'STATE_VAR', statevar)
        
        if dim == 5:
            state_var = fout.createVariable(statevar, 'f4', ('NE', 'NZ','NY','NX',))
            lat = fout.createVariable('XLAT', 'f4', ('NE','NY','NX'))
            lon = fout.createVariable('XLONG', 'f4', ('NE','NY','NX'))
        
        if dim == 4:
            state_var = fout.createVariable(statevar, 'f4', ('NE','NY','NX',))
            lat = fout.createVariable('XLAT', 'f4', ('NE','NY','NX'))
            lon = fout.createVariable('XLONG', 'f4', ('NE','NY','NX'))


        fout.variables[statevar][:] = stateray
        fout.variables['XLAT'][:] = yray
        fout.variables['XLONG'][:] = xray
        
        fin.close()
        del fin
        fout.close()
        del fout
        print('File created')
        
print('State operation complete')
