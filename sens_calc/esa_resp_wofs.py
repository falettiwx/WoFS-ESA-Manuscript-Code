# ESA_response
#
# Chris Weiss - 6/9/21
# Modified by Billy Faletti for WoFS - 10/7/21
# Modified by Billy Faletti for bash scripting - 7/10/22
#
# Program produces a netcdf file with a 1x(#member) array of response values
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

ne = 36 #number of ensemble members
indir = sys.argv[1]
outdir = sys.argv[2]
respvarlist = os.getenv('respvar').split(',') # sys.argv[3]
zh = int(sys.argv[4]) # vertical level of response function
dx=3000  #grid resolution (m)
dy=3000  #grid resolution (m)
nx=300  #extent in # gridpoints; grabbed from metadata
ny=300
rf_center_lon = float(sys.argv[5])  #gridpoint of center of response function  Nashville -86.78
rf_center_lat = float(sys.argv[6])     
rf_xl = float(sys.argv[7])             #extent of response box, in deg (x-left,y-right,etc.)
rf_xr = float(sys.argv[8])
rf_yb = float(sys.argv[9])
rf_yt = float(sys.argv[10])
methodlist = os.getenv('methodlist').split(',')    #type of response: choices (avg, max, min, sum, gridsum)
threshold = os.getenv('threshold').split(',')       #threshold value for gridsum
condition = os.getenv('condition').split(',')      #condition for gridsum threshold (>, <, =, >=, <=, !=)

for i, method in enumerate(methodlist):
    for respvar in respvarlist:
        
        #Read in files for processing
        #
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

        files.sort()  #should have sorted directory paths to each ensemble file to be processed


        #Process files to generate response function
        #
        for f, infile in enumerate(files):
            resptime = (infile[0][-15:-11] + '-' + infile[0][-11:-9] + '-' + infile[0][-9:-7] + 
             '_' + infile[0][-7:-5] + '_' + infile[0][-5:-3] + '_00')
            try:  #open cm1out file
                fin = netCDF4.Dataset(infile[0], "r")
                #print("Opening %s \n" % infile)
            except:
                print("%s does not exist! \n" %infile)
                sys.exit(1)

            outname = "resp_" + respvar + "_" + method + "_" + resptime + ".nc"
            output_path = outdir + '/' + outname
            response_variables = np.zeros((len(list(respvar[0])),ne))
            
            ### IF SOMETHING IS MESSED UP, ONLY THING FROM ORIGINAL CHANGED BELOW TOP DEFINITIONS IS THIS LOOP
            print('Iterating...')
            for g in range(ne):
                x = fin.variables['XLONG'][0,0]
                y = fin.variables['XLAT'][0,0]
                if len(fin.variables[respvar].shape) == 5:
                    z = fin.variables['Z_MSL'][0,0]
                
                # check for number of dimensions to see if variable is 2D or 3D, then take 2D slice accordingly
                if len(fin.variables[respvar].shape) == 5:
                    resp = fin.variables[respvar][g,0,zh] # IF MESSED UP, CHANGE THIS G TO :
                if len(fin.variables[respvar].shape) == 4:
                    resp = fin.variables[respvar][g,0] # IF MESSED UP, CHANGE THIS G TO :
                
                area  = (x<(rf_center_lon+rf_xr))&(x>(rf_center_lon+rf_xl)) & (y<(rf_center_lat+rf_yt)) & (y>(rf_center_lat+rf_yb))
                r_area = np.where(area,resp,np.nan)
            
                if np.nanmax(r_area) == np.nan:
                    print('THEY ARE ALL NANS')

                if method == "avg":
                    r_trim = np.nanmean(r_area)
                if method == "max":
                    r_trim = np.nanmax(r_area)
                if method == "min":
                    r_trim = np.nanmin(r_area)
                if method == "sum":
                    r_trim = np.nansum(r_area)
                if method == "gridsum":
                    if condition[i] == '>':
                        r_trim = (r_area > float(threshold[i])).sum()
                    if condition[i] == '<':
                        r_trim = (r_area < float(threshold[i])).sum()
                    if condition[i] == '=':
                        r_trim = (r_area == float(threshold[i])).sum()
                    if condition[i] == '>=':
                        r_trim = (r_area >= float(threshold[i])).sum()
                    if condition[i] == '<=':
                        r_trim = (r_area <= float(threshold[i])).sum()
                    if condition[i] == '!=':
                        r_trim = (r_area != float(threshold[i])).sum()
                response_variables[0,g] = r_trim # IF MESSED UP, CHANGE THIS G TO F
                ####### END INSERTED FOR LOOP

            # Create response file
            #
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                fout = netCDF4.Dataset(output_path, "w")
            except:
                print("Could not create %s!\n" % output_path)
    
            fout.createDimension('NE', ne)
            setattr(fout,'DX',dx)
            setattr(fout,'DY',dy)
            setattr(fout,'RF_CENTER_LON',rf_center_lon)
            setattr(fout,'RF_CENTER_LAT',rf_center_lat)
            setattr(fout,'RF_X_LEFT',rf_xl)
            setattr(fout,'RF_X_RIGHT',rf_xr)
            setattr(fout,'RF_Y_BOTTOM',rf_yb)
            setattr(fout,'RF_Y_TOP',rf_yt)
            setattr(fout,'RESP_VAR', respvar)
            setattr(fout,'RESP_TIME', resptime)
            setattr(fout,'METHOD', method)
            
            r1 = fout.createVariable(respvar, 'f4', ('NE',))
            r1.long_name = "{0} response".format(respvar)
            fout.variables[respvar][:] = np.array(response_variables[0,:])
    
            fout.close()
            del fout
            print('File created')
            print(output_path)

print('Response operation complete')
