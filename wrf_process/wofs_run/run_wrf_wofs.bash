#!/bin/sh
# Runscript for executing WRF within a singularity container
# Created by Billy Faletti <wfaletti@ttu.edu>
# Modified from original by Tyler Wixtrom <tyler.wixtrom@ttu.edu>


# Parse input arguments
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            nproc)              nproc=${VALUE} ;;
            dir)                  dir=${VALUE} ;;
            wrf_nml)          wrf_nml=${VALUE} ;;
            indata)            indata=${VALUE} ;;
	    jid)                  jid=${VALUE} ;;
            *)
    esac


done

# Copy WRF run directory and executables
cd ${dir}
mkdir ${dir}/run
cd ${dir}/run
ln -sf /comsoftware/wrf/WRF-${WRF_VERSION}/run/* ${dir}/run
ln -sf /comsoftware/wrf/WRF-${WRF_VERSION}/main/*.exe ${dir}/run

# Copy in supplied namelist
rm ${dir}/run/namelist.input
cp ${wrf_nml} ${dir}/run/namelist.input

# Copy initial and boundary condition data to run directory
rm ${dir}/run/wrfinput* ${dir}/run/wrfbdy* 
cp ${indata}/* ${dir}/run