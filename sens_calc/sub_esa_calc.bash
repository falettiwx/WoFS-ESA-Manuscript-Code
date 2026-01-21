#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=sens_calc
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition nocona
#SBATCH --account=default
#SBATCH --nodes=1 --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --array=1-1:1
#SBATCH --account=default


module load gcc python

# Activate conda environment
source /home/wfaletti/miniconda3/etc/profile.d/conda.sh
conda activate sens_calc

state_indir=/lustre/work/wfaletti/wofs/sens_calc/infiles_state
resp_indir=/lustre/work/wfaletti/wofs/sens_calc/infiles_resp
state_resp_dir=/lustre/work/wfaletti/wofs/sens_calc/state_resp
sens_outdir=/lustre/work/wfaletti/wofs/sens_calc/sens_out

export statevar='T2,TD2,U10,V10,MLCAPE,MLCIN,SRH1,SRH3,SHEAR_U1,SHEAR_V1,SHEAR_U6,SHEAR_V6,T850,T700' # string of variables to split in python script (comma-delimited, no spaces)
export respvar='UH_HRSWT25_WRF,CREF_HRSWT,REFD_MAX' # string of variables to split in python script (comma-delimited, no spaces)

zh=0
rf_center_lon=-100.3    #gridpoint of center of response function
rf_center_lat=40.45
rf_xl=-0.5             #extent of response box, in deg (x-left,y-right,etc.)
rf_xr=0.5
rf_yb=-0.65
rf_yt=0.65

export methodlist='max'    #string of methods desired ; type of response: choices (avg, max, min, sum, gridsum)
export threshold='40'        #string of threshold values ; threshold value for gridsum
export condition='>='      #string of condition types ; condition for gridsum threshold (>, <, =, >=, <=, !=)


conda run python esa_state_wofs.py ${state_indir} ${state_resp_dir} ${statevar} 

conda run python esa_resp_wofs.py ${resp_indir} ${state_resp_dir} ${respvar} ${zh} ${rf_center_lon} ${rf_center_lat} ${rf_xl} ${rf_xr} ${rf_yb} ${rf_yt} ${methodlist} ${threshold} ${condition}

conda run python esa_sens_wofs.py ${state_resp_dir} ${sens_outdir}

conda deactivate
