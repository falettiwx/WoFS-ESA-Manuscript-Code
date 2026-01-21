#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=wrf_test
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition nocona
#SBATCH --account=default
#SBATCH --nodes=4 --ntasks=32
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --array=1-1:1
#SBATCH --account=default

# Set some variables for later
nproc=32  # should be the same as ntasks above
dir=/lustre/scratch/wfaletti/wrf_test  # empty directory for running WRF in, use $SCRATCH
savedir=/lustre/scratch/wfaletti/wrf_save
indata=${dir}/data  # location of inital and boundary condition data. This should be wrfinput and wrfbdy files.
image=/home/wfaletti/wrfv4-nocona.img  # set location and same of WRF container
wrf_nml=/lustre/scratch/wfaletti/wrf_test/namelist.input  # set location of WRF namelist
user=`whoami`
jid=${SLURM_JOB_ID}

# Create run and save directories if not already done
mkdir -p ${savedir}
mkdir -p ${dir}

# Load MPI
module load gcc openmpi

# $HOME is mounted by default, set environment variable to also bind $WORK and $SCRATCH
export WORK=/lustre/work/${user}
export SCRATCH=/lustre/scratch/${user}
export SINGULARITY_BIND="${WORK},${SCRATCH}"

# Launch the container and execute the supplied script
cd ${dir}
singularity exec ${image} ${dir}/run_wrf_wofs.bash jid=${jid} nproc=${nproc} dir=${dir} wrf_nml=${wrf_nml} indata=${indata}

# run wrf.exe
mpirun -n ${nproc} -machinefile ${HOME}/machinefile.${jid}_1 singularity exec ${image} ${dir}/run/wrf.exe
cat ${dir}/run/rsl.out.* > ${dir}/rslout_wrf.log
cat ${dir}/run/rsl.error.* > ${dir}/rslerror_wrf.log
#rm ${dir}/run/rsl.*

# Copy output to save directory
cp ${dir}/run/rsl*.log ${savedir}
cp ${dir}/run/wrfout* ${savedir}

# Clean WRF and WPS from run directory
#rm -rf ${dir}/WPS*
#rm -rf ${dir}/run
