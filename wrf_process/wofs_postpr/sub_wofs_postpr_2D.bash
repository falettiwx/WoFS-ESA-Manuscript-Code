#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=wofs_pp
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition nocona
#SBATCH --account=default
#SBATCH --nodes=1 --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --array=1-1:1
#SBATCH --account=default

module load gcc python

export init_str=20190517_2100
export outstep=15   # minutes between output files
export indir=/lustre/scratch/wfaletti/wofs/wofs${init_str}/mem_num_*
export combinedir=/lustre/scratch/wfaletti/wofs/wofs_pp_${init_str}/
savedir=/lustre/work/wfaletti/wofs/wofs_save/wofs_${init_str}/

# Make directories to store output
mkdir ${combinedir}
mkdir ${savedir}

# Activate conda environment
source /home/wfaletti/miniconda3/etc/profile.d/conda.sh
conda activate wofs_post

conda run python wofs_calcvars_2d.py ${indir}

# Execute combination script
ln -sf /lustre/work/wfaletti/wofs/wofs_pp/wofs_combine_2d.py ${combinedir}
cd ${combinedir}
conda run python wofs_combine_2d.py ${init_str} ${outstep} ${combinedir}

# Move files to wofs_save directory
mv wofs_i* ${savedir}
