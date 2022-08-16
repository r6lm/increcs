#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -N IU-seed75
#$ -l h_rt=24:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU: 
#$ -pe gpu-titanx 1
#
# Request system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=50G
# send emails
#$ -M ro6lm@outlook.com
#$ -m beas

# Initialise the environment modules and load CUDA version 8.0.61
. /etc/profile.d/modules.sh
module load anaconda

# try with visible devices
# source /exports/applications/support/set_cuda_visible_devices.sh

# Run the executable
source activate alpha
cd /exports/eddie/scratch/s2110626/diss/increcs/code/scripts

python IU.py --seed 5
