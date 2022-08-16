#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N BM
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
 
bash s$SGE_TASK_ID/s$SGE_TASK_ID.sh 
