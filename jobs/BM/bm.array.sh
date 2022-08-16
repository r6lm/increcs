#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -cwd                 
#$ -l h_vmem=2G
 
bash s$SGE_TASK_ID/s$SGE_TASK_ID.sh 
