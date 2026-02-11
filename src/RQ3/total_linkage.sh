#!/bin/bash
#SBATCH -J total_linkage        # job name
#SBATCH -o /bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/Replication_Package/RQ3/total_linkage.o%j    # output and error file name (%j expands to jobID)
#SBATCH -n 1                # total number of tasks requested
#SBATCH -N 1                # number of nodes you want to run on
#SBATCH --cpus-per-task 16  # request cores (64 per node)
#SBATCH --gres=gpu:1        # request a gpu (4 per node)
#SBATCH -p gpu-l40          # queue (partition)
#SBATCH -t 168:00:00         # run time (hh:mm:ss)
#SBATCH --mail-type=end
#SBATCH --mail-user=ericmelin@u.boisestate.edu
. ~/.bashrc
mamba activate ORNL2
python /bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/Replication_Package/RQ3/total_linkage.py