#!/bin/bash


module load nvidia/cuda/9.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/usr/lib64/nvidia/:/home/users/b/bochinski/cudnn/cuda/lib64
source ~/tf/bin/activate

$@

deactivate