#!/bin/bash
#
# Submission script for the finetune_script program
#
# Comments starting with #OAR are used by the resource manager if using "oarsub -S"

# Note : quoting style of parameters matters, follow the example
#OAR -l /nodes=1/gpunum=1, walltime=00:30:00
#OAR -p gpu='YES' and gpumem>45000 and (host='nefgpu47.inria.fr' or host='nefgpu48.inria.fr' or host='nefgpu49.inria.fr' or host='nefgpu50.inria.fr' or host='nefgpu51.inria.fr')

#OAR -t besteffort

# The job is submitted to the default queue
#OAR -q default

nvidia-smi 
printenv

# get the number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

module load gcc-nvptx/9.2.0

source activate llama2
# Path to the binary to run
torchrun --nproc_per_node=$NUM_GPUS --master_port=9801 inference_perplexity.py 
#deepspeed --num_gpus=$NUM_GPUS inference.py --deepspeed deepspeed_config.json
conda deactivate