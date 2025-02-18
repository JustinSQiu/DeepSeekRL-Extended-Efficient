#!/bin/bash
#SBATCH --job-name=r1_train_freeze
#SBATCH --output=slurm_output_freeze_first_layer.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --constraint=48GBgpu

source deepseek/bin/activate
module load cuda/11.7

nvidia-smi
nvcc --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py --output_dir "final1" --verbose --freeze_first_layer 1
# python plotter.py --log_dir "final1"