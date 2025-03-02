#!/bin/bash
#SBATCH --job-name=r1_train
#SBATCH --output=slurm_output_inversion.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --constraint=48GBgpu

source /nlp/data/jsq/deepseek/bin/activate
which python

module load cuda/11.7

nvidia-smi
nvcc --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py --dataset_name "matrix_inversion" --evaluator "matrix_inversion" --output_dir "output_matrix_inv" --num_train_iters 50 --verbose

# python plotter.py --log_dir "final1"

# replace: job name, slurm output, freeze tag, output dir
# /home1/j/jsq/dev/DeepSeekRL-Extended-Efficient