#!/bin/bash
#SBATCH --job-name=r1_train
#SBATCH --output=slurm_output/slurm_output_rref.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --constraint=48GBgpu

source /nlp/data/jsq/deepseekv2/bin/activate

module load cuda/11.7

nvidia-smi
nvcc --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py --dataset_name "matrix_RREF" --evaluator "matrix_RREF" --output_dir "output_matrix_RREF" --num_train_iters 50 --eval_iterations 10 --verbose

# python plotter.py --log_dir "final1"

# replace: job name, slurm output, freeze tag, output dir
# /home1/j/jsq/dev/DeepSeekRL-Extended-Efficient