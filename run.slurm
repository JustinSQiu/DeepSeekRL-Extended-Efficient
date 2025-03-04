#!/bin/bash
#SBATCH --job-name=r1_train_matrix_inverse
#SBATCH --output=slurm_output/slurm_output_matrix_inverse.txt
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

python main.py --dataset_name "matrix_inverse" --evaluator "matrix_inverse" --output_dir "matrix_inverse" --num_train_iters 100 --eval_iterations 20 --verbose --max_completion_length 1024 --gradient_accumulation_steps 1

# python plotter.py --log_dir "final1"

# replace: job name, slurm output, freeze tag, output dir
# /home1/j/jsq/dev/DeepSeekRL-Extended-Efficient