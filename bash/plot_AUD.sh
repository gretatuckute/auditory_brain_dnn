#!/bin/bash
#SBATCH --job-name=plot20220815
#SBATCH --time=01:00:00
#SBACTH --ntasks=1
#SBATCH --output="plot20220815-%j.out"
#SBATCH --mem=20G
#SBATCH -p evlab

source /om2/user/gretatu/anaconda/etc/profile.d/conda.sh
conda activate control-neural

cd /om/user/`whoami`/aud-dnn/aud_dnn/analyze/
python AUD_plot_main.py