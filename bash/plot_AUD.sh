#!/bin/bash
#SBATCH --job-name=plot20230503
#SBATCH --time=03:00:00
#SBACTH --ntasks=1
#SBATCH --output="out/plot20230503-%j.out"
#SBATCH --mem=6G
#SBATCH -p evlab

source /mindhive/mcdermott/u/gretatu/anaconda/etc/profile.d/conda.sh
conda activate auditory_brain_dnn

cd /mindhive/mcdermott/u/gretatu/auditory_brain_dnn/aud_dnn/analyze/
python AUD_plot_main.py