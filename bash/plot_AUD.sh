#!/bin/bash
#SBATCH --job-name=plot20230427
#SBATCH --time=02:30:00
#SBACTH --ntasks=1
#SBATCH --output="out/plot20230427-%j.out"
#SBATCH --mem=5G
#SBATCH -p evlab

source /mindhive/mcdermott/u/gretatu/anaconda/etc/profile.d/conda.sh
conda activate auditory_brain_dnn

cd /mindhive/mcdermott/u/gretatu/auditory_brain_dnn/aud_dnn/analyze/
python AUD_plot_main.py