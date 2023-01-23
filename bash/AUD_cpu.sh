#!/bin/bash
#SBATCH --job-name=run_20221213
#SBATCH --time=18:20:00
#SBACTH --ntasks=1
#SBATCH --output="run_20221213-%j.out"
#SBATCH --mem=12G
#SBATCH -p evlab

source /om2/user/gretatu/anaconda/etc/profile.d/conda.sh
conda activate control-neural

cd /om/user/`whoami`/aud-dnn/aud_dnn
python AUD_main.py --source_model "${1}" --source_layer "${2}" --randnetw "${3}"