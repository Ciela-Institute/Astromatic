#!/bin/bash
#SBATCH --account=def-lplevass
#SBATCH --array=0-10
#SBATCH --mem-per-cpu=32G
#SBATCH --time=24:00:00

source $HOME/environments/astromatic/bin/activate

python $ASTROMATIC_PATH/Problems/P2_lens_inference/lensing_pipeline.py\
  --dataset_name=gravitational_lenses_20220608\
  --data_type=lens\
  --size=50000\
  --rpf=5000\
  --npix=128\
  --zl=0.5\
  --zs=1.5
