#!/bin/bash

source /home/virtual_envs/ml/bin/activate

#nvidia-smi

echo "You would require an entire dataset to train using this script."

python accuracy.py -test_dir="/usr/xtmp/mammo/Lo1136i_with_fa/validation/" \
                   -model_dir="/usr/xtmp/mammo/saved_models/vgg11/0112_topkk=9_fa=0.001_random=4/" \
	           -model_name="quantized_stat_alt_fused5nopush01.0000.pth" \
