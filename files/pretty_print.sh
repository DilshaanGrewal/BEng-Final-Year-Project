#!/bin/bash

source /home/virtual_envs/ml/bin/activate

#nvidia-smi

echo "Begun pretty printing."

python pretty_print.py -test_model_dir '/usr/xtmp/mammo/saved_models/vgg11/0112_topkk=9_fa=0.001_random=4/' \
                                -test_model_name 'quantized_fused5nopush01.0000.pth'

echo "End."
