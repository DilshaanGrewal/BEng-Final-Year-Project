#!/bin/bash

source /home/virtual_envs/ml/bin/activate

#nvidia-smi

echo "Begun generating explanations."

python local_analysis.py -test_img_name '0929_arr.npy' \
                                -test_img_dir '/usr/xtmp/mammo/Lo1136i_with_fa/validation/Cabbage/' \
                                -test_img_label 1 \
                                -test_model_dir '/usr/xtmp/mammo/saved_models/vgg11/0112_topkk=9_fa=0.001_random=4' \
                                -test_model_name 'quantized_fused5nopush01.0000.pth'

echo "End."
