#!/bin/bash

# source /home/virtual_envs/ml/bin/activate

# nvidia-smi

echo "You would require an entire dataset to train using this script."

# python main.py -latent=512 -experiment_run='0112_topkk=9_fa=0.001_random=4' \
#                         -base="vgg11" \
#                         -last_layer_weight=-1 \
#                         -fa_coeff=0.001 \
#                         -topk_k=9 \
#                         -train_dir="/usr/xtmp/mammo/Lo1136i_with_fa/train_augmented_5000/" \
#                         -push_dir="/usr/xtmp/mammo/Lo1136i_finer/by_margin/train/" \
#                         -test_dir="/usr/xtmp/mammo/Lo1136i_with_fa/validation/" \
#                         -random_seed=4 \
#                         -finer_dir="/usr/xtmp/mammo/Lo1136i_finer/by_margin/train_augmented_250/" \
#                         #-model="/usr/xtmp/mammo/saved_models/vgg11/0112_topkk=9_fa=0.001_random=4/6nopush0.8559.pth" \

python main.py -latent=512 -experiment_run='0112_topkk=9_fa=0.001_random=4' \
                        -base="vgg19" \
                        -last_layer_weight=-1 \
                        -fa_coeff=0.001 \
                        -topk_k=9 \
                        -train_dir="C:\Users\dilsh\OneDrive\Documents\Dilshaan\Third_Year\BEng_Individual_Project\code\BEng-Final-Year-Project\data\train" \
                        -push_dir="C:\Users\dilsh\OneDrive\Documents\Dilshaan\Third_Year\BEng_Individual_Project\code\BEng-Final-Year-Project\data\protoypes" \
                        -test_dir="C:\Users\dilsh\OneDrive\Documents\Dilshaan\Third_Year\BEng_Individual_Project\code\BEng-Final-Year-Project\data\validation" \
                        -random_seed=4 \
                        #-finer_dir="/usr/xtmp/mammo/Lo1136i_finer/by_margin/train_augmented_250/" \  <--- not needed for dataset
                        #-model="/usr/xtmp/mammo/saved_models/vgg11/0112_topkk=9_fa=0.001_random=4/6nopush0.8559.pth" \