source /home/virtual_envs/ml/bin/activate

#nvidia-smi

echo "You would require an entire dataset to train using this script."

python qat.py -train_dir="/usr/xtmp/mammo/Lo1136i_with_fa/train_augmented_5000/" \
                   -model_dir="/usr/xtmp/mammo/saved_models/vgg11/0112_topkk=9_fa=0.001_random=4/" \
                   -model_name="5nopush0.9504.pth" \
