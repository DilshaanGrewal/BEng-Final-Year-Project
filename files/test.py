import torch
import model
load_model_dir = "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.001_random=4/129nopush0.2381.pth"
ppnet = torch.load(load_model_dir)
