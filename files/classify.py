import os
import torch.nn.functional as nnf
import shutil
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import argparse
import re
from dataHelper import DatasetFolder
from helpers import makedir
import model
import push
from PIL import Image, ImageOps
from numpy import asarray
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
import random
from skimage.transform import resize
from settings import train_dir, test_dir, train_push_dir, img_size, prototype_shape,num_classes, prototype_activation_function,\
                     train_batch_size, test_batch_size, train_push_batch_size, class_names, add_on_layers_type, class_specific


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-test_model_dir', nargs=1, type=str, default='0')
parser.add_argument('-test_model_name', nargs=1, type=str, default='0')
parser.add_argument('-test_image', nargs=1, type=str, default='0')
args = parser.parse_args()

test_image = args.test_image[0]

load_model_dir = args.test_model_dir[0]
load_model_name = args.test_model_name[0]
load_model_path = os.path.join(load_model_dir, load_model_name)

if "quantized" in load_model_path:
    base_architecture = load_model_dir.split('/')[-3]
    ppnet = model.construct_PPNet(base_architecture="vgg11",
                                      pretrained=True, img_size=img_size, 
                                      prototype_shape=prototype_shape,
                                      topk_k=9,
                                      num_classes=num_classes,
                                      prototype_activation_function=prototype_activation_function,
                                      add_on_layers_type=add_on_layers_type,
                                      last_layer_weight=-1,
                                      class_specific=class_specific)
    ppnet = ppnet.to(device)        
    ppnet_multi = torch.nn.DataParallel(ppnet)
    ppnet_multi.eval()
    ppnet_multi.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(ppnet_multi, inplace=True)
    torch.quantization.convert(ppnet_multi, inplace=True)
    ppnet_multi.load_state_dict(torch.load(load_model_path))
else:
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)

img = Image.open(test_image)
img = ImageOps.grayscale(img)
narr = asarray(img)
#narr = np.load("/usr/xtmp/mammo/Lo1136i_with_fa/validation/Carrot/1024_arr.npy")
narr = np.stack([narr, narr, narr])
t = torch.from_numpy(narr).float() / 255
t = t.unsqueeze(0)

input = t.to(device)

grad_req = torch.no_grad()
with grad_req:
    output, min_distances, upsampled_activation = ppnet_multi(input)
    print(torch.nn.functional.softmax(output, dim=1))
    print(class_names[output.data.cpu().numpy().argmax()])

#test_dataset =DatasetFolder(
#    "/usr/xtmp/mammo/Lo1136i_with_fa/validation",
#    loader=np.load,
#    extensions=("npy",),
#    transform = transforms.Compose([
#        torch.from_numpy,
#    ]))
#test_loader = torch.utils.data.DataLoader(
#    test_dataset, batch_size=1, shuffle=False,
#    num_workers=4, pin_memory=False)
#
#for i, (image, label, patient_id) in enumerate(test_loader):
#    input = image.to(device)
#    grad_req = torch.no_grad()
#    with grad_req:
#        output, min_distances, upsampled_activation = ppnet_multi(input)
#        print(label)
#        print(output.data.cpu().numpy().argmax())
