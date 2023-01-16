import torch
import torch.quantization.quantize_fx as quantize_fx
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from dataHelper import DatasetFolder
import re
import numpy as np
import os
import copy
from skimage.transform import resize
from helpers import makedir, find_high_activation_crop, silent_print
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
import argparse
import pandas as pd
import ast
import png
from torchinfo import summary
import pytorch_model_summary as pms
from collections import namedtuple

from settings import img_size, prototype_shape, num_classes, base_architecture, class_specific, \
                     prototype_activation_function, add_on_layers_type, prototype_activation_function_in_numpy
import torch.nn.utils.prune as prune
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-test_model_dir', nargs=1, type=str, default='0')
parser.add_argument('-test_model_name', nargs=1, type=str, default='0')
args = parser.parse_args()


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
    print("Hello")

print(summary(ppnet_multi, input_size=(1, 3, 224, 224)))
print(ppnet_multi)
print("Model's state_dict")
for param_tensor in ppnet_multi.state_dict():
    print(param_tensor, "\t", ppnet_multi.state_dict()[param_tensor])
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
def fuse_all_conv_bn(model):
    stack = []
    for name, module in model.named_children(): # immediate children
        if list(module.named_children()): # is not empty (not a leaf)
            fuse_all_conv_bn(module)

        if True:
            if True:
                #setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                setattr(model, name, torch.nn.Identity())
                print(name)
            else:
                stack.append((name, module))

print(ppnet_multi.module.features.features)
#print(ppnet_multi.named_children())
stack = []
stack.append(ppnet_multi)
while stack != []:
    m = stack.pop()
    print(list(m.named_children()))
    for name, module in m.named_children():
        stack.append(module)
        print(module)
        print(name)
        print(isinstance(module, torch.nn.Conv2d))
        print(isinstance(module, torch.nn.BatchNorm2d))
#fuse_all_conv_bn(ppnet_multi)
for m in ppnet_multi.modules():
    print(m)
print_size_of_model(ppnet_multi)
#for name, module in ppnet_multi.named_children():torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
#    print(name)
#    print(list(module.named_children()))
#moduls_to_fuse =  [['conv2', 'relu2']]
#net_fused = torch.quantization.fuse_modules(ppnet_multi.module, moduls_to_fuse)

#print(summary(net_fused, input_size=(1, 3, 224, 224)))
#print(net_fused)
#print("Model's state_dict")
#for param_tensor in net_fused.state_dict():
#        print(param_tensor, "\t", net_fused.state_dict()[param_tensor])

print("================")
for module in ppnet_multi.module.features.children():
    print(f"layer has {module}")
#prune.random_unstructured(ppnet_multi.module.features, name='features', amount = 0.3)

print(summary(ppnet_multi, input_size=(1, 3, 224, 224)))

