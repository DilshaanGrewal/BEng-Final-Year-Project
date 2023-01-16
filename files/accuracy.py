import os
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

import argparse
import re
from dataHelper import DatasetFolder
from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
import random

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("-test_dir", type=str)
parser.add_argument("-model_dir", type=str)
parser.add_argument("-model_name", type=str)
args = parser.parse_args()
load_model_dir = args.model_dir + args.model_name
model_name = args.model_name
test_dir = args.test_dir
model_dir = args.model_dir
base_architecture = load_model_dir.split('/')[-3]


from torchinfo import summary
from settings import img_size, prototype_shape, num_classes, \
        prototype_activation_function, add_on_layers_type, prototype_activation_function_in_numpy
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size
from settings import class_specific
import torch.nn.utils.prune as prune


# test set
test_dataset =DatasetFolder(
    test_dir,
    loader=np.load,
    extensions=("npy",),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

if "quantized" in model_name and "qat" not in model_name:
    ppnet = model.construct_PPNet(base_architecture="vgg11",
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  topk_k=9,                                                                                                                                         num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,                                                                                      add_on_layers_type=add_on_layers_type,
                                  last_layer_weight=-1,                                                                                                                             class_specific=class_specific)

    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)
    torch.quantization.fuse_modules(ppnet_multi.module.features.features, [['0', '1'],['3', '4']], inplace=True)
    ppnet_multi.eval()

    ppnet_multi.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    #pnet_multi.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(ppnet_multi, inplace=True)
    torch.quantization.convert(ppnet_multi, inplace=True)
    ppnet_multi.load_state_dict(torch.load(load_model_dir))
elif "qat" in model_name:
    ppnet = model.construct_PPNet(base_architecture="vgg11",
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  topk_k=9,                                                                                                                                         num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,                                                                                      add_on_layers_type=add_on_layers_type,
                                  last_layer_weight=-1,                                                                                                                             class_specific=class_specific)

    ppnet = ppnet.to(device)
    ppnet = ppnet.to(memory_format=torch.channels_last)
    ppnet_multi = torch.nn.DataParallel(ppnet)
    #torch.quantization.fuse_modules(ppnet_multi.module.features.features, [['0', '1'],['3', '4']], inplace=True)
    ppnet_multi.eval()

    optimizer = torch.optim.SGD(ppnet_multi.parameters(), lr = 0.0001)
    ppnet_multi.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    ppnet_multi.train()
    torch.quantization.prepare_qat(ppnet_multi, inplace=True)
    torch.quantization.convert(ppnet_multi, inplace=True)
    ppnet_multi.load_state_dict(torch.load(load_model_dir))
    print(ppnet_multi.module.features.features)
else:
    ppnet = torch.load(load_model_dir)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)
    #
    torch.quantization.quantize_dynamic(ppnet_multi, dtype=torch.qint8, inplace=True)
    #torch.quantization.fuse_modules(ppnet_multi.module.features.features, [['0', '1'],['3', '4']], inplace=True)
print_size_of_model(ppnet_multi)
times = []
with torch.no_grad():
    for i in range(7):
        auc, accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific)
        times.append(accu)

print(times)
print(summary(ppnet_multi, input_size=(1, 3, 224, 224)))
