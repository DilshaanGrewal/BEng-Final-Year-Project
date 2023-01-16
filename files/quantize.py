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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for i, (image, label, patient_id) in enumerate(data_loader):
            print(str(i) + "/" + str(len(data_loader)))
            target = label.to(device)
            image = image.to(device)
            output, min_distances, upsampled_activation = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1,1))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("-train_dir", type=str)
parser.add_argument("-model_dir", type=str)
parser.add_argument("-model_name", type=str)
args = parser.parse_args()
load_model_dir = args.model_dir + args.model_name
train_dir = args.train_dir
model_dir = args.model_dir
name_split = args.model_name.split('.',1)
model_name = name_split[0]

from settings import img_size, prototype_shape, num_classes, \
        prototype_activation_function, add_on_layers_type, prototype_activation_function_in_numpy
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size
from settings import class_specific

train_dataset = DatasetFolder(
    train_dir,
    augmentation=False,
    loader=np.load,
    extensions=("npy",),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)

ppnet = torch.load(load_model_dir)
ppnet = ppnet.to(device)
ppnet_multi = torch.nn.DataParallel(ppnet)
ppnet_multi.eval()

torch.quantization.fuse_modules(ppnet_multi.module.features.features, [['0', '1'],['3', '4']], inplace=True)
#print(ppnet_multi.module.features.features)
num_calibration_batches = 32
criterion = torch.nn.CrossEntropyLoss()

#pnet_multi.qconfig = torch.quantization.default_qconfig
ppnet_multi.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(ppnet_multi, inplace=True)
evaluate(ppnet_multi, criterion, train_loader, neval_batches=num_calibration_batches)
torch.quantization.convert(ppnet_multi, inplace=True)

save.save_model_w_condition(model=ppnet_multi, model_dir=model_dir, model_name="quantized_stat_alt_fused" + model_name, accu=1,
                                        target_accu=0.00, state_dict=True)

print("Model saved to " + str(model_dir) + ".")
