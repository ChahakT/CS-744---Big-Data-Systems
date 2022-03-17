# Author : Chahak Tharani (netid: ctharani)
# Date : 12th October 2021

import sys
import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import argparse
from time import time
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

device = "cpu"
torch.set_num_threads(8)

batch_size = 256 # batch for one node

# Method to parse arguments
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type = int)
    parser.add_argument("--master-ip", dest = "master_ip", type = str)
    parser.add_argument("--num-nodes", dest = "num_nodes", type = int)
    return  parser.parse_args()

# Method to set process group for DDP
def dist_setup(args):
    os.environ['MASTER_ADDR'] = args.master_ip
    os.environ['MASTER_PORT'] = '23465'
    dist.init_process_group(backend="gloo", rank=args.rank, world_size=args.num_nodes)

# Method to destroy the process group
def cleanup():
    dist.destroy_process_group()

# Method to train model
def train_model(ddp_model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    ddp_model.train()
    train_loader.sampler.set_epoch(epoch)
    running_loss = 0.0
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # startTime = int(time() * 1000)
        optimizer.zero_grad()
        output = ddp_model(data)
        # fwdTime = int(time() * 1000)
        loss = criterion(output, target)
        loss.backward()
        # bwdTime = int(time() * 1000)
        optimizer.step()
        # optTime = int(time() * 1000)
        running_loss += loss.item()

	# Get time for different phases in training stage.
        #if batch_idx < 40:
        #    print('{}\t{}\t{}'.format(fwdTime-startTime,  bwdTime-fwdTime,  optTime-bwdTime))

	# Get Average loss and Average running time
        #if batch_idx ==0:
        #    startTime = int(time() * 1000)
        #    print('start time is ', startTime)
        #    running_loss = 0.0
        #if batch_idx < 40:
        #    print(loss.item())
        #if batch_idx == 39:
        #    endTime = int(time() * 1000)
        #    print('end time is ', endTime)
        #    print('average time is ', (endTime - startTime)/39)
        #    print('average loss is ', running_loss / 39)

	# Get running loss after every 20 iterations
        # if batch_idx % 20 == 19:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), running_loss / 20))
        #     running_loss = 0.0

    print('Finished Training')

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main(args):
    torch.manual_seed(140)
    np.random.seed(140)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    
    sampler = DistributedSampler(training_set,
                             num_replicas=args.num_nodes,
                             rank=args.rank,
                             shuffle=True)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=4,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    shuffle=False,
                                                    pin_memory=True)
    
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=4,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)

    # Wrap the model to a DDP module
    model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)
    cleanup()

if __name__ == "__main__":
    # Parse the arguments
    args = parse_args(sys.argv)
    # Setup a process group
    dist_setup(args)
    batch_size //= args.num_nodes
    main(args)
