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
from torch import distributed as dist
import sys
import argparse
import datetime

device = "cpu"
torch.set_num_threads(8)

batch_size = 256 # batch for one node

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type = int)
    parser.add_argument("--master-ip", dest="master_ip", type = str)
    parser.add_argument("--num-nodes", dest="num_nodes", type = int)
    return  parser.parse_args()

def dist_setup(rank, world_size, master_ip):
    dist.init_process_group(backend="gloo", init_method=master_ip+":23456", rank=rank, world_size=world_size)

def train_model(model, train_loader, optimizer, criterion, epoch, rank):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    running_loss = 0.0
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
        if batch_idx == 1:
            running_loss = 0.0
            first_itr_time = datetime.datetime.now()
        if batch_idx == 40:
            last_itr_time = datetime.datetime.now()
            final_running_loss = running_loss
        data, target = data.to(device), target.to(device)
        #timestamp before start of forward pass
        fwd_start = datetime.datetime.now()
        optimizer.zero_grad()
        output = model(data)
        #timestamp after end of forward pass and before start of backward pass
        fwd_end = datetime.datetime.now()
        loss = criterion(output, target)
        loss.backward()
        #timestamp after end of backward pass
        bkwd_end = datetime.datetime.now()
        for p in model.parameters():
            if(rank == 0):
                #initialize gather_list and scatter_list to tensors of the same size as grad, with zeros
                gatherList = [torch.zeros_like(p.grad)] * dist.get_world_size()
                scatterList = [torch.zeros_like(p.grad)] * dist.get_world_size()
                gathered = torch.distributed.gather(p.grad, gather_list=gatherList, dst=0)
                mean = sum(gatherList[i] for i in range(dist.get_world_size()))
                torch.distributed.scatter(p.grad, scatter_list=[mean for _ in range(dist.get_world_size())], src=0)
            else:
                torch.distributed.gather(p.grad, gather_list=None, dst=0)
                torch.dist.scatter(p.grad, scatter_list=None, src=0)
        #timestamp before start of optimizer step
        opt_start = datetime.datetime.now()
        optimizer.step()
        opt_end = datetime.datetime.now()

        running_loss += loss.item()
        #if batch_idx % 20 == 19:
        if batch_idx < 40:
            #Uncomment the below lines to display metrics
            #print('{}\t{}\t{}'.format((fwd_end - fwd_start).total_seconds(), (bkwd_end - fwd_end).total_seconds(), (opt_end - opt_start).total_seconds()))
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.item()))
            print('{}'.format(loss.item()))
#        running_loss = 0.0

    print('Finished Training')
    total_time = (last_itr_time - first_itr_time).total_seconds()
    print('Running Time for 40 iterations excluding first: {} seconds'.format(total_time))
    print('Average time per iteration over first 40 iterations excluding first: {} seconds'.format(total_time / 39))
    print('Average loss over 40 iterations excluding first:{}'.format(final_running_loss / 39))

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


def main(rank, world_size):
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
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set, num_replicas=world_size, rank=rank, seed=80)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=world_size,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=world_size,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)

    # running training for one epoch
    # set seed
    torch.manual_seed(80)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch, rank)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    args = parse_args(sys.argv)
    world_size = args.num_nodes
    master_ip = args.master_ip
    rank = args.rank
    batch_size //= world_size

    dist_setup(rank, world_size, master_ip)
    main(rank, world_size)