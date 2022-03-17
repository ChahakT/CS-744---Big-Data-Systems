import os
import time
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
import argparse
import model as mdl
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

device = "cpu"

RANK = None
NUM_WORKERS = None

COMM_PORT = 1024
batch_size = 256 # batch for one node


def dist_setup(master_ip):
    dist.init_process_group(backend="gloo",
                            init_method=f"tcp://{master_ip}:{COMM_PORT}",
                            rank=RANK,
                            world_size=NUM_WORKERS)

def set_seeds(seed_val):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    # remember to exit the train loop at end of the epoch
    total_loss = 0.0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()

        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.reduce_op.SUM, async_op=False)
            p.grad /= NUM_WORKERS

        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 20 == 19:
            print('Batch: {} [{}/{} ({:.0f}%)] Average Loss(20): {:.4f}'.format(
                batch_idx,
                batch_idx * len(data), len(train_loader.dataset),
                (batch_idx / len(train_loader)) * 100,
                (total_loss / 20)))

            total_loss = 0.0


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


def main():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
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

    distributed_sampler = DistributedSampler(training_set)
    train_loader = torch.utils.data.DataLoader(training_set,
                                               num_workers=2,
                                               batch_size=batch_size,
                                               sampler=distributed_sampler,
                                               pin_memory=True)

    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", required=True, dest="master_ip", help='IP for master node')
    parser.add_argument("--num-nodes", required=True, dest="num_nodes", type=int, help='Number of nodes')
    parser.add_argument("--rank", required=True, dest="rank", type=int, help='Rank of nodes')
    args = parser.parse_args()

    RANK = args.rank
    NUM_WORKERS = args.num_nodes
    batch_size //= NUM_WORKERS

    set_seeds(140)
    dist_setup(args.master_ip)
    main()
