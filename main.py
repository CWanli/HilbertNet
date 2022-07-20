'''Train HilbertNet with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import os
import argparse

from models import hilbertnet
from utils import progress_bar
from logger import Logger, savefig
from dataset.ModelNetDataLoader import TrainModelNet, TestModelNet

import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset  # For custom datasets
import random

parser = argparse.ArgumentParser(description='PyTorch HilbertNet Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--training_epoch', default=200, type=int, help='training epoch')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda'

# Data
print('==> Preparing data..')
trainset = TrainModelNet('./dataset/train_modelnet.csv', augmentation=True)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
print('%s training samples' % len(trainset))

testset = TestModelNet('./dataset/test_modelnet.csv')
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
print('%s testing samples' % len(testset))

# Model
print('==> Building model..')
d_64 = np.load('./hilbert_curve/hilbert_distance_64x64.npy')
c_32 = np.load('./hilbert_curve/hilbert_coordinate_32x32.npy')
d_32 = np.load('./hilbert_curve/hilbert_distance_32x32.npy')
c_16 = np.load('./hilbert_curve/hilbert_coordinate_16x16.npy')
d_16 = np.load('./hilbert_curve/hilbert_distance_16x16.npy')
c_8 = np.load('./hilbert_curve/hilbert_coordinate_8x8.npy')
net = hilbertnet(40, d_64, c_32, d_32, c_16, d_16, c_8)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.checkpoint + '/best.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    logger = Logger(os.path.join('./log',args.checkpoint, 'log.txt'), title='', resume=True)
else:
    if not os.path.isdir(os.path.join('./log',args.checkpoint)):
        os.mkdir(os.path.join('./log',args.checkpoint))
    logger = Logger(os.path.join('./log',args.checkpoint, 'log.txt'), title='')
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc.'])

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(
                       net.parameters(),
                       lr=args.lr,
                       betas = (0.9, 0.999),
                       eps = 1e-08,
                       weight_decay = 1e-4,
                        )

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [51, 101, 151], 0.5)

# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_avg = 0
    acc_avg = 0
    for batch_idx, (images, points, targets) in enumerate(trainloader):
        images, points, targets = images.to(device), points.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs_1d, outputs_2d = net(images, points)
        loss = criterion(outputs_1d, targets) + criterion(outputs_2d, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = (outputs_1d + outputs_2d).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loss_avg = train_loss/(batch_idx+1)
        acc_avg = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (loss_avg, acc_avg, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving Checkpoint..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('./checkpoint/' + args.checkpoint):
        os.mkdir('./checkpoint/' + args.checkpoint )
    torch.save(state,'./checkpoint/' + args.checkpoint + '/checkpoint.pth')

    return (loss_avg, acc_avg)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_avg = 0
    acc_avg = 0
    with torch.no_grad():
        for batch_idx, (images, points, targets) in enumerate(testloader):
            images, points, targets = images.to(device), points.to(device), targets.to(device)

            outputs_1d, outputs_2d = net(images, points)
            loss = criterion(outputs_1d, targets) + criterion(outputs_2d, targets)

            test_loss += loss.item()
            _, predicted = (outputs_1d + outputs_2d).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss_avg = test_loss/(batch_idx+1)
            acc_avg = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss_avg, acc_avg, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving Best..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./checkpoint/' + args.checkpoint):
            os.mkdir('./checkpoint/' + args.checkpoint )
        torch.save(state,'./checkpoint/' + args.checkpoint + '/best.pth')
        best_acc = acc

    print('Best acc: %.2f' % best_acc)
    return (loss_avg, acc_avg, best_acc)


if __name__ == '__main__':

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    for epoch in range(start_epoch, start_epoch+args.training_epoch):
        scheduler.step()
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.training_epoch, scheduler.get_lr()[0]))
        train_loss, train_acc = train(epoch)
        test_loss, test_acc, best_acc = test(epoch)
        # append logger file
        logger.append([scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc, best_acc])
        savefig(os.path.join('./log', args.checkpoint, 'log.pdf'))
        logger.plot()
    logger.close()




