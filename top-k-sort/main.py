#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import getopt
import sys
import os
import math
import time
import argparse

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_

from dnc.dnc import DNC

import dataset

parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('-bits', type=int, default=7, help='size of bit representation of numbers')
parser.add_argument('-rnn_type', type=str, default='lstm', help='type of recurrent cells to use for the controller')
parser.add_argument('-nhid', type=int, default=256, help='number of hidden units of the inner nn')
parser.add_argument('-dropout', type=float, default=0, help='controller dropout')
parser.add_argument('-memory_type', type=str, default='dnc', help='dense or sparse memory: dnc | sdnc | sam')

parser.add_argument('-nlayer', type=int, default=2, help='number of layers')
parser.add_argument('-nhlayer', type=int, default=4, help='number of hidden layers')
parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')

parser.add_argument('-batch_size', type=int, default=50, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=20, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=64, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=4, help='number of read heads')

parser.add_argument('-cuda', type=int, default=0, help='Cuda GPU ID, -1 for CPU')
parser.add_argument('-debug', action='store_true', help='plot memory content')

parser.add_argument('-iterations', type=int, default=100000, metavar='N', help='total number of iteration')
parser.add_argument('-test_iterations', type=int, default=10, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=10, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=10, metavar='N', help='check point frequency')

args = parser.parse_args()
print(args)

if args.cuda != -1:
    print('Using CUDA.')
    T.manual_seed(1111)
else:
    print('Using CPU.')


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def criterion(predictions, targets):
    return T.mean(
            -1 * F.logsigmoid(predictions) * (targets) - T.log(1 - F.sigmoid(predictions) + 1e-9) * (1 - targets)
    )

def image_show(img, title=None):
    import matplotlib.pyplot as plt
    plt.imshow(img.cpu().detach().numpy())
    if title is not None: plt.title(title)
    plt.show()

def show_example(input_data, target_data, output_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.set_title("Input Data")
    ax2.set_title("Target Data")
    ax3.set_title("Output Data")
    ax1.imshow(input_data[0].cpu().detach().numpy().T)
    ax2.imshow(target_data[0].cpu().detach().numpy().T)
    ax3.imshow(output_data[0].cpu().detach().numpy().T)

    plt.show()

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    if not os.path.isdir(ckpts_dir):
        os.mkdir(ckpts_dir)

    batch_size = args.batch_size
    summarize_freq = args.summarize_freq
    check_freq = args.check_freq

    mem_slot = args.mem_slot
    mem_size = args.mem_size
    read_heads = args.read_heads

    rnn = DNC(
        input_size=args.bits+2,
        hidden_size=args.nhid,
        rnn_type=args.rnn_type,
        num_layers=args.nlayer,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=mem_slot,
        cell_size=mem_size,
        read_heads=read_heads,
        gpu_id=args.cuda,
        debug=args.debug,
        batch_first=True,
        independent_linears=True
    )

    print(rnn)
    if args.cuda != -1:
        rnn = rnn.cuda(args.cuda)

    last_save_losses = []
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98])

    check_ptr = os.path.join(ckpts_dir, 'best.pth')
    if os.path.isfile(check_ptr):
        curr_state = T.load(check_ptr)
        epoch = curr_state["epoch"] + 1
        rnn.load_state_dict(curr_state["rnn_state"])
        optimizer.load_state_dict(curr_state["opti_state"])
        print("Model loaded.")
    else:
        epoch = 1

    (chx, mhx, rv) = (None, None, None)

    for epoch in range(epoch,args.iterations + 1):
        llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=args.iterations))
        optimizer.zero_grad()

        input_data, target_output = dataset.generate_data(batch_size, args.bits, args.cuda)

        if rnn.debug:
            output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
        else:
            output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

        # show_example(input_data, target_output, F.sigmoid(output))
            
        loss = criterion(output, target_output)

        loss.backward()

        optimizer.step()
        loss_value = loss.item()

        summarize = (epoch % summarize_freq == 0)
        take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)

        mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }

        last_save_losses.append(loss_value)

        if summarize:
            loss = np.mean(last_save_losses)
            llprint("\n\tAvg. Logistic Loss: %.4f\n" % (loss))
            if np.isnan(loss):
                raise Exception('nan Loss')

        if summarize and rnn.debug:
            loss = np.mean(last_save_losses)
            last_save_losses = []

        if take_checkpoint:
            llprint("\nSaving Checkpoint ... "),
            curr_state = {
                "epoch":epoch,
                "rnn_state":rnn.state_dict(),
                "opti_state":optimizer.state_dict()
            }
            # check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
            # T.save(curr_state, check_ptr)
            check_ptr = os.path.join(ckpts_dir, 'best.pth')
            T.save(curr_state, check_ptr)
            llprint("Done!\n")

    for i in range(args.test_iterations):
        llprint("\rIteration %d/%d" % (i, args.test_iterations))
        input_data, target_output = dataset.generate_data(1, args.bits, args.cuda)

        if args.debug:
            output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
        else:
            output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
        target_output = target_output(F.sigmoid(output))

        # show_example(input_data, target_output, F.sigmoid(output))

