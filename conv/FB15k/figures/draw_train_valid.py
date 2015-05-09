#!/usr/env/bin python
# -*- coding: utf-8 -*-

import sys
import pyparsing
from pyparsing import *
import matplotlib.pyplot as plt


float_number = Regex(r'-?\d+(\.\d*)?([eE][\+-]\d+)?')
valid_train_msg = 'MEAN RANK >> valid:' + float_number + ', train:' + float_number

valid_train_fig, valid_train_ax = plt.subplots()
valid_train_ax.set_xlabel('Mean Rank (validation)')
valid_train_ax.set_ylabel('Mean Rank (train)')
valid_train_lines = []
valid_train_labels = []
train_valid_fig, train_valid_ax = plt.subplots()
train_valid_ax.set_xlabel('Mean Rank (train)')
train_valid_ax.set_ylabel('Mean Rank (validation)')
train_valid_lines = []
train_valid_labels = []
for name in sys.argv[1:]:
  train_ranks = []
  valid_ranks = []
  with open(name, 'r') as f:
    for line in f:
      if line.strip():
        arr = valid_train_msg.parseString(line)
        valid_rank = arr[1]
        train_rank = arr[-1]
        valid_ranks.append(valid_rank)
        train_ranks.append(train_rank)
  l1, = valid_train_ax.plot(valid_ranks, train_ranks, 'o-')
  l2, = train_valid_ax.plot(train_ranks, valid_ranks, 'o-')
  label = name.split('.')[0]
  valid_train_lines.append(l1)
  valid_train_labels.append(label)
  train_valid_lines.append(l2)
  train_valid_labels.append(label)
valid_train_fig.legend(valid_train_lines, valid_train_labels, 'upper right')
train_valid_fig.legend(train_valid_lines, train_valid_labels, 'upper right')

train_valid_fig.savefig('train_valid.png', dpi=300)
valid_train_fig.savefig('valid_train.png', dpi=300)

