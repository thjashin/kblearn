#!/usr/env/bin python
# -*- coding: utf-8 -*-

import sys
import pyparsing
from pyparsing import *
import matplotlib.pyplot as plt


float_number = Regex(r'-?\d+(\.\d*)?([eE][\+-]\d+)?')
valid_train_msg = 'MEAN RANK >> valid:' + float_number + ', train:' + float_number
test_msg = '##### NEW BEST VALID >> test:' + float_number

test_train_fig, test_train_ax = plt.subplots()
test_train_ax.set_xlabel('Mean Rank (test)')
test_train_ax.set_ylabel('Mean Rank (train)')
test_train_lines = []
test_train_labels = []
train_test_fig, train_test_ax = plt.subplots()
train_test_ax.set_xlabel('Mean Rank (train)')
train_test_ax.set_ylabel('Mean Rank (test)')
train_test_lines = []
train_test_labels = []

for name in sys.argv[1:]: 
  train_ranks = []
  test_ranks = []

  with open(name, 'r') as f:
    recent_train = None
    for line in f:
      if line.strip():
        try:
          arr = valid_train_msg.parseString(line)
          recent_train = arr[-1]
        except:
          pass
        try:
          arr = test_msg.parseString(line)
          test_rank = arr[1]
          train_ranks.append(recent_train)
          test_ranks.append(test_rank)
        except:
          pass

  l1, = test_train_ax.plot(test_ranks, train_ranks, 'o-')
  l2, = train_test_ax.plot(train_ranks, test_ranks, 'o-')
  label = name.split('.')[0]
  test_train_lines.append(l1)
  test_train_labels.append(label)
  train_test_lines.append(l2)
  train_test_labels.append(label)

test_train_fig.legend(test_train_lines, test_train_labels, 'upper right')
train_test_fig.legend(train_test_lines, train_test_labels, 'upper right')

train_test_fig.savefig('train_test.png', dpi=300)
test_train_fig.savefig('test_train.png', dpi=300)

