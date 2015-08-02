#!/bin/bash

THEANO_FLAGS=device=gpu0 python FB15k_TransE.py 1>log/transe.log 2>log/transe.log & 

