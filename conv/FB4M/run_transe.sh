#!/bin/bash

THEANO_FLAGS=device=gpu0 python FB4M_TransE.py 1>log/transe.log 2>log/transe.log &

