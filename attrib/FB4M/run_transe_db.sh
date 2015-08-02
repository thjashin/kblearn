#!/bin/bash

THEANO_FLAGS=device=gpu1 python FB4M_TransE.py 1>log_db/transe.log 2>log_db/transe.log & 

