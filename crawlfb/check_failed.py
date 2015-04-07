#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


def load_entities(rdf_file):
    mids = set()
    with open(rdf_file, 'r') as f:
        for line in f:
            if line.strip():
                arr = line.strip().split('\t')
                subj = arr[0].strip()
                obj = arr[2].strip()
                mids.add(subj)
                mids.add(obj)
    return mids


def get_dumped_entities(item_file):
    with open(item_file, 'r') as f:
        items = json.load(f)
    ret = [i['mid'] for i in items]
    return ret


train_entities = load_entities('/Users/cc/data/FB15k/freebase_mtr100_mte100-train.txt')
dev_entities = load_entities('/Users/cc/data/FB15k/freebase_mtr100_mte100-valid.txt')
test_entities = load_entities('/Users/cc/data/FB15k/freebase_mtr100_mte100-test.txt')
mids = train_entities.union(dev_entities).union(test_entities)

dumped_mids = get_dumped_entities('items.json')
failed_mids = mids.difference(dumped_mids)
print 'failed mids:', len(failed_mids)
print list(failed_mids)

