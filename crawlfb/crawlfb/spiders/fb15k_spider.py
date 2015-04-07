#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scrapy
from crawlfb.items import CrawlfbItem


def read_entities(data_file):
    mids = set()
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                arr = line.strip().split('\t')
                subj = arr[0].strip()
                obj = arr[2].strip()
                mids.add(subj)
                mids.add(obj)
    return mids


def gen_urls():
    train_mids = read_entities('/Users/cc/data/FB15k/freebase_mtr100_mte100-train.txt')
    dev_mids = read_entities('/Users/cc/data/FB15k/freebase_mtr100_mte100-valid.txt')
    test_mids = read_entities('/Users/cc/data/FB15k/freebase_mtr100_mte100-test.txt')
    print 'train_mids:', len(train_mids)
    print 'dev_mids:', len(dev_mids)
    print 'test_mids:', len(test_mids)
    mids = train_mids.union(dev_mids).union(test_mids)
    urls = ['http://www.freebase.com' + mid for mid in mids]
    return urls


class FB15kSpider(scrapy.Spider):
    name = 'fb15k'
    allowed_domains = ['www.freebase.com']
    start_urls = gen_urls()

    def parse(self, response):
        item = CrawlfbItem()
        item['name'] = response.xpath('//div[@class="page-title img"]/h1/text()').extract()[0].strip()
        item['mid'] = response.xpath('//div[@class="page-title img"]/div[@class="meta"]/span[1]/span[@class="mid"]/text()').extract()[0].strip()
        item['description'] = response.xpath('//div[@data-id="/common/topic/description"]/div[2]/ul/li/span/span/text()').extract()[0].strip()
        return item


if __name__ == "__main__":
    urls = gen_urls()
    print urls[0]
    print 'urls:', len(urls)

