#!/usr/bin/env python
#coding: utf-8

import os
import sys
import json
import copy
import gflags
import random
import collections
import numpy as np

gflags.DEFINE_string("dataset", "cifar10", "set dataset, support mnist, cifar10, cifar100,miniimage")

#noise_type: SYMMETRY, PAIR
gflags.DEFINE_string("noise_type", "PAIR", "set noise type, support three kinds types, SYMMETRY, PAIR  types")

#noise_rate: from 0.0 to 1.0
gflags.DEFINE_float("noise_rate", 0.45, "set noise rate, the range of valid value is from 0.0 to 1.0")

# class number
dataset2classnum = {"mnist": 10, "cifar10": 10, "cifar100": 100, "miniimage":100}

def creat_db_txt(data_dir, save_dir,flag):
    string = ""
    savefile = ""
    if flag == "train":
        string = "_data.txt"
        savefile = "/train.txt"
        n = 3
        m = 6
        k = -2
    else:
        string = "/validation_test.txt"
        savefile = "/test.txt"
        n = 1
        m = 5
        k = -3
    with open(data_dir) as file:
        f = open(save_dir,'a')
        for line in file.readlines():
            curLine = line.strip().split(" ")
            for i in xrange(10):
                del curLine[0]
            for i in xrange(n):
                del curLine[1]
            for i in xrange(m):
                del curLine[2]
            curLine[0] = curLine[0][2:-2]
            curLine[1] = curLine[1][2:k]
            curLine.insert(1, ' ')
            f.writelines(curLine)
            f.write('\n')
        f.close()
    file.close()


def corrupt():
    """
    Corrupt clean dataset by noise type and rate
    :return:
    """
    gflags.FLAGS(sys.argv)
    # dataset name
    dataset = gflags.FLAGS.dataset
    # noise type
    noise_type = gflags.FLAGS.noise_type
    # noise rate
    noise_rate = gflags.FLAGS.noise_rate
    # class number
    class_num = dataset2classnum[dataset]
    # data path
    data_path = "./data/%s/" % dataset
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    print("dataset: %s, noise_type: %s, noise_rate: %f" % (dataset, noise_type , noise_rate))
    if noise_type == "SYMMETRY":
        c_m = noise_rate / (class_num-1) * np.ones((class_num, class_num))
        for c in xrange(class_num):
            c_m[c][c] = 1 - noise_rate
    elif noise_type == "PAIR":
        c_m = np.zeros((class_num, class_num))
        for c in xrange(class_num):
            c_m[c][c] = 1 - noise_rate
            c_m[c][(c+1)%class_num] = noise_rate

    print(c_m)
    items = list()
    for line in open(data_path+"ori_data.txt"):
        item = json.loads(line)
        item["ori_id"] = copy.deepcopy(item["id"])
        items.append(item)
    random.shuffle(items)

    m = int(round(noise_rate * len(items)))
    flipper = np.random.RandomState(0)
    for item in items:
        i = int(item["id"][0])
        flipped = flipper.multinomial(1, c_m[i, :], 1)[0]
        _i = np.where(flipped == 1)[0][0]
        #print i, _i
        _id = str(_i)
        item["id"][0] = _id

    cnt = 0
    predix_dir = data_path + str(noise_type) + "_%02d" % int(100 * noise_rate)
    if not os.path.exists(predix_dir):
        os.mkdir(predix_dir)
    with open(predix_dir + "_data.txt", "wb") as f:
        for item in items:
            if item["id"][0] != item["ori_id"][0]:
                item["id"].append("1")
                cnt += 1
            else:
                item["id"].append("0")
            f.write(json.dumps(item) + "\n")
        f.close()
    print("noise ratio:", round(100.0 * cnt / len(items), 2))
    if not os.path.exists(predix_dir):
        os.mkdirs(predix_dir)
    creat_db_txt(predix_dir+"_data.txt",predix_dir+"/train.txt", "train")
    creat_db_txt(data_path+"validation_test.txt",predix_dir+"/test.txt", "test")
    
if __name__ == "__main__":
    corrupt()
