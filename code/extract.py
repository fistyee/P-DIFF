#!/usr/bin/env python
#coding: utf-8
import sys
import cv2
import json
import gflags
import cPickle
import numpy as np
import os
import struct
import png
from array import array
def unpickle(file):
    print("extracting: %s" % file)
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def cifar10_extract(dataset):
    
    #items = unpickle("batches.meta")
    #print items
    #sys.exit(0)
    image_dir = dataset+"/images/"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    root_dir = dataset+"/cifar-10-batches-py/"
    h, w, c = 32, 32, 3
    f = open(dataset+"/ori_data.txt", 'a')
    for i in xrange(1, 6):
        items = unpickle(root_dir+"data_batch_%d" % i)
        prefix = "batch_%d/" % i
        if not os.path.exists(image_dir+prefix):
            os.makedirs(image_dir+prefix)
        for j in xrange(10000):
            image_file = image_dir + prefix + items["filenames"][j]
            #print image_file
            img = items["data"][j]
            #print img.shape
            img = img.reshape((c, h, w))
            #print img.shape
            img = np.transpose(img, (1,2,0))
            #print img.shape
            img = img[:,:,(2,1,0)]
            #print img.shape
            cv2.imwrite(image_file, img)
            label = items["labels"][j]
            item = dict()
            item["image_file"] = image_file
            item["id"] = [str(label)]
            item["size"] = {"width": w, "height": h}
            item["box"] = {"x": 0, "y": 0, "w": w, "h": h}
            f.writelines(json.dumps(item))
            f.write('\n')
            #print json.dumps(item)
         #   break##
        #break##
    f.close()
    items = unpickle(root_dir+"test_batch")
    prefix = "batch_6/"
    if not os.path.exists(image_dir+prefix):
        os.makedirs(image_dir+prefix)
    f = open(dataset+"/validation_test.txt", 'a')
    for j in xrange(10000):
        image_file = image_dir + prefix + items["filenames"][j]
        #print image_file
        img = items["data"][j]
        #print img.shape
        img = img.reshape((c, h, w))
        #print img.shape
        img = np.transpose(img, (1,2,0))
        #print img.shape
        img = img[:,:,(2,1,0)]
        #print img.shape
        cv2.imwrite(image_file, img)
        label = items["labels"][j]
        item = dict()
        item["image_file"] = image_file
        item["id"] = [str(label)]
        item["size"] = {"width": w, "height": h}
        item["box"] = {"x": 0, "y": 0, "w": w, "h": h}
        f.writelines(json.dumps(item))
        f.write('\n')

def write_txt(dataset, image_dir, items, types):
    n = 0
    string = ""
    if types == "train":
        n = 50000
        string="ori_data"
    else:
        n = 10000
        string="validation_test"
    h, w, c = 32, 32, 3
    f = open(dataset+"/"+string+".txt", 'a')
    for j in xrange(n):
        image_file = image_dir + items["filenames"][j]
        img = items["data"][j]
        img = img.reshape((c, h, w))
        img = np.transpose(img, (1,2,0))
        img = img[:,:,(2,1,0)]
        cv2.imwrite(image_file, img)
        label = items["fine_labels"][j]
        item = dict()
        item["image_file"] = image_file
        #item["type"] = "train"
        item["id"] = [str(label)]
        item["size"] = {"width": w, "height": h}
        item["box"] = {"x": 0, "y": 0, "w": w, "h": h}
        f.writelines(json.dumps(item))
        f.write('\n')
        #break
    f.close()
        #print json.dumps(item)

def cifar100_extract(dataset):
    image_dir = dataset+"/images/train/"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    root_dir = dataset+"/cifar-100-python/"
    #items = unpickle(root_dir+"train")
    #write_txt(dataset, image_dir, items, "train")
    items_test = unpickle(root_dir+"test")
    write_txt(dataset, image_dir, items_test, "validation_test")

def mnist_extract(dataset):
    trainimg = dataset+'/raw/train-images-idx3-ubyte'
    trainlabel = dataset+'/raw/train-labels-idx1-ubyte'
    testimg = dataset+'/raw/t10k-images-idx3-ubyte'
    testlabel = dataset+'/raw/t10k-labels-idx1-ubyte'
    trainfolder = dataset+'/images/train'
    testfolder = dataset+'/images/test'
    if not os.path.exists(trainfolder): os.makedirs(trainfolder)
    if not os.path.exists(testfolder): os.makedirs(testfolder)

    trimg = open(trainimg, 'rb')
    teimg = open(testimg, 'rb')
    trlab = open(trainlabel, 'rb')
    telab = open(testlabel, 'rb')
    struct.unpack(">IIII", trimg.read(16))
    struct.unpack(">IIII", teimg.read(16))
    struct.unpack(">II", trlab.read(8))
    struct.unpack(">II", telab.read(8))

    trimage = array("B", trimg.read())
    teimage = array("B", teimg.read())
    trlabel = array("b", trlab.read())
    telabel = array("b", telab.read())

    trimg.close()
    teimg.close()
    trlab.close()
    telab.close()

    trainfolders = [os.path.join(trainfolder, str(i)) for i in range(10)]
    testfolders = [os.path.join(testfolder, str(i)) for i in range(10)]

    for dir in trainfolders:
        if not os.path.exists(dir):
            os.makedirs(dir)
    for dir in testfolders:
        if not os.path.exists(dir):
            os.makedirs(dir)
    string = "ori_data"
    f = open(dataset+"/"+string+".txt", 'a')
    for (i, label) in enumerate(trlabel):
        filename = os.path.join(trainfolders[label], str(i) + ".png")
        #print("writing " + filename)
        with open(filename, "wb") as img:
            image = png.Writer(28, 28, greyscale=True)
            data = [trimage[(i*28*28 + j*28) : (i*28*28 + (j+1)*28)] for j in range(28)]
            image.write(img, data)
            item = dict()
            item["image_file"] = filename
            item["id"] = [str(label)]
            item["size"] = {"width": 28, "height": 28}
            item["box"] = {"x": 0, "y": 0, "w": 28, "h": 28}
            f.writelines(json.dumps(item))
            f.write('\n')
        #break
    f.close()
    string="validation_test"
    f = open(dataset+"/"+string+".txt", 'a')
    for (i, label) in enumerate(telabel):
        filename = os.path.join(testfolders[label], str(i) + ".png")
        print("writing " + filename)
        with open(filename, "wb") as img:
            image = png.Writer(28, 28, greyscale=True)
            data = [teimage[(i*28*28 + j*28) : (i*28*28 + (j+1)*28)] for j in range(28)]
            image.write(img, data)
            item = dict()
            item["image_file"] = filename
            item["id"] = [str(label)]
            item["size"] = {"width": 28, "height": 28}
            item["box"] = {"x": 0, "y": 0, "w": 28, "h": 28}
            f.writelines(json.dumps(item))
            f.write('\n')
        #break
    f.close()
