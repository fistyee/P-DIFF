#!/usr/bin/env bash

# dataset: mnist, cifar10, cifar100, cloth1m.cnn, cloth1m.resnet101, miniimage
dataset=cifar100
if [ $# -eq 1 ]; then
    dataset=$1
fi



nohup /data/image_server/git/projects/image_server/lib/caffe/build/tools/caffe train -solver ./prototxt/solver.prototxt.${dataset} --gpu=2 2>&1 > ${dataset}_45.txt
