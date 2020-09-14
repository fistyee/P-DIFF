#!/usr/bin/env bash

# dataset: mnist, cifar10, cifar100, miniimage,cloth1m.cnn,cloth1m.resnet101
dataset=cifar100
noise_type=SYMMETRY
noise_rate=20
if [ $# -eq 1 ]; then
    dataset=$1
    noise_type=$2
    noise_rate=$3
fi

if [ ${dataset} == "mnist" ]; then
    model_iter=93750
    test_iter=100
elif [ ${dataset} == "cifar10" ]; then
    model_iter=70000
    test_iter=100
elif [ ${dataset} == "cifar100" ]; then
    model_iter=70000
    test_iter=100
elif [ ${dataset} == "miniimage" ]; then
    model_iter=70000
    test_iter=100
elif [ ${dataset} == "cloth1m.cnn"]; then
    model_iter=6562500
    test_iter=400
fi

nohup /data/image_server/git/projects/image_server/lib/caffe/build/tools/caffe test -model ./prototxt/train_val.prototxt.${dataset} -weights ./model/${dataset}/${noise_type}_${noise_rate}/_iter_${model_iter}.caffemodel --iterations=${test_iter} 2>&1 > logg/${dataset}_${noise_type}_${noise_rate}.txt

