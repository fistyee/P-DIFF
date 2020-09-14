#!/usr/bin/env bash
#author:huangyangyu

#dataset:mnist, cifar10, cifar100
dataset=cifar10
noise_type=SYMMETRY
noise_rate=45
if [ $# -eq 3 ]; then
    dataset=$1
    noise_type=$2
    noise_rate=$3
fi
data_dir=./data/${dataset}/${noise_type}_${noise_rate}/
if [ ${dataset} == "mnist" ]; then
    gray=--gray
    resize_height=28
    resize_width=28
elif [ ${dataset} == "cifar10" ]; then
    gray=
    resize_height=32
    resize_width=32
elif [ ${dataset} == "cifar100" ]; then
    gray=
    resize_height=32
    resize_width=32
elif [ ${dataset} == "clothing1m" ]; then
    gray=
    resize_height=256
    resize_width=256
fi

# train lmdb
./caffe/build/tools/convert_imageset ./ ${data_dir}train.txt ${data_dir}train_db/ ${gray} -resize_height ${resize_height} -resize_width ${resize_width} -backend lmdb
# test lmdb
./caffe/build/tools/convert_imageset ./ ${data_dir}test.txt ${data_dir}test_db/ ${gray} -resize_height ${resize_height} -resize_width ${resize_width} -backend lmdb


