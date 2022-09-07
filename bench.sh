#!/bin/bash

export OMP_NUM_THREADS=1

mkdir -p output

set -eux

prefix=`date "+%Y%m%d-%H%M%S"`

python -u main.py classification --auto_augment_policy=all --random_erase_prob=all &> output/${prefix}_output_classification_all.log
python -u main.py detection --data_augmentation=all &> output/${prefix}_output_detection_all.log
