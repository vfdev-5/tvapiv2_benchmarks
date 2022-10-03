#!/bin/bash

export OMP_NUM_THREADS=1

mkdir -p output

set -eux

prefix=`date "+%Y%m%d-%H%M%S"`


# python -u main.py detection --with_time --data_augmentation=lsj --single_dtype=PIL &> tmp/$prefix-with_time-out1.log.tmp
# python -u main.py debug_det --data_augmentation=lsj

# python -u main.py detection --data_augmentation=lsj &> $prefix-out2.log.tmp


# python -u main.py with_time --data_augmentation=lsj &> $prefix-with_time-out1.log.tmp
# python -u main.py with_time --data_augmentation=lsj --single_dtype=Feature &> $prefix-with_time-out2.log.tmp
# python -u main.py with_time --data_augmentation=lsj --single_dtype=Tensor &> $prefix-with_time-out3.log.tmp
# python -u main.py with_time --data_augmentation=lsj --single_dtype=PIL &> $prefix-with_time-out4.log.tmp

# python -u main.py detection --data_augmentation=multiscale &> $prefix-out1.log.tmp
# python -u main.py detection --data_augmentation=multiscale --single_dtype=Feature &> $prefix-out2.log.tmp
# python -u main.py detection --data_augmentation=multiscale --single_dtype=Tensor &> $prefix-out3.log.tmp
# python -u main.py detection --data_augmentation=multiscale --single_dtype=PIL &> $prefix-out4.log.tmp

# python -u main.py classification --auto_augment_policy=ra &> $prefix-out1.log.tmp
# python -u main.py classification --auto_augment_policy=ra --single_dtype=Feature &> $prefix-out2.log.tmp
# python -u main.py classification --auto_augment_policy=ra --single_dtype=Tensor &> $prefix-out3.log.tmp
# python -u main.py classification --auto_augment_policy=ra --single_dtype=PIL &> $prefix-out4.log.tmp

# python -u main.py detection --data_augmentation=lsj &> $prefix-out1.log.tmp
# python -u main.py detection --data_augmentation=lsj --single_dtype=Feature &> $prefix-out2.log.tmp
# python -u main.py detection --data_augmentation=lsj --single_dtype=Tensor &> $prefix-out3.log.tmp
# python -u main.py detection --data_augmentation=lsj --single_dtype=PIL &> $prefix-out4.log.tmp

# python -u main.py detection --with_time --data_augmentation=all &> output/${prefix}_output_detection_all.log
# python -u main.py classification --with_time --auto_augment_policy=all --random_erase_prob=all &> output/${prefix}_output_classification_all.log
# python -u main.py segmentation --with_time &> output/${prefix}_output_segmentation_all.log

python -u main.py classification_pil_vs_features --with_time --auto_augment_policy=all --random_erase_prob=all &> output/${prefix}_output_classification_all_pil_vs_features.log