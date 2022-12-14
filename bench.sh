#!/bin/bash

export OMP_NUM_THREADS=1

mkdir -p output

vision_commit=$(cd /vision && git log -1 --format=format:"%h")

set -eux

prefix=`date "+%Y%m%d-%H%M%S"`
echo ${vision_commit}

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

# python -u main.py detection --with_time --data_augmentation=all &> output/${prefix}_output_detection_all_${vision_commit}.log
# python -u main.py classification --with_time --auto_augment_policy=all --random_erase_prob=all &> output/${prefix}_output_classification_all_${vision_commit}.log
# python -u main.py segmentation --with_time &> output/${prefix}_output_segmentation_all_${vision_commit}.log

# python -u main.py classification --with_time --single_dtype=Feature --auto_augment_policy=imagenet --random_erase_prob=1.0 &> output/${prefix}_output_classification_imagenet_re_${vision_commit}.log
# python -u main.py classification_pil_vs_features --with_time --auto_augment_policy=all --random_erase_prob=all &> output/${prefix}_output_classification_all_pil_vs_features_${vision_commit}.log
# python -u main.py classification --with_time --single_dtype=PIL --auto_augment_policy=all --random_erase_prob=all &> output/${prefix}_output_classification_all_pil_${vision_commit}.log

# python -u main.py classification --with_time --single_dtype=Feature --auto_augment_policy=all --random_erase_prob=all &> output/${prefix}_output_classification_all_${vision_commit}.log

# python -u main.py single_transform --t_name=RandomEqualize --t_args="(1.0,)" --single_dtype=Tensor &> output/${prefix}_output_classification_all_pil_${vision_commit}.log

# python -u main.py all_transforms &> output/${prefix}_all_transforms_${vision_commit}.log

# python -u main.py classification --with_time --single_dtype="(Tensor, Feature)" --single_api="v2" --auto_augment_policy=all --random_erase_prob=all &> output/${prefix}_output_classification_all_ten_vs_feat_v2_${vision_commit}.log

# SOMETHING WRONG HERE: >>> python -u main.py single_transform --t_name=GaussianBlur --t_args="(3, 0.7)" --single_dtype=Tensor
# OMP_NUM_THREADS=1 python -u main.py single_op --f_name=gaussian_blur --f_kwargs='{"kernel_size": 3, "sigma": 0.7}' --single_dtype=Tensor

# python -u check_adjust_color_ops.py &> output/$(date "+%Y%m%d-%H%M%S")-output-adjust-color-ops.log

# OMP_NUM_THREADS=6 python -u check_flips_bboxes.py &> output/${prefix}-output-flips-bboxes_${vision_commit}.log

# OMP_NUM_THREADS=6 python -u check_crop_bboxes.py &> output/${prefix}-output-crop-bboxes_${vision_commit}.log

# OMP_NUM_THREADS=6 python -u check_pad_bboxes.py &> output/${prefix}-output-pad-bboxes_${vision_commit}.log

# OMP_NUM_THREADS=6 python -u check_perspective_bboxes.py &> output/${prefix}-output-perspective-bboxes_${vision_commit}.log

OMP_NUM_THREADS=6 python -u check_elastic_bboxes.py &> output/${prefix}-output-elastic-bboxes_${vision_commit}.log