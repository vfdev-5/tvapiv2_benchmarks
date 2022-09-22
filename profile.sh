#!/bin/bash

export OMP_NUM_THREADS=1

mkdir -p output

set -eux

prefix=`date "+%Y%m%d-%H%M%S"`

# python -u main.py profile_transform --t_name=RandomResizedCrop --t_args="(224,)" &> output/${prefix}_profile_randomresizedcrop.log

# python -u main.py profile_transform --t_name=RandomErasing --t_args="(1.0, )" --single_dtype=Tensor &> output/${prefix}_profile_randomerasing.log

python -u main.py profile_transform --t_name=ScaleJitter --t_args="((1024, 1024), )" --single_dtype=Feature &> output/${prefix}_profile_scalejitter.log

python -u main.py profile_transform --t_name=FixedSizeCrop --t_args="((1024, 1024), )" --t_kwargs="{'fill': 0}" --single_dtype=Feature &> output/${prefix}_profile_fixedsizecrop.log
