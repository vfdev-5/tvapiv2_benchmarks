#!/bin/bash

python -u main.py classification --auto_augment_policy=all --random_erase_prob=all &> output_classification_all.log
