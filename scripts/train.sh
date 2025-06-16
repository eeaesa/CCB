#!/bin/bash
export PATH=/home/a100/anaconda3/envs/hdc/bin:$PATH
now=$(date +"%Y%m%d_%H%M%S")

cd /mnt/home/hdc/WorkSpace_lab/Semi-Supervised-Semantic-Segmentation/my-4s-work/DTPC3
# cd scripts
# sh train.sh

config_isic2018=configs/isic2018.yaml
config_KvasirSEG=configs/KvasirSEG.yaml
i=1_4

python train_2d_DTPC.py --config $config_isic2018 --label-rat $i
python train_2d_DTPC.py --config $config_KvasirSEG --label-rat $i

config_ACDC=configs/ACDC.yaml
i=3
python train_2d_ACDC_DTPC.py --config $config_ACDC --label-rat $i
