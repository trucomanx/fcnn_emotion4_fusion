#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="training_data_results.json"

model_list=["cls_ncod6",
            "cls_ncod7",
            "cls_ncod8",
            "cls_ncod9",
            "cls_ncod10",
            "cls_ncod11",
            "cls_ncod12",
            "cls_ncod13",
            "cls_ncod14",
            "cls_ncod15",
            "cls_ncod16"
            ];

info_list=[ "train_categorical_accuracy",
            "train_loss",
            "val_categorical_accuracy",
            "val_loss",
            "test_categorical_accuracy",
            "test_loss",
            "number_of_parameters"
            ];

sep=",";

image_ext=".eps";
'

# HD
BaseDir='/media/fernando/Expansion'
# 
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_fusion_1'

DName='ber2024-fusion'

InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION'
InTrF='train.csv'
InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/ber2024-source/ncod20_efficientnet_b3_efficientnet_b3_step1'
InTsF='test.csv'

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout/'main.py'

################################################################################

ipynb-py-convert training_holdout.ipynb training_holdout.py

for ncod in 6 7 8 9 10 11 12 13 14 15 16 ; do
    echo " "
    python3 training_holdout.py --epochs  5000 \
                                --patience 500 \
                                --seed 0 \
                                --ncod $ncod \
                                --batch-size 1024 \
                                --dataset-train-dir $InTrD \
                                --dataset-train-file $InTrF \
                                --dataset-test-dir $InTsD \
                                --dataset-test-file $InTsF \
                                --dataset-name $DName \
                                --output-dir $OutDir
done




rm -f training_holdout.py

