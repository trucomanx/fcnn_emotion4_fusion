#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="training_data_results.json"

model_list=["cls_ncod7",
            "cls_ncod9",
            "cls_ncod11",
            "cls_ncod13",
            "cls_ncod15",
            "cls_ncod17",
            "cls_ncod19",
            "cls_ncod21"
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
InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/ber2024-source/efficientnet_b3_efficientnet_b3_ncod20'
InTsF='test.csv'

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout/'main.py'

################################################################################

ipynb-py-convert training_holdout.ipynb training_holdout.py

for ncod in 7 9 11 13 15 17 19 21; do
    echo " "
    python3 training_holdout.py --epochs  2000 \
                                --patience 100 \
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

