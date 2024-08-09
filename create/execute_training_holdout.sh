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

InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/ber2024-source/ncod20_efficientnet_b3_efficientnet_b3_step1'
InTrF='train.csv'
InDmD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/dummy/L30000_p0.15'
InDmF='test.csv'
InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/ber2024-source/ncod20_efficientnet_b3_efficientnet_b3_step1'
InTsF='test.csv'

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout/'main.py'

################################################################################

ipynb-py-convert training_holdout.ipynb training_holdout.py

NcodList=(6 7 8 9 10 11 12 13 14 15 16)
SeedList=(0 0 0 0 0  0  0  0  0  0  0)

#NcodList=(11 11 11 11 11 11)
#SeedList=(19 23 29 31 37 41)

for i in "${!NcodList[@]}" ; do 
    echo " "
    Ncod=${NcodList[$i]}
    Seed=${SeedList[$i]}
    python3 training_holdout.py --epochs  500 \
                                --patience 50 \
                                --seed $Seed \
                                --ncod $Ncod \
                                --batch-size 1024 \
                                --dataset-train-dir $InTrD \
                                --dataset-train-file $InTrF \
                                --dataset-dummy-dir $InDmD \
                                --dataset-dummy-file $InDmF \
                                --dataset-test-dir $InTsD \
                                --dataset-test-file $InTsF \
                                --dataset-name $DName \
                                --output-dir $OutDir
done




rm -f training_holdout.py

