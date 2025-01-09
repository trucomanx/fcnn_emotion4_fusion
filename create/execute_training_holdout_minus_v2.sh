#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="training_data_results.json"

model_list=["cls_minus53_ncod5",
            "cls_minus53_ncod6",
            "cls_minus53_ncod7",
            "cls_minus53_ncod8",
            "cls_minus53_ncod9",
            "cls_minus53_ncod10",
            "cls_minus53_ncod11",
            "cls_minus53_ncod12",
            "cls_minus53_ncod13",
            "cls_minus53_ncod14",
            "cls_minus53_ncod15",
            "cls_minus53_ncod16"
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
#BaseDir='/media/fernando/Expansion'
BaseDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'
# 
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_fusion_v2'

DName='ber2024-fusion-v2-minus53'

InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/ber2024-source/ncod53_efficientnet_b3_efficientnet_b3_minus_v2'
InTrF='train.csv'
InDmD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/dummy/L30000_p0.15'
InDmF='train.csv'
InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/ber2024-source/ncod53_efficientnet_b3_efficientnet_b3_minus_v2'
InTsF='test.csv'

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout/'main.py'

################################################################################

ipynb-py-convert training_holdout.ipynb training_holdout.py

NcodList=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
SeedList=(0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0)

#NcodList=(11 11 11 11 11 11)
#SeedList=(19 23 29 31 37 41)

for i in "${!NcodList[@]}" ; do 
    echo " "
    Ncod=${NcodList[$i]}
    Seed=${SeedList[$i]}
    python3 training_holdout.py --epochs  1000 \
                                --patience 300 \
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
                                --output-dir $OutDir \
                                --minus 53
done




rm -f training_holdout.py

