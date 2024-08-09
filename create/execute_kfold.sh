#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="kfold_data_results.json"

model_list=["cls_ncod6",
            "cls_ncod7",
            "cls_ncod8",
            "cls_ncod9",
            "cls_ncod10",
            "cls_ncod11",
            "cls_ncod12",
            "cls_ncod13"
            ];

info_list=[ "mean_val_categorical_accuracy",
            "std_val_categorical_accuracy",
            "mean_val_loss",
            "mean_train_categorical_accuracy",
            "mean_train_loss",
            "number_of_parameters"];

erro_bar=[("mean_val_categorical_accuracy","std_val_categorical_accuracy")];

p_matrix="val_categorical_accuracy";

sep=",";

image_ext=".eps";
'

# HD
BaseDir='/media/fernando/Expansion'
# 
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_fusion_1'

DName='ber2024-fusion'

if [ "$DName" = "ber2024-fusion" ]; then
    InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/ber2024-source/ncod20_efficientnet_b3_efficientnet_b3_step1'
    InTrF='train.csv'
    InDmD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/dummy/L30000_p0.15'
    InDmF='test.csv'
fi

################################################################################

mkdir -p $OutDir/$DName/cross-validation
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/cross-validation/'main.py'

################################################################################

ipynb-py-convert kfold_validation.ipynb kfold_validation.py

for ncod in 4 21 22 23 24 25 26 ; do #5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    echo " "
    python3 kfold_validation.py --epochs  500 \
                                --patience 50 \
                                --seed 0 \
                                --ncod $ncod \
                                --batch-size 1024 \
                                --dataset-dir $InTrD \
                                --dataset-file $InTrF \
                                --dataset-dummy-dir $InDmD \
                                --dataset-dummy-file $InDmF \
                                --dataset-name $DName \
                                --output-dir $OutDir

done

rm -f kfold_validation.py

