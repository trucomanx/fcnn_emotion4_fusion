#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="kfold_data_results.json"

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
            "cls_minus53_ncod16",
            "cls_minus53_ncod17",
            "cls_minus53_ncod18",
            "cls_minus53_ncod19",
            "cls_minus53_ncod20"
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
#BaseDir='/media/fernando/Expansion'
BaseDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_fusion_v2'

DName='ber2024-fusion-v2-minus53'


if [ "$DName" = "ber2024-fusion-v2-minus53" ]; then
    InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/ber2024-source/ncod53_efficientnet_b3_efficientnet_b3_minus_v2'
    InTrF='train.csv'
    InDmD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/dummy/L30000_p0.15'
    InDmF='test.csv'
fi
################################################################################

mkdir -p $OutDir/$DName/cross-validation_minus53
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/cross-validation_minus53/'main.py'

################################################################################

ipynb-py-convert kfold_validation.ipynb kfold_validation.py

for ncod in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 ; do # 
    echo " "
    python3 kfold_validation.py --epochs  1000 \
                                --patience 250 \
                                --seed 0 \
                                --ncod $ncod \
                                --batch-size 1024 \
                                --dataset-dir $InTrD \
                                --dataset-file $InTrF \
                                --dataset-dummy-dir $InDmD \
                                --dataset-dummy-file $InDmF \
                                --dataset-name $DName \
                                --output-dir $OutDir \
                                --minus 53

done

rm -f kfold_validation.py

