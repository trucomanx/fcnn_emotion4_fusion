#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="kfold_data_results.json"

model_list=["cls_minus81_ncod5",
            "cls_minus81_ncod6",
            "cls_minus81_ncod7",
            "cls_minus81_ncod8",
            "cls_minus81_ncod9",
            "cls_minus81_ncod10",
            "cls_minus81_ncod11",
            "cls_minus81_ncod12",
            "cls_minus81_ncod13",
            "cls_minus81_ncod14",
            "cls_minus81_ncod15",
            "cls_minus81_ncod16",
            "cls_minus81_ncod17",
            "cls_minus81_ncod18",
            "cls_minus81_ncod19",
            "cls_minus81_ncod20",
            "cls_minus81_ncod21",
            "cls_minus81_ncod22",
            "cls_minus81_ncod23",
            "cls_minus81_ncod24",
            "cls_minus81_ncod25",
            "cls_minus81_ncod26",
            "cls_minus81_ncod27",
            "cls_minus81_ncod28",
            "cls_minus81_ncod29",
            "cls_minus81_ncod30"
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
BaseDir='/media/maquina02/HD/Dados/Fernando'
#BaseDir='/media/fernando/Expansion'
#BaseDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_fusion_full'

DName='full2024-fusion-minus81'


if [ "$DName" = "full2024-fusion-minus81" ]; then
    InTrD=$BaseDir'/DATASET/TESE/full2024-fusion/ncod81_efficientnet_b3_efficientnet_b3_minus'
    InTrF='train.csv'
    InDmD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/dummy/L30000_p0.15'
    InDmF='train.csv'
fi
################################################################################

mkdir -p $OutDir/$DName/cross-validation_minus81
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/cross-validation_minus81/'main.py'

################################################################################

ipynb-py-convert kfold_validation.ipynb kfold_validation.py

for ncod in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 ; do # 
    echo " "
    python3 kfold_validation.py --epochs  2000 \
                                --patience 300 \
                                --seed 0 \
                                --ncod $ncod \
                                --batch-size 1024 \
                                --dataset-dir $InTrD \
                                --dataset-file $InTrF \
                                --dataset-dummy-dir $InDmD \
                                --dataset-dummy-file $InDmF \
                                --dataset-name $DName \
                                --output-dir $OutDir \
                                --minus 81

done

rm -f kfold_validation.py

