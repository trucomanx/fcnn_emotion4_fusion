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
            "cls_ncod16",
            "cls_ncod17",
            "cls_ncod18",
            "cls_ncod19",
            "cls_ncod20",
            "cls_ncod21",
            "cls_ncod22",
            "cls_ncod23",
            "cls_ncod24",
            "cls_ncod25",
            "cls_ncod26",
            "cls_ncod27",
            "cls_ncod28",
            "cls_ncod29",
            "cls_ncod30",
            "cls_ncod31",
            "cls_ncod32",
            "cls_ncod33",
            "cls_ncod34",
            "cls_ncod35",
            "cls_ncod36",
            "cls_ncod37",
            "cls_ncod38",
            "cls_ncod39",
            "cls_ncod40"
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
#BaseDir='/media/maquina02/HD/Dados/Fernando'
#BaseDir='/media/fernando/Expansion'
BaseDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'
# 
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_fusion_full'

DName='full2024-fusion-drop-plus'

InTrD=$BaseDir'/DATASET/TESE/PER/PER-TOOLS/dataset_fusion/full2024-fusion-drop-plus/ncod81_efficientnet_b3_efficientnet_b3'
InTrF='train.csv'
InDmD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-FUSION/dummy/L30000_p0.15'
InDmF='train.csv'
InTsD=$BaseDir'/DATASET/TESE/PER/PER-TOOLS/dataset_fusion/full2024-fusion-drop-plus/ncod81_efficientnet_b3_efficientnet_b3'
InTsF='test.csv'

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout/'main.py'

################################################################################

ipynb-py-convert training_holdout.ipynb training_holdout.py

#NcodList=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
#SeedList=(0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0)

NcodList=(31 32 33 34 35 36 37 38 39 40)
SeedList=(0  0  0  0  0  0  0  0  0  0)

for i in "${!NcodList[@]}" ; do 
    echo " "
    Ncod=${NcodList[$i]}
    Seed=${SeedList[$i]}
    python3 training_holdout.py --epochs  2000 \
                                --patience 200 \
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

