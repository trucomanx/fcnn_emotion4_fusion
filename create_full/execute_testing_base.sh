#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="testing_data_results.json"

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
            "cls_ncod30"
            ];

info_list=[ "test_categorical_accuracy",
            "test_loss",
            "number_of_parameters"
            ];

sep=",";

image_ext=".eps";
'

# HD
#BaseDir='/media/maquina02/HD/Dados/Fernando'
#BaseDir='/media/fernando/Expansion'
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'
BaseDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'


OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_fusion_full'

ModelBaseDir=$BaseDir'/OUTPUTS/DOCTORADO2/FUSION/fcnn_emotion4_fusion_full/full2024-fusion/training_validation_holdout'

DName='full2024-fusion'

################################################################################

#TestDName='full2024-fusion-drop-face25'
#TestDName='full2024-fusion-drop-face10'
#TestDName='full2024-fusion-drop-plus'
TestDName='full2024-fusion'


if [ "$TestDName" = "full2024-fusion-drop-face25" ]; then
    InTsD=$BaseDir'/DATASET/TESE/full2024-fusion-drop-face25/ncod81_efficientnet_b3_efficientnet_b3'
    InTsF='test.csv'
    BaseName='testing_base-drop-face25'
fi

if [ "$TestDName" = "full2024-fusion-drop-face10" ]; then
    InTsD=$BaseDir'/DATASET/TESE/full2024-fusion-drop-face10/ncod81_efficientnet_b3_efficientnet_b3'
    InTsF='test.csv'
    BaseName='testing_base-drop-face10'
fi

if [ "$TestDName" = "full2024-fusion-drop-plus" ]; then
    InTsD=$BaseDir'/DATASET/TESE/full2024-fusion-drop-plus/ncod81_efficientnet_b3_efficientnet_b3'
    InTsF='test.csv'
    BaseName='testing_base-drop-plus'
fi

if [ "$TestDName" = "full2024-fusion" ]; then
    InTsD=$BaseDir'/DATASET/TESE/full2024-fusion/ncod81_efficientnet_b3_efficientnet_b3'
    InTsF='test.csv'
    BaseName='testing_base'
fi

################################################################################

mkdir -p $OutDir/$DName/$BaseName
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/$BaseName/'main.py'

################################################################################

ipynb-py-convert testing_base.ipynb testing_base.py

NcodList=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 )

for i in "${!NcodList[@]}" ; do 
    echo " "
    Ncod=${NcodList[$i]}
    
    ModelFile=$ModelBaseDir/'cls_ncod'$Ncod'/model_ncod'$Ncod'.h5'
    python3 testing_base.py --ncod $Ncod \
                            --dataset-test-dir $InTsD \
                            --dataset-test-file $InTsF \
                            --dataset-name $DName \
                            --base-name $BaseName \
                            --model-file $ModelFile \
                            --output-dir $OutDir
done

rm -f testing_base.py

