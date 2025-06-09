# fcnn_emotion4_fusion
fcnn_emotion4_fusion

# Using library
Since the code uses an old version of keras, it needs to be placed at the beginning of the main.py code.

    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

    import FusionEmotion4Lib.Classifier as fec
    import numpy as np

    # cls=fec.Emotion4Classifier(ncod=20,skel_size=None); # Without drop-face and without skel minus
    # cls=fec.Emotion4Classifier(ncod=11,skel_size=81);   # Without drop-face and with skel minus
    # cls=fec.Emotion4Classifier(ncod=39,skel_size=81);   # With drop-face and with skel minus
    cls=fec.Emotion4Classifier(ncod=39,skel_size=81);

    vec=np.random.rand(89); # 81+4+4

    res=cls.predict_vec(vec);

    print(res);


# Installation summary - Dataset BER2024

    git clone https://github.com/trucomanx/fcnn_emotion4_fusion
    gdown 19I8TAOQhi2NMz-I81ih5Lz8zDXG-7y4O
    unzip models_fusion_v2.zip -d fcnn_emotion4_fusion/library/FusionEmotion4Lib/models
    cd fcnn_emotion4_fusion/library
    python3 setup.py sdist
    pip3 install dist/FusionEmotion4Lib-*.tar.gz

# Installation summary - Dataset FULL2024

    git clone https://github.com/trucomanx/fcnn_emotion4_fusion
    gdown 1gk8BYQDDF_8t_IUC4tLjYXxdWjOFIWtE
    unzip models_fusion_full.zip -d fcnn_emotion4_fusion/library/FusionEmotion4Lib/models
    cd fcnn_emotion4_fusion/library
    python3 setup.py sdist
    pip3 install dist/FusionEmotion4Lib-*.tar.gz
    
# Installation summary - Dataset FULL2024-DROP-FACE

    git clone https://github.com/trucomanx/fcnn_emotion4_fusion
    gdown 1DFduCOACBDi7AM5rstCItWP3tRj7Jqc4
    unzip models_fusion_full_drop_plus.zip -d fcnn_emotion4_fusion/library/FusionEmotion4Lib/models
    cd fcnn_emotion4_fusion/library
    python3 setup.py sdist
    pip3 install dist/FusionEmotion4Lib-*.tar.gz
 
    

