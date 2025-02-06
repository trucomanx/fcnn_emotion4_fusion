# fcnn_emotion4_fusion
fcnn_emotion4_fusion

# Using library
Since the code uses an old version of keras, it needs to be placed at the beginning of the main.py code.

    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

    import FusionEmotion4Lib.Classifier as fec
    import numpy as np

    cls=fec.Emotion4Classifier();

    vec=np.random.rand(12);

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
    


