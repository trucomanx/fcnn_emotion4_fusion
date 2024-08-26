#!/usr/bin/python3

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
sys.path.append('../library');

import FusionEmotion4Lib.Classifier as fec
import numpy as np

cls=fec.Emotion4Classifier(ncod=11);

vec=np.random.randn(12);

print("")
print("cls.predict_vec(vec)")
res=cls.predict_vec(vec);
print(res);

mat=np.stack([vec,vec]);

print("")
print("cls.predict_mat(mat)")
res=cls.predict_mat(mat);
print(res);

print("")
print("cls.from_skel_npmatrix(mat)")
res=cls.from_skel_npmatrix(mat);
print(res);
