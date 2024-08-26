#!/usr/bin/python

import FusionEmotion4Lib.lib_model as mpp

class Emotion4Classifier:
    """Class to classify 4 body languages.
    
    The class Emotion4Classifier classify daa in 4 body languages.
    
    Args:
        :param file_of_weight: Archivo donde se encuentran los pesos.
        
    Atributos:
        modelo: Model returned by tensorflow.
    """
    def __init__(self,file_of_weight='',ncod=11, skel_size=None):
        """Inicializer of class Emotion4Classifier.
        
        Args:
            param file_of_weight: Archivo donde se encuentran los pesos.
        """
        
        if len(file_of_weight)>0:
            self.model = mpp.create_model(  load_weights=False,
                                            file_of_weight=file_of_weight,
                                            ncod=ncod,
                                            minus=skel_size);
        else:
            self.model = mpp.create_model(  load_weights=True,
                                            file_of_weight='',
                                            ncod=ncod,
                                            minus=skel_size);

    def from_skel_npvector(self,npvector):
        """Classify a skeleton data from a numpy vector object with N elements 
        
        Args:
            npvector: Numpy vector with N elements 
        
        Returns:
            int: The class.
        """
        return mpp.evaluate_model_from_npvector(self.model,
                                                npvector);

    def from_skel_npmatrix(self,npmatrix):
        """Classify a skeleton data from a numpy matrix object with N elements 
        
        Args:
            npmatrix: Numpy matrix with N columns and L lines
        
        Returns:
            numpy.vector: Numpy vector with The class.
        """
        return mpp.evaluate_model_from_npmatrix(self.model,
                                                npmatrix);

    def predict_vec(self,npvector):
        """Classify a skeleton data from a numpy vector object with N elements 
        
        Args:
            npvector: Numpy vector with N elements 
        
        Returns:
            int: The class of image.
        """
        return mpp.predict_model_from_npvector( self.model,
                                                npvector);

    def predict_mat(self,npmatrix):
        """Classify a skeleton data from a numpy matrix object with N columns ans L lines
        
        Args:
            npmatr: Numpy matrix with N columns and L lines
        
        Returns:
            int: The class of image.
        """
        return mpp.predict_model_from_npmatrix( self.model,
                                                npmatrix);

    def target_labels(self):
        """Returns the categories of classifier.
        
        Returns:
            list: The labels of categories resturned by the methods from_skeleton_npvector().
        """
        return ['negative','neutro','pain','positive'];


