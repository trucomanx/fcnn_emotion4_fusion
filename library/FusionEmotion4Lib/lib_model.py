#!/usr/bin/python

import os
import sys
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


def load_model_from_intern(model,model_fname):
    path_actual = os.path.realpath(__file__);
    directorio_actual = os.path.dirname(path_actual);
    path_of_model=os.path.join(directorio_actual,'models',model_fname);
    
    if os.path.exists(path_of_model):
        print('')
        print("Loading the weights in:",path_of_model);
        try:
            model.load_weights(path_of_model);
            print("Loaded the weights in:",path_of_model);
            
        except Exception:
            print("Error loading the weights in:",path_of_model);
            exit();
    else:
        print("Error loading, file no found:",path_of_model);
        exit();
    return model;

def load_model_from_extern(model,file_of_weight):
    print('')
    print("Loading the weights in:",file_of_weight);
    if os.path.exists(file_of_weight):
        #
        try:
            obj=model.load_weights(file_of_weight);
            print("Loaded the weights in:",file_of_weight);
        except Exception:
            print("Error loading the weights in:",file_of_weight);
            exit();
    else:
        print("Error loading, file no found:",file_of_weight);
        exit();
    
    return model;




def create_model(load_weights=True, file_of_weight='', ncod=11, minus=None):
    '''
    Retorna un modelo para la clasificación.
    Adicionalmente, si el archivo `file_of_weight` existe los pesos son cargados.
    
    :param file_of_weight: Archivo donde se encuentran los pesos.
    :type file_of_weight: str
    :return: Retorna un modelo de red neuronal
    :rtype: tensorflow.python.keras.engine.sequential.Sequential
    '''
    
    func_act=tf.keras.layers.LeakyReLU(alpha=0.01);
    
    if minus is None:
        # modelo nuevo
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(12,) ),
            tf.keras.layers.Dense(ncod, activation=func_act),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
    else:
        # modelo nuevo
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(minus+8,) ),
            tf.keras.layers.Dense(2*ncod+1, activation=func_act),
            tf.keras.layers.Dense(ncod, activation=func_act),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
    
    if load_weights==True:
        if minus is None:
            model=load_model_from_intern(model,'model_ncod'+str(ncod)+'.h5');
        else:
            model=load_model_from_intern(model,'model_minus'+str(minus)+'_ncod'+str(ncod)+'.h5');
    
    if len(file_of_weight)!=0:
        model=load_model_from_extern(model,file_of_weight);
    
    return model;

def evaluate_model_from_npvector(model, npvector):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde um numpy vector.
    
    :param model: Modelo de la red neuronal.
    :type model: tensorflow.python.keras.engine.sequential.Sequential
    :param npvector: Vector a testar.
    :type npvector: Numpy array
    :return: Retorna la classificación.
    :rtype: int
    '''
    #vec = npvector.reshape(1,npvector.size);
    vec = np.expand_dims(npvector, axis=0);
    
    # Verifique o tipo de dado e converta se necessário
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    
    res=model.predict( vec, verbose=0);
    
    return np.argmax(res);

def predict_model_from_npvector(model, npvector):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde um numpy vector.
    
    :param model: Modelo de la red neuronal.
    :type model: tensorflow.python.keras.engine.sequential.Sequential
    :param npvector: Vector a testar.
    :type npvector: Numpy array
    :return: Retorna la classificación.
    :rtype: int
    '''
    #vec = npvector.reshape(1,npvector.size);
    vec = np.expand_dims(npvector, axis=0);
    
    # Verifique o tipo de dado e converta se necessário
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    
    res=model.predict( vec, verbose=0);
    
    return res[0];    

def save_model_history(hist, fpath,show=True, labels=['accuracy','loss']):
    ''''This function saves the history returned by model.fit to a tab-
    delimited file, where model is a keras model'''

    acc      = hist.history[labels[0]];
    val_acc  = hist.history['val_'+labels[0]];
    loss     = hist.history[labels[1]];
    val_loss = hist.history['val_'+labels[1]];

    EPOCAS=len(acc);
    
    rango_epocas=range(EPOCAS);

    plt.figure(figsize=(16,8))
    #
    plt.subplot(1,2,1)
    plt.plot(rango_epocas,    acc,label=labels[0]+' training')
    plt.plot(rango_epocas,val_acc,label=labels[0]+' validation')
    plt.legend(loc='upper right')
    #plt.title('Analysis accuracy')
    plt.ylabel(labels[0])
    plt.xlabel('Epochs')
    #
    plt.subplot(1,2,2)
    plt.semilogy(rango_epocas,    loss,label=labels[1]+' training')
    plt.semilogy(rango_epocas,val_loss,label=labels[1]+' validation')
    plt.legend(loc='upper right')
    #plt.title('Analysis loss')
    plt.ylabel(labels[1])
    plt.xlabel('Epochs')
    #
    plt.savefig(fpath+'.plot.png')
    if show:
        plt.show()
    
    print('max_val_acc', np.max(val_acc))
    
    ###########
    
    # Open file
    fid = open(fpath, 'w')
    print(labels[0]+',val_'+labels[0]+','+labels[1]+',val_'+labels[1], file = fid)

    try:
        # Iterate through
        for i in rango_epocas:
            print('{},{},{},{}'.format(acc[i],val_acc[i],loss[i],val_loss[i]),file = fid)
    except KeyError:
        print('<no history found>', file = fid)

    # Close file
    fid.close()
    
    return acc, val_acc

def save_model_stat_kfold(VALIDATION_ACCURACY,VALIDATION_LOSS, fpath):
    '''
    Salva los datos de accuracy y loss en un archivo de tipo m.
    
    :param VALIDATION_ACCURACY: Lista de accuracies
    :type VALIDATION_ACCURACY: list of floats
    :param VALIDATION_LOSS: Lista de loss
    :type VALIDATION_LOSS: list of floats
    :param fpath: Archivo donde se guardaran los datos.
    :type fpath: str
    :return: Retorna el valor medio de las acuracias.
    :rtype: float
    '''
    fid = open(fpath, 'w')
    
    #
    print('mean_val_acc={}'.format(np.mean(VALIDATION_ACCURACY)),';', file = fid)
    
    #
    print('std_val_acc={}'.format(np.std(VALIDATION_ACCURACY)),';', file = fid)
    
    #
    print('mean_val_loss={}'.format(np.mean(VALIDATION_LOSS)),';', file = fid)
    
    #
    print('std_val_loss={}'.format(np.std(VALIDATION_LOSS)),';', file = fid)
    
    #
    print('val_acc=[', end='', file = fid)
    k=1;
    for value in VALIDATION_ACCURACY:
        if k==len(VALIDATION_ACCURACY):
            print('{}'.format(value),end='', file = fid);
        else:
            print('{}'.format(value),end=';', file = fid);
        k=k+1;
    print('];', file = fid)
    
    #
    print('val_loss=[', end='', file = fid)
    k=1;
    for value in VALIDATION_LOSS:
        if k==len(VALIDATION_LOSS):
            print('{}'.format(value),end='', file = fid);
        else:
            print('{}'.format(value),end=';', file = fid);
        k=k+1;
    print('];', file = fid)
    
    fid.close()
    return np.mean(VALIDATION_ACCURACY);


def get_model_parameters(model):
    return model.count_params();

from tensorflow.python.keras.utils.layer_utils import count_params
def save_model_parameters(model, fpath):
    '''
    Salva en un archivo la estadistica de la cantidoda de parametros de un modelo
    
    :param model: Modelos a analizar
    :type model: str
    :param fpath: Archivo donde se salvaran los datos.
    :type fpath: str
    '''
    trainable_count = count_params(model.trainable_weights)
    
    fid = open(fpath, 'w')
    print('parameters_total={}'.format(model.count_params()),';', file = fid);
    print('parameters_trainable={}'.format(trainable_count),';', file = fid);
    fid.close();

