# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:51:01 2017

@author: lming
"""
import logging

import numpy as np
from keras.engine.saving import load_model

from keras.models import Sequential
from keras.layers import Activation, Dense, Input, Flatten,Dropout, Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding,GRU, SimpleRNN,GlobalAveragePooling1D, LSTM
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf

from keras import backend as K, optimizers

logger = logging.getLogger()

def getClassifier(num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, model_path=None, learning_rate=0.01):
    with tf.device('/cpu:0'):
        embedding_layer=Embedding((num_words+1), EMBEDDING_DIM, weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
        sequence_input=Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences=embedding_layer(sequence_input)

    with tf.device('/device:GPU:0'):
        embedded_sequences=Dropout(0.3)(embedded_sequences)
        #embedded_sequences=SimpleRNN(50, return_sequences=True)(embedded_sequences)
        embedded_sequences=Conv1D(50, 3, activation='relu')(embedded_sequences)
        #doc=AttentionWithContext()(embedded_sequences)
        doc=GlobalAveragePooling1D()(embedded_sequences)
        doc=Dense(2)(doc)
        preds=Activation('softmax')(doc)
    classifier = Model(sequence_input, preds)
    optimizer = optimizers.Adagrad(lr=learning_rate, decay=0.01)
    classifier.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['mse', 'accuracy'])

    if model_path is not None:
        logger.info(" >>> Load pretrained classifier at {} and transfer weights".format(model_path))
        pretrained_model = load_model(model_path)
        pretrained_layers = [l for l in pretrained_model.layers if "embedding" not in l.name]
        layers = [l for l in classifier.layers if "embedding" not in l.name]
        assert(len(pretrained_layers) == len(layers))
        for pre_l, cur_l in zip(pretrained_layers, layers):
            cur_l.set_weights(pre_l.get_weights())
    return classifier

def getPolicy(k, state_dim):
    policy = Sequential()
    policy.add(TimeDistributed(Dense(1), input_shape=(k, state_dim)))
    policy.add(Reshape((k,), input_shape=(k,1)))
    policy.add(Activation('softmax'))
    optimizer = optimizers.Adam(lr=1e-4)
    policy.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['mse', 'accuracy'])
    return policy
    
def getState(x_trn, y_trn, x_new, model):
    docembeddings=get_intermediatelayer(model,4,x_trn)
    #bag of embeddings from training set
    trainembedding=sum(docembeddings)[0]
    #ratio of pos/neg labels in training set
    count_y_trn=sum(y_trn)
    trainratio=count_y_trn/sum(count_y_trn)
    #expand the dimension
    x_candi = np.expand_dims(x_new, axis=0)
    #candidate embedding
    newembedding=get_intermediatelayer(model, 4, x_candi)
    candiembedding=sum(newembedding)[0]
    #expected prediction
    y_predicted=model.predict(x_candi)
    candiprediction=sum(y_predicted)
    #concatenate all 4 arrays into a single array
    state=np.concatenate([trainembedding, trainratio, candiembedding, candiprediction])
    return state
        
def getAState(x_trn, y_trn, x_neglist, model):
    samples=[]
    for point in x_neglist:
        s=getState(x_trn, y_trn, point, model)
        samples.append(s)
    return samples
    

def getbottleFeature(model,X):
    Inp = model.input
    Outp = model.get_layer('Hidden').output
    curr_layer_model = Model(Inp, Outp)
    bottle_feature = curr_layer_model.predict(X)
    return bottle_feature


#get the output of intermediate layer
def get_intermediatelayer(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations    