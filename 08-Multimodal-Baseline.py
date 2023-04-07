# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Changes made by kts4 / noami2
# - `glove` was not available and so needed to install `mittens` instead
#   - `from mittens import GloVe as glove`
# - Removed `merge` from import from `keras.layers`
#   - No longer available and not used.
# - `set_session`, `clear_session` and `get_session` are no longer available from `keras.backend.tensorflow_backend`
#   - Loaded from `tf.compat.v1.keras.backend` and `tf.keras.backend` instead
# - `reset_keras()` function was missing. 
#   - Copied over from nb7
# - `tf.contrib.layers.l2_regularizer` no longer available
#   - Used `tf.keras.regularizers.l2` instead
# - `tf.contrib.layers.xavier_initializer` no longer available
#   - Used `tf.keras.initializers.GlorotUniform` instead (TODO: this is different to nb7!)
# - Needed to add function for `mean` so that the pickle files could be loaded
#   - Otherwise got error
# - We do not have the `FastText` model and so all references to it and combined model have to be commented out
# - `iter_num` was set to `2` rather than `11`
#   - Need it at `11` so that there are 10 runs of the code
# - Changes to newlines / spacings / printing etc.

# %%
import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec, FastText
from mittens import GloVe as glove
# from glove import Corpus

import collections
import gc 

import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input, concatenate, Activation, Concatenate, LSTM, GRU
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, BatchNormalization, GRU, Convolution1D, LSTM
from keras.layers import UpSampling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,MaxPool1D

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
from keras.utils import np_utils
# from keras.backend.tensorflow_backend import set_session, clear_session, get_session
import tensorflow as tf


from sklearn.utils import class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')


# %%
# Reset Keras Session
def reset_keras(model):
    sess = tf.compat.v1.keras.backend.get_session()
    tf.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    gc.collect() # if it's done something you should see a number being outputted

def create_dataset(dict_of_ner):
    temp_data = []
    for k, v in sorted(dict_of_ner.items()):
        temp = []
        for embed in v:
            temp.append(embed)
        temp_data.append(np.mean(temp, axis = 0)) 
    return np.asarray(temp_data)

def make_prediction_multi_avg(model, test_data):
    probs = model.predict(test_data)
    y_pred = [1 if i>=0.5 else 0 for i in probs]
    return probs, y_pred

def save_scores_multi_avg(predictions, probs, ground_truth,                           
                          embed_name, problem_type, iteration, hidden_unit_size,                          
                          sequence_name, type_of_ner):
    
    auc   = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc   = accuracy_score(ground_truth, predictions)
    F1    = f1_score(ground_truth, predictions)
    
    result_dict = {}    
    result_dict['auc'] = auc
    result_dict['auprc'] = auprc
    result_dict['acc'] = acc
    result_dict['F1'] = F1
    
    result_path = "results/"
    file_name = str(sequence_name)+"-"+str(hidden_unit_size)+"-"+embed_name
    file_name = file_name +"-"+problem_type+"-"+str(iteration)+"-"+type_of_ner+"-avg-.p"
    pd.to_pickle(result_dict, os.path.join(result_path, file_name))

    print(auc, auprc, acc, F1)
    
def avg_ner_model(layer_name, number_of_unit, embedding_name):

    if embedding_name == "concat":
        input_dimension = 200
    else:
        input_dimension = 100

    sequence_input = Input(shape=(24,104))

    input_avg = Input(shape=(input_dimension, ), name = "avg")        
#     x_1 = Dense(256, activation='relu')(input_avg)
#     x_1 = Dropout(0.3)(x_1)
    
    if layer_name == "GRU":
        x = GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)

    x = keras.layers.Concatenate()([x, input_avg])

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    
    logits_regularizer = tf.keras.regularizers.L2(l2=0.01)
    
    preds = Dense(1, 
                  activation='sigmoid',
                  use_bias=False,
                  kernel_initializer=tf.keras.initializers.GlorotUniform(), 
                  kernel_regularizer=logits_regularizer
                 )(x)
    
    
    opt = Adam(lr=0.001, decay = 0.01)
    model = Model(inputs=[sequence_input, input_avg], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    
    return model


# %%
def mean(a):
    return sum(a) / len(a)


# %%
type_of_ner = "new"

x_train_lstm = pd.read_pickle("data/"+type_of_ner+"_x_train.pkl")
x_dev_lstm   = pd.read_pickle("data/"+type_of_ner+"_x_dev.pkl")
x_test_lstm  = pd.read_pickle("data/"+type_of_ner+"_x_test.pkl")

y_train = pd.read_pickle("data/"+type_of_ner+"_y_train.pkl")
y_dev   = pd.read_pickle("data/"+type_of_ner+"_y_dev.pkl")
y_test  = pd.read_pickle("data/"+type_of_ner+"_y_test.pkl")

ner_word2vec = pd.read_pickle("data/"+type_of_ner+"_ner_word2vec_limited_dict.pkl")
# ner_fasttext = pd.read_pickle("data/"+type_of_ner+"_ner_fasttext_limited_dict.pkl")
# ner_concat   = pd.read_pickle("data/"+type_of_ner+"_ner_combined_limited_dict.pkl")

train_ids = pd.read_pickle("data/"+type_of_ner+"_train_ids.pkl")
dev_ids   = pd.read_pickle("data/"+type_of_ner+"_dev_ids.pkl")
test_ids  = pd.read_pickle("data/"+type_of_ner+"_test_ids.pkl")

# %%
embedding_types = ['word2vec']#, 'fasttext', 'concat']
embedding_dict  = [ner_word2vec]#, ner_fasttext, ner_concat]
target_problems = ['mort_hosp', 'mort_icu', 'los_3', 'los_7']


num_epoch = 100
model_patience = 5
monitor_criteria = 'val_loss'
batch_size = 64
iter_num = 11
unit_sizes = [128, 256]

#layers = ["LSTM", "GRU"]
layers = ["GRU"]

for each_layer in layers:
    for each_unit_size in unit_sizes:
        for embed_dict, embed_name in zip(embedding_dict, embedding_types): 

            temp_train_ner = dict((k, ner_word2vec[k]) for k in train_ids)
            temp_dev_ner   = dict((k, ner_word2vec[k]) for k in dev_ids)
            temp_test_ner  = dict((k, ner_word2vec[k]) for k in test_ids)

            x_train_ner = create_dataset(temp_train_ner)
            x_dev_ner   = create_dataset(temp_dev_ner)
            x_test_ner  = create_dataset(temp_test_ner)

            for iteration in range(1, iter_num):
                for each_problem in target_problems: 
                    
                    print ("Layer: ", each_layer) 
                    print ("Hidden unit: ", each_unit_size) 
                    print ("Embedding: ", embed_name)
                    print ("Iteration number: ", iteration)
                    print ("Problem type: ", each_problem)
                    print ("__________________")

                    early_stopping_monitor = EarlyStopping(monitor=monitor_criteria, 
                                                           patience=model_patience)
                    
                    best_model_name = "avg-"+str(embed_name)+"-"+str(each_problem)+"-"+"best_model.hdf5"
                    
                    checkpoint = ModelCheckpoint(best_model_name, 
                                                 monitor='val_loss', 
                                                 verbose=0,
                                                 save_best_only=True, 
                                                 mode='min', 
                                                 period=1)


                    callbacks = [early_stopping_monitor, checkpoint]

                    model = avg_ner_model(each_layer, 
                                          each_unit_size, 
                                          embed_name)
                    
                    model.fit([x_train_lstm, x_train_ner], 
                              y_train[each_problem], 
                              epochs=num_epoch, 
                              verbose=0, 
                              validation_data=([x_dev_lstm, x_dev_ner], y_dev[each_problem]), 
                              callbacks=callbacks, 
                              batch_size=batch_size )

                    model.load_weights(best_model_name)

                    probs, predictions = make_prediction_multi_avg(model, [x_test_lstm, x_test_ner])
                    
                    save_scores_multi_avg(predictions, 
                                          probs, 
                                          y_test[each_problem], 
                                          embed_name, 
                                          each_problem, 
                                          iteration, 
                                          each_unit_size, 
                                          each_layer, 
                                          type_of_ner)
                    
                    reset_keras(model)
                    #del model
                    tf.keras.backend.clear_session()
                    gc.collect()
