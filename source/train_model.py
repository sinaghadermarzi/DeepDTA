#!/usr/bin/env python
from __future__ import print_function
import sys
import os
from lifelines.utils import concordance_index



import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn

### We modified Pahikkala et al. (2014) source code for cross-val process ###

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
tf.compat.v1.set_random_seed(0)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)

from datahelper import *
from itertools import product
from gen_arguments import argparser, logging
import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input, Embedding,Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
import time




if len(sys.argv) != 2:
    print("\nUsage: train_model.py <path to training csv> \n")
    exit(1)

train_csv_path = sys.argv[1]




def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII 
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput) 
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    # plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel


def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select




def gridsearch_cv(cv_dataset, fold_inds, runmethod, prfmeasure, conf):
# def general_nfold_cv(XD, XT, Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets,
#                      val_sets):  ## BURAYA DA FLAGS LAZIM????
    paramset1 = conf.num_windows  # [32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = conf.smi_window_lengths  # [4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = conf.seq_window_lengths  # [8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = conf.num_epoch  # 100
    batchsz = conf.batch_size  # 256

    logging("---Parameter Search-----", conf)

    n_folds = len(fold_inds)
    grid_size = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(n_folds)] for y in range(grid_size)]
    all_losses = [[0 for x in range(n_folds)] for y in range(grid_size)]
    print(all_predictions)
    fold_list = list(range(n_folds))
    for i in range(n_folds):
        val_inds  = fold_inds[i]
        val_X_d , val_X_t, val_Y, _ , _ = cv_dataset.get_objects(val_inds)
        train_folds = fold_list.copy()
        train_folds.remove(i)
        train_inds = []
        for j in train_folds:
            train_inds += fold_inds[j]

        train_X_d, train_X_t, train_Y, _, _ = cv_dataset.get_objects(val_inds)
        pointer = 0

        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)):  # learning rate
                param2value = paramset2[param2ind]
                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]
                    # prepares the neural network and layers and stuff
                    gridmodel = runmethod(conf, param1value, param2value, param3value)
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
                    gridres = gridmodel.fit(([np.array(train_X_d), np.array(train_X_t)]), train_Y,
                                            batch_size=batchsz, epochs=epoch,
                                            validation_data=(
                                            ([np.array(val_X_d), np.array(val_X_t)]), val_Y),
                                            shuffle=False, callbacks=[es])

                    predicted_labels = gridmodel.predict([np.array(val_X_d), np.array(val_X_t)])
                    loss, rperf2 = gridmodel.evaluate(([np.array(val_X_d), np.array(val_X_t)]), np.array(val_Y),
                                                      verbose=0)
                    rperf = prfmeasure(list(val_Y), list(predicted_labels))

                    logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" %
                            (param1ind, param2ind, param3ind, i, rperf, rperf2, loss), conf)


                    all_predictions[pointer][
                        i] = rperf  # TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][i] = loss

                    pointer += 1

    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):
            for param3ind in range(len(paramset3)):

                avgperf = 0.
                for foldind in range(n_folds):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= n_folds
                # print(epoch, batchsz, avgperf)
                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1ind, param2ind, param3ind]

                pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses

from config import get_config
from dataset import dta_dataset
import random



if __name__=="__main__":
    conf = get_config()
    conf.log_dir = conf.log_dir + str(time.time()) + "/"

    if not os.path.exists(conf.log_dir):
        os.makedirs(conf.log_dir)

    logging(str(conf), conf)
    # run_regression(conf)

    ## load training set
    train_dataset = dta_dataset(conf, train_csv_path)

    ## generate cross validation folds
    n_folds = 5
    n_train = train_dataset.get_num_pairs()
    inds = list(range(n_train))
    random.shuffle(inds)
    fold_size= int(n_train/n_folds)
    fold_inds = [inds[i*fold_size:(i+1)*fold_size] for i in range(n_folds-1)]
    fold_inds.append(inds[(n_folds-1)*fold_size:])
    ## do grid search
    perfmeasure = concordance_index
    bestpointer, best_param_inds, bestperf, allperf, all_losses = gridsearch_cv(train_dataset , fold_inds ,build_combined_categorical , perfmeasure ,conf)

    best_params = [conf.num_windows[best_param_inds[0]], conf.smi_window_lengths[best_param_inds[1]], conf.seq_window_lengths[best_param_inds[2]]]

    [p1, p2 ,p3] =  best_params

    model = build_combined_categorical(conf, p1, p2, p3)

    train_inds = []
    for i in range (n_folds-1):
        train_inds+= fold_inds[i]
    val_inds = fold_inds[n_folds-1]


    train_X_d , train_X_t, train_Y, _, _ = train_dataset.get_objects(train_inds)
    val_X_d , val_X_t, val_Y, _, _ = train_dataset.get_objects(val_inds)

    epoch = conf.num_epoch  # 100
    batchsz = conf.batch_size  # 256
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    main_train_res = model.fit(([np.array(train_X_d), np.array(train_X_t)]), train_Y,
                            batch_size=batchsz, epochs=epoch,
                            validation_data=(
                            ([np.array(val_X_d), np.array(val_X_t)]), val_Y),
                            shuffle=False, callbacks=[es])



    model.save_weights("saved_model/model_weights")
    with open("saved_model/model_hyperparams.txt", "w") as parf:
        parf.writelines("\t".join([str(x) for x in best_params]))



