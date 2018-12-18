# coding:utf-8
from __future__ import print_function
import os
import os.path
import json
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import sys
import time
import numpy as np
from collections import defaultdict
import pandas as pd

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv1D, GRU, LSTM
from keras.models import Model
from keras.layers import GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import utils.data_preprocessing as dp
import tensorflow as tf
from keras import backend as K
from utils.attention_layer import Attention_layer

class TextClassifier(object):
    def __init__(self, conf_path, ispredict=0):
        try:
            param = json.load(open(conf_path))
        except Exception as e:
            print("%s, parameter load ERROR!"%(conf_path))
            print(e)
            sys.exit(0)
        #  read parameters
        self.dataset_path = param["dataset_path"]  # "../data/en_community_content_clean.csv"
        self.input_dataset_path = param["input_dataset_path"]  # "../data/en_community_content.csv"
        self.weights_path = param["weights_path"]  # "../data/spam_faxttext.h5"  # model parameters save path
        self.maxlen = param["maxlen"]  # 1953  # max length of sentence
        self.batch_size = param["batch_size"]  # 128  # batch size
        self.nb_epoch = param["nb_epoch"]  # 1  # number of epoch
        self.embedding_length = param["embedding_length"]  # 300  # length of word embedding
        self.gpuid = param["gpuid"]  # 0  # GPU id for GPU model
        self.model_type = param["model_type"]  # 'CNNFastText'  # support textcnn, textrnn, attnlstm, fasttext
        self.ngram_range = param["ngram_range"]
        self.split_rate = param["split_rate"]
        self.word2id = {}
        self.words_ngram = {}
        # add 0831
        self.category2id = {}
        self.id2category = {}
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)
        if ispredict == 1:
            self.model = self.load_model()


    def preprocess(self):
        print("preprocess data...")
        df_dataset, self.nb_classes = dp.load_data_from_disk(self.input_dataset_path, self.dataset_path)

        print("generate and save words...")
        self.word2id = dp.generate_words(df_dataset)

        print("generate and save category2id...")
        self.category2id = dp.generate_category2id(df_dataset)

        self.max_features = len(self.word2id)
        # split train and etst
        # add 0831
        x_train, y_train, x_test, y_test, _ = dp.split_train_test(df_dataset,
                                                                            self.word2id,
                                                                            self.split_rate,
                                                                            self.category2id)
        # add fasttext features
        # if self.model_type == 'cnnfasttext':
        x_train, x_test, self.max_features_ngram, self.words_ngram = dp.fasttext_data(x_train, x_test,
                                                                                      self.max_features,
                                                                                      self.ngram_range)

        # padding
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        y_train = np_utils.to_categorical(np.array(y_train), num_classes=self.nb_classes)
        y_test = np_utils.to_categorical(np.array(y_test), num_classes=self.nb_classes)
        print("max_features: %d" % self.max_features)
        print("max len: %d" % self.maxlen)
        print("fasttext max_features_ngrams: %d" % self.max_features_ngram)
        print("x_train:", x_train.shape)
        print("x_test:", x_test.shape)
        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)
        return x_train, y_train, x_test, y_test

    # # FastText CNN Model
    # def create_model(self):
    #     """ CNNFastText model """
    #     max_features_ngram = self.max_features_ngram + 1  # input dims
    #     inputs = Input(shape=(self.maxlen,))
    #     embed = Embedding(max_features_ngram, self.embedding_length)(inputs)
    #     conv_3 = Conv1D(filters=256, kernel_size=3, padding="valid", activation="relu", strides=1)(embed)
    #     conv_4 = Conv1D(filters=256, kernel_size=4, padding="valid", activation="relu", strides=1)(embed)
    #     conv_5 = Conv1D(filters=256, kernel_size=5, padding="valid", activation="relu", strides=1)(embed)
    #     pool_3 = GlobalMaxPooling1D()(conv_3)
    #     pool_4 = GlobalMaxPooling1D()(conv_4)
    #     pool_5 = GlobalMaxPooling1D()(conv_5)
    #     cat = Concatenate()([pool_3, pool_4, pool_5])
    #     output = Dropout(0.25)(cat)
    #     dense1 = Dense(256, activation='relu')(output)
    #     bn = BatchNormalization()(dense1)
    #     #dense2 = Dense(self.nb_classes, activation='softmax')(bn)
    #     dense2 = Dense(self.nb_classes, activation='sigmoid')(bn)
    #     model = Model(inputs=inputs, outputs=dense2)
    #     return model


    #Fasttext AttnGRU
    def create_model(self):
        """ Attention Model with GRU model"""
        max_features_ngram = self.max_features_ngram + 1  # input dims  # input dims
        inputs = Input(shape=(self.maxlen,))
        embed = Embedding(max_features_ngram, self.embedding_length)(inputs)
        gru = GRU(256, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(embed)
        output = Attention_layer()(gru)
        dense1 = Dense(256, activation='relu')(output)
        bn = BatchNormalization()(dense1)
        dense2 = Dense(self.nb_classes, activation='softmax')(bn)
        model = Model(inputs=inputs, outputs=dense2)
        return model

    # Fasttext LSTM
    # def create_model(self):
    #     max_features_ngram = self.max_features_ngram + 1  # input dims
    #     model = Sequential()
    #     model.add(Embedding(max_features_ngram, output_dim=256))
    #     model.add(LSTM(128))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(self.nb_classes, activation='sigmoid'))
    #     return model

    # FastText 1-layer MLP-1
    # def create_model(self):
    #     """ CNNFastText model """
    #     max_features_ngram = self.max_features_ngram + 1  # input dims
    #     inputs = Input(shape=(self.maxlen,))
    #     embed = Embedding(max_features_ngram, self.embedding_length)(inputs)
    #     flatten = Flatten()(embed)
    #     #reshape = Reshape(self.embedding_length*self.maxlen,)(embed)
    #     dense = Dense(512, activation='relu')(flatten)
    #     bn = BatchNormalization()(dense)
    #     #dense = Dense(self.nb_classes, activation='softmax')(bn)
    #     dense = Dense(self.nb_classes, activation='sigmoid')(bn)
    #     model = Model(inputs=inputs, outputs=dense)
    #     return model

    #
    # # # FastText 1-layer MLP-2
    # def create_model(self):
    #     """ CNNFastText model """
    #     max_features_ngram = self.max_features_ngram + 1  # input dims
    #     model = Sequential()
    #     model.add(Embedding(max_features_ngram, self.embedding_length, input_length=self.maxlen))
    #     model.add(Flatten())
    #     model.add(Dense(512))
    #     model.add(Activation('relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(self.nb_classes))
    #     model.add(Activation('sigmoid'))
    #     return model

    # add 0831
    def evaluation(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        row_max = [np.where(x==np.max(x))[0][0] for x in y_pred]
        y_pred = np.zeros(shape=(len(row_max), self.nb_classes))
        for row, col in enumerate(row_max):
            y_pred[row, col] = 1.0
        target_names = ['class ' + str(i) for i in range(0, self.nb_classes)]
        y_test = [np.where(x==np.max(x))[0][0] for x in y_test]
        y_pred = [np.where(x==np.max(x))[0][0] for x in y_pred]
        cm = confusion_matrix(y_test, y_pred)
        print('confusion matrix: \n')
        print(cm)
        print("classification report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

    def train(self):
        """ load dataset and train model """
        x_train, y_train, x_test, y_test = self.preprocess()
        """ GPU parameter """
        with tf.device('/gpu:' + str(self.gpuid)):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
            tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                             log_device_placement=True,
                                             gpu_options=gpu_options))
            model = self.create_model()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
            print("model summary")
            model.summary()
            print("checkpoint_dir: %s" % self.weights_path)
            callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                         ModelCheckpoint(self.weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
                         ]

            print("training started...")
            tic = time.process_time()
            model.fit(x_train,
                        y_train,
                        batch_size=self.batch_size,
                        epochs=self.nb_epoch,
                        validation_data=(x_test, y_test),
                        shuffle=1,
                        callbacks= callbacks)
            toc = time.process_time()
            print("training ended...")
            print(" ----- total Computation time = " + str((toc - tic) / 60) + " mins ------ ")
            model.save_weights(self.weights_path, overwrite=True)
            self.evaluation(model, x_test, y_test)


    def load_model(self):
        print("load model...")
        self.word2id = dp.load_words(self.words_path)
        self.max_features = len(self.word2id)
        self.words_ngram = dp.load_ngram_words(self.words_ngram_path)
        self.max_features_ngram = np.max(list(self.words_ngram.values()))
        model = self.create_model()
        model.load_weights(self.weights_path)
        # add 0831
        self.category2id = dp.load_category2id(self.category2id_path)
        self.id2category = dict((v, k) for k,v in self.category2id.items())
        return model

    def predict(self, text):
        """ predict """
        text = dp.denoise_text(text)
        if not text.strip():
            return [[1.0/self.nb_classes] * self.nb_classes]
        x_test = [self.word2id.get(w, 0) for w in text.split()]
        if self.model_type == 'cnnfasttext':
            x_test = dp.predict_fasttext_data(x_test, self.words_ngram, self.ngram_range)
        x_test = sequence.pad_sequences([x_test], maxlen=self.maxlen)
        y_predicted = self.model.predict(x_test, batch_size=1)
        return y_predicted

    K.clear_session()
    tf.reset_default_graph()
