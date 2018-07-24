import os
import pickle
import sys

import numpy as np
import pandas as pd
from keras.layers import (GRU, LSTM, Bidirectional, Concatenate, Conv1D, Dense,
                          Dropout, GlobalMaxPooling1D, Input, SimpleRNN)
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical, plot_model

from CNLP.NLP_Base import nlp_base

np.random.seed(0)

class nlp_model(nlp_base):
    def __init__(self, tot_class, wkdir = os.getcwd()+'/'):
        '''
        << summary >>
        initialize the class
        << inputs >>
        wkdir: working directory where the data are stored
        << outputs >>
        None
        '''
        super().__init__(wkdir)
        assert tot_class >=2, 'tot_class must >= 2!'
        self.tot_class = tot_class

        print('working folder: %s' % self.wkdir)

    def load_data(self, read_csv, sep='@', txt_col = 0, label_col = 1, rm_short = 0, is_remove_special = True,
                    remove_list = (), is_shuffle = True):
        '''
        << summary >>
        load scratch data, data must be three columns: [acct_num, label, address]
        << inputs >>
        read_csv: a list contains all data files, e.g. ['data1.csv', 'data2.csv']
        sep: symbol used in data file to seperate columns
        txt_col: column index of the text
        label_col: column index of the label
        rm_short: remove data shorter than the assigned length 
        is_remove_special: whether to remove special symbols
        remove_list: words to be removed from the raw data
        is_shuffle: whether to shuffle the data
        << outputs >>
        [file] 
            self.wkdir+'output/stat_data.pkl': (self.data)
        [var] 
            self.dada: the read file
        ''' 
        print('\n\n>>>>>>>>>> load_data <<<<<<<<<<')
        self.data = super().load_data(read_csv, sep, txt_col, label_col, rm_short, is_remove_special, 
                    remove_list, is_shuffle, self.wkdir+'/output/model_data.pkl')

        y_set = set(self.data[:,1])

        return self.data

    def preprocessing(self, padding_size = 20, padding_mode = 'back', max_words = 20000, 
                        ex_words = (), test_split = 0.1, is_shuffle = True ):
        '''
        << summary >>
        preprocessing data for training and testing
        << inputs >>
        padding_size: padding_size of each address
        padding_mode: how to perform text padding, 'back'/'front'
            'back' => keep first padding_size words, put zero in the back
            'front' => keep last padding_size words, put zero in the front 
        max_words: max_words in the generated dictionary
        ex_words: words to be excluded from the dictionary
        test_split: portion of splitting test data
        is_shuffle: whether to shuffle data
        << outputs >>
        [var]
            x_train: sequence vectors for training
            y_train: label for training
            x_test: sequence vectors for testing
            y_test: label for testing
            token_dict: token dictionary
        [file]
            self.wkdir+'output/model_seq_data.pkl' (x_train, y_train, x_test, y_test)
            self.wkdir+'output/model_token_dict.pkl' (token_dict)
            self.wkdir+'output/model_token_dict.txt' (token_dict in txt format)
        '''
        print('\n\n>>>>>>>>>> preprocessing <<<<<<<<<<')
        if not hasattr(self,'data'):
            self.data = pickle.load(open(self.wkdir+'output/model_data.pkl','rb'))
        
        token_list, token_freq = self.build_freq_list(self.data[:,0])
        token_dict = self.build_dict(token_freq, max_words, ex_words)
        text_seq_array = self.text_to_sequence(token_list, token_dict, padding_size, padding_mode)
        
        if is_shuffle:
            pm = np.random.permutation(len(text_seq_array))
        else:
            pm = np.arange(len(text_seq_array))
        test_frac = int(test_split*len(self.data[:,1]))   
        
        x_test = text_seq_array[pm[:test_frac]]
        x_train = text_seq_array[pm[test_frac:]]
        y_test = self.data[pm[:test_frac],1]
        y_train = self.data[pm[test_frac:],1]
        if self.tot_class > 2:
            y_test = to_categorical(y_test, self.tot_class)
            y_train = to_categorical(y_train, self.tot_class)

        print('shape of x_train =', x_train.shape)
        print('shape of y_train =', y_train.shape)
        print('shape of x_test =', x_test.shape)
        print('shape of y_test =', y_test.shape)

        pickle.dump((x_train, y_train, x_test, y_test),\
                 open(self.wkdir+'output/model_seq_data.pkl','wb'))
        pickle.dump(token_dict,open(self.wkdir+'output/model_token_dict.pkl','wb'))
        
        # output full dictionary for simple look-up
        res_dict = dict(zip(token_dict.values(),token_dict.keys()))
        with open(self.wkdir+'/output/model_token_dict.txt','w') as file:
            [file.write('%i : %s\n' % (n, res_dict[n])) for n in range(1,max(res_dict.keys())+1)]
        
        # save parameters for building model 
        pickle.dump((len(token_dict), len(x_train[0,:])),
            open(self.wkdir+'/output/model_preprocessing_tmp.pkl','wb'))

        return x_train, y_train, x_test, y_test, token_dict

    def build_rnn(self, embedding_size = 128, is_bidirectional = False, depth = 3, cell = 'GRU', 
                cell_size = 128, dense_size = 20, dr = 0.4):
        '''
        << summary >>
        build keras model
        << inputs >>
        embedding_size: dimension of the embedding layer
        is_bidirectional: whether the model is bidirectional
        depth: depth of the RNN neural network
        cell: cell of the RNN neuron, 'SimpleRNN'/'GRU'/'LSTM'
        cell_size: number of neurons of each cell
        dense_size: size of the final fully-connected layer
        dr: dropout rate for RNN and the final fully-connected layer
        << outputs >>
        [file]:
            self.wkdir+'/output/model.h5': the model file
        [var]:
            model: the keras model object
        '''
        print('\n\n>>>>>>>>>> build RNN model <<<<<<<<<<')
        # load token_dict_size and padding_size
        token_dict_size, padding_size = \
            pickle.load(open(self.wkdir+'/output/model_preprocessing_tmp.pkl','rb'))

        # define layer wrapper
        layer_wrap = []
        for n in range(depth):
            if n == depth-1:
                return_sequences = False
            else:
                return_sequences = True

            if cell == 'Simple':
                layer_tmp = SimpleRNN(cell_size, dropout = dr, recurrent_dropout = dr, return_sequences = return_sequences)
            elif cell == 'LSTM':
                layer_tmp = LSTM(cell_size, dropout = dr, recurrent_dropout = dr, return_sequences = return_sequences)
            elif cell == 'GRU':
                layer_tmp = GRU(cell_size, dropout = dr, recurrent_dropout = dr, return_sequences = return_sequences)
            if is_bidirectional:
                layer_tmp = Bidirectional(layer_tmp)
            layer_wrap.append(layer_tmp)

        # construct model
        model = Sequential()
        model.add(Embedding(token_dict_size+1, embedding_size, input_length = padding_size))
        [model.add(layer_wrap[n]) for n in range(depth)]
        model.add(Dense(dense_size, activation='relu'))

        if self.tot_class == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(self.tot_class, activation='softmax'))     
        print(model.summary())   
        try:
            import pydot
        except:
            print('\n ==> plot_model is not available, model will not output in png format\n')
        else:
            print('\n ==> model has been output to '+self.wkdir+'output/model.png\n')
            plot_model(model, self.wkdir+'output/model.png', show_shapes = True)

        model.save(self.wkdir+'/output/model.h5')

        return model

    def build_cnn(self, embedding_size = 128, n_gram = [2,3,4], filters = 32, dr = 0.5):
        print('\n\n>>>>>>>>>> build CNN model <<<<<<<<<<')
        # load token_dict_size and padding_size
        token_dict_size, padding_size = \
            pickle.load(open(self.wkdir+'/output/model_preprocessing_tmp.pkl','rb'))

        # (batch_size, padding_size)
        x_in = Input(shape=(padding_size,))
        # (batch_size, padding_size, embedding_size)
        embedding = Embedding(token_dict_size+1, embedding_size)(x_in)
        pooling_out = []
        for n in n_gram:
            # (batch_size, padding_size, filters)
            conv = Conv1D(filters = filters, kernel_size=n, strides=1, padding='same', activation='relu')(embedding)
            # (batch_size, padding_size, filters)
            drop = Dropout(dr)(conv)
            # (batch_size, filters)
            pooling = GlobalMaxPooling1D()(drop)
            # append all pooling outputs
            pooling_out.append(pooling)
        # (batch_size, filters*len(n_gram))
        merged = Concatenate()(pooling_out)
        drop = Dropout(dr)(merged)

        if self.tot_class ==2:
            x_out = Dense(1, activation = 'sigmoid')(drop)
        else:
            x_out = Dense(self.tot_class, activation = 'softmax')(drop)

        model = Model(inputs = x_in, outputs = x_out)
        print(model.summary()) 
        
        try:
            import pydot
        except:
            print('\n ==> plot_model is not available, model will not output in png format\n')
        else:
            print('\n ==> model has been output to '+self.wkdir+'output/model.png\n')
            plot_model(model, self.wkdir+'output/model.png', show_shapes = True)

        model.save(self.wkdir+'/output/model.h5')

        return model

    def train(self, optimizer = 'RMSprop', lr = 0.001, batch_size = 128, epochs = 10, validation_split = 0.1, verbose = 1):
        '''
        << summary >>
        train the model
        << inputs >>
        optimizer: the optimizer, 'RMSprop'/'SGD'/'Adam'
        batch_size: training batch size
        epochs: number of training epochs
        validation_split: portion of total data for validation
        verbose: 1: silent, 2: progress bar, 3: one line per epoch
        << outputs >>
        [var]
            history
        [file]
            self.wkdir+'/output/model.h5': keras model object
        '''
        print('\n\n>>>>>>>>>> train <<<<<<<<<<')
        # load preprocessing data
        x_train, y_train, x_test, y_test = \
            pickle.load(open(self.wkdir+'/output/model_seq_data.pkl','rb'))
        token_dict = pickle.load(open(self.wkdir+'/output/model_token_dict.pkl','rb'))
        
        # define optimizer
        if optimizer == 'Adam':
            optimizer = Adam(lr)
        elif optimizer == 'RMSprop':
            optimizer = RMSprop(lr)
        elif optimizer == 'SGD':
            optimizer = SGD(lr)

        print(' ***** reloading model *****')
        model = load_model(self.wkdir+'output/model.h5')
        print(model.summary())
        if self.tot_class == 2:
            model.compile(loss='binary_crossentropy', optimizer=optimizer, 
            metrics=['binary_accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
            metrics=['categorical_accuracy'])  
        history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 
                            verbose = verbose, validation_split = validation_split)            
        model.save(self.wkdir+'/output/model.h5')

        return history
        
    def test(self, binary_threshold = 0.5):
        '''
        << summary >>
        test performance of the model 
        << inputs >>
        binary_threshold: threshold for binary classification, only use when total_class = 2
        << outputs >>
        metric: the metrics
        '''
        print('\n\n>>>>>>>>>> test <<<<<<<<<<')
        x_train, y_train, x_test, y_test = \
            pickle.load(open(self.wkdir+'output/model_seq_data.pkl','rb'))

        model = load_model(self.wkdir+'output/model.h5')
        y_pred_prob = model.predict(x_test, verbose = 1)
        metric = self.eval_model(y_test.astype(int), y_pred_prob, self.tot_class, binary_threshold)

        return metric
