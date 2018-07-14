import os
import pickle

import numpy as np
import pandas as pd
from keras.models import load_model

from CNLP.NLP_Base import nlp_base

np.random.seed(0)

class nlp_predict(nlp_base):
    def __init__(self, tot_class, wkdir = os.getcwd()+'/' ):
        '''
        << summary >>
        initial the class
        << inputs >>
        wkdir: working folder, i.e. where data are put
        cachedir: cache folder, i.e. where the pretrained token_dict, model are put
        << outputs >>
        None
        '''
        if (wkdir[-1]!='/') and (wkdir[-1]!='\\'):
            wkdir+='/' 
        self.wkdir = wkdir
        self.tot_class = tot_class
        
    def load_data(self, read_csv, sep='@', txt_col = 0, label_col = 1, rm_short = 0, 
        is_remove_special = True, remove_list = (), is_save_pkl = False):
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
            self.wkdir+'output/predict_data.pkl': (self.data)
        [var] 
            self.dada: the read file
        ''' 
        print('\n\n>>>>>>>>>> load_data <<<<<<<<<<')
        if is_save_pkl:
            save_pkl = self.wkdir+'output/predict_data.pkl'
        else:
            save_pkl = None

        self.data = super().load_data(read_csv, sep, txt_col, label_col, rm_short,
            is_remove_special, remove_list, False, save_pkl)

        return self.data

    def predict(self, dict_file, model_file, padding_mode = 'back', is_metric = False):
        '''
        << summary >>
        to predict the probability of fraud address 
        << inputs >>
        dict_file: file name of the pretrained dict file (must be in cache folder)
        model_file: file name of the pretrained Keras model file (must be in cache folder)
        padding_mode: how to perform text padding, 'back'/'front'
            'back' => keep first padding_size words, put zero in the back
            'front' => keep last padding_size words, put zero in the front 
        is_metric: whether the evaulate metrics based on the label of the data
        << outputs >>
        [file]
            self.wkdir+'output/predict_result.pkl': (y_pred_prob)
        [var]
            y_pred_prob: the predicted probability of addresses in self.data
        '''
        print('\n\n>>>>>>>>>> predict <<<<<<<<<<')
        if not hasattr(self, 'data'):
            self.data = pickle.load(open(self.wkdir+'output/predict_data.pkl','rb'))

        # load word dict
        model_token_dict = pickle.load(open(dict_file,'rb'))
        
        # load model
        model = load_model(model_file)
        padding_size = model.get_config()[0]['config']['batch_input_shape'][1]
        print('\npadding size of the model = %d' % padding_size)
        # create sequence 
        token_list, token_freq = self.build_freq_list(self.data[:,0])
        data_seq = self.text_to_sequence(token_list, model_token_dict, padding_size, padding_mode)

        print('===== predict results =====')
        self.y_pred_prob = model.predict(data_seq, batch_size = 512, verbose = 1)

        if is_metric:
            self.eval_model(self.data[:,1].astype(int), self.y_pred_prob, self.tot_class, binary_threshold = 0.5)

        pickle.dump(self.y_pred_prob, open(self.wkdir+'output/predict_result.pkl','wb'))

        return self.y_pred_prob

    def output(self, output_file = 'predict_result.txt'):
        '''
        << summary >>
        output a table that contains the cleaned data and predicted probability and classes
        << inputs >>
        output_file: file name of the output csv file (will be put in wkdir)
         << outputs >>
         [var]
            df: the pandas dataframe with predicted results
        [file]
            self.wkdir+'output/'+output_file
        '''
        print('\n\n>>>>>>>>>> output <<<<<<<<<<')
        if not hasattr(self, 'data'):
            self.data = pickle.load(open(self.wkdir+'output/predict_data.pkl','rb'))
        if not hasattr(self, 'y_pred_prob'):
            self.y_pred_prob = pickle.load(open(self.wkdir+'output/predict_result.pkl','rb'))

        if self.tot_class == 2:
            df = pd.DataFrame(np.hstack((self.data, self.y_pred_prob, np.round(self.y_pred_prob).astype(int))), 
                columns = ['text', 'label','pred_prob','pred_label'])
        else:
            df = pd.DataFrame(np.hstack((self.data, self.y_pred_prob, np.reshape(np.argmax(self.y_pred_prob,1),(-1,1)))), 
                columns = ['text', 'label']+['pred_prob'+str(n) for n in range(self.tot_class)]+['pred_label'])            
        if output_file is not None:
            df.to_csv(self.wkdir+'output/'+output_file, sep = '@')
        print('\noutput result: %s' % self.wkdir+'/output/'+output_file)
        return df
