import os
import pickle
import re

import jieba
import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from tqdm import tqdm

np.random.seed(0)

# main =========================

class nlp_base:   
    def __init__(self, wkdir):
        if (wkdir[-1]!='/') and (wkdir[-1]!='\\'):
            wkdir+='/' 
        if not os.path.isdir(wkdir+'output/'):
            print('./output created to store all output files')
            os.makedirs(wkdir+'output')
        self.wkdir = wkdir

    def text_clean(self, data, is_remove_special = True, remove_list = ()):
        '''
        << summary >>
        clean text data
        << inputs >>
        data: the text data, arranged as a list, each element is an article
        is_remove_special: whether to remove special chacters, True / False
        remove_list: additional characters to be removed ['我', 'wow']
        '''
        print('\n===== text cleaning =====')
        text_remove = []
        if is_remove_special:
            text_remove = [r'\s',r'\n',r'\t',r"[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）①②③④)]+"]

        text_remove += list(remove_list)
        for n in tqdm(range(len(data))):
            for _ in text_remove:
                data[n] = re.sub(_,'', data[n])    
        return data

    def load_data(self, read_csv, sep='@', txt_col = 0, label_col = 1, rm_short = 0, 
        is_remove_special = True, remove_list = (), is_shuffle = True, save_pkl = None):
        '''
        read_csv: file name of the csv in the wkdir, data is assumed to be arranged 
                  as two columns [text, label]. if not, assign the txt_col and label col. 
        sep: separation symbol in the csv file
        txt_col: column index of the text part in the csv file
        label_col: column index of the label part in the csv file
        rm_short: remove text data if shorter than rm_short
        is_remove_special: whether to remove special symbols
        is_shuffle: whether to shuffle the data
        save_pkl: file name of a pkl file that stores all read data. 
                  will be saved in the wkdir for future read
        '''
        if type(read_csv) == type('a'):
            read_csv = [read_csv]
        for n, f in enumerate(read_csv):
            if (f.find('/') == -1) and (f.find('\\') == -1):
                # relative path
                print('\n >>> reaing file: %s%s\n' % (self.wkdir,f))
                tmp = pd.read_csv(self.wkdir+f, sep = sep, engine = 'python', 
                    error_bad_lines=False, warn_bad_lines=True)
            else:
                # absolute path 
                print('\n >>> reaing file: %s\n' % f)
                tmp = pd.read_csv(f, sep = sep, engine = 'python', 
                    error_bad_lines=False, warn_bad_lines=True)
            # remove unresonable short address
            tmp = tmp[tmp.iloc[:,txt_col].apply(str).map(len) > rm_short]
            if n == 0:
                data = np.array(tmp)
            else:
                data = np.vstack((data, tmp))
        data = data[:,[txt_col,label_col]]
        assert len(data) >= 1, 'no data loaded after removing unresonable (too short / not-fomatted) addresses'
        print('\nshape of data = ', data.shape)
        data[:,0] = self.text_clean(data[:,0], is_remove_special, remove_list)
        
        # shuffle good and bad users
        if is_shuffle:
            print('all data are shuffled as requested per the user!')
            np.random.shuffle(data)
        
        if save_pkl is not None:
            pickle.dump(data, open(save_pkl,'wb'))

        return data

    def tokenize(self, text, add_word = (), del_word = ()):
        '''
        << summary >>
        to tokenize a given text
        << input >>
        text: the text to be tokenized, e.g. "今天天氣很好"
        add_word: add new words to the dictionary, e.g. "三洨"
        del_word: delete existing words from the dictionary, e.g "天氣”
        << output >>
        word_list: the tokenized text as a list, e.g. ['今天',' 天氣','很好']
        '''
        [jieba.add_word(txt) for txt in add_word]
        [jieba.del_word(txt) for txt in del_word]
        word_list = list(jieba.cut(text, cut_all = False))

        return word_list

    def build_freq_list(self, text_list):
        '''
        << summary >>
        tokenize all texts in a list
        << input >>
        text_list: the list contains all the texts (articles) to be tokenized.  
        << output >>
        token_list: a list where all the textes are tokeninzed, e.g. ['今天','天氣','很好‘]
        token_freq: the object returned by nltk.probability.FreqDist 
        '''
        token_list = []
        token_all = []
        print('\n===== tokenize whole dataset =====')
        self.tokenize('test') # just to make the output before tqdm
        for n in tqdm(range(len(text_list))):
            tmp = self.tokenize(text_list[n])
            token_all += tmp
            token_list += [tmp]
        token_freq = FreqDist(token_all)
        token_count = list(map(lambda x: len(x), token_list))
        print('\ntotal tokens in the dataset: %d' % len(set(token_all)))
        print('max token = %d, min token = %d\n' % (max(token_count), min(token_count)))

        return token_list, token_freq
    
    def build_dict(self, token_freq, max_words, ex_words):
        '''
        << summary >>
        build a dictionary to label all the words using their sequence indices
        << input >>
        token_freq: the output from build_freq_list
        max_words: max number of words in the dictionary
        ex_words: words to be exclude from the dictionary
        << output >>
        word_dict: the dictionary of selected words and its corresponding index
                   e.g. {'今天':1, '天氣':2}
        '''
        word_dict={}
        for n, word in enumerate(token_freq.most_common(max_words)):
            if word not in ex_words:
                word_dict[word[0]] = n+1 # leave 0 for word padding):


        return word_dict
    
    def text_to_sequence(self, token_list, word_dict, padding_size, padding_mode):
        '''
        << summary >>
        convert text to sequence index
        << input >>
        token_list: the token_list output via build_freq_list 
        word_dict: the output from build_dict
        padding_size: padding_size of each text in the text_list
        padding_mode: how to perform text padding, 'back'/'front'
            'back' => keep first padding_size words, put zero in the back
            'front' => keep last padding_size words, put zero in the front 
        << ouput >>
        text_seq_array: the text sequence array, np.array, tot_sample x padding_size 
        '''

        print('===== begin text to sequence =====')
        for n in tqdm(range(len(token_list))):
            # label all words by its order in the dict
            tmp = [ word_dict[word] for word in token_list[n] if word in word_dict]
            if padding_mode == 'back':
                # add 0 in its back if text is shorder than padding_size
                tmp = tmp[:min(len(tmp),padding_size)] + [0]*(padding_size-min(len(tmp),padding_size))
                token_list[n]= tmp[:padding_size]
            elif padding_mode == 'front':
                tmp = [0]*(padding_size-min(len(tmp),padding_size)) + tmp[-min(len(tmp),padding_size):] 
                token_list[n]= tmp[-padding_size:]

        text_seq_array = np.array(token_list).astype(int)  

        return text_seq_array 

    def eval_model(self, y_true, y_pred_prob, tot_class, binary_threshold = 0.5):
        '''
        << symmary >>
        evaluate the performance of your model
        << inputs >>
        y_true: true class of the each sample. For binary problem, it is a 1D-array
                for multiclass problem, it should be coded in one-hot format
        y_pred_prob: predicted probability of each class. For binary problem, it's
                a 1D-array, for multiclass problem, it should tot_sample x tot_class. 
        '''
        if tot_class == 2:
            y_pred_class = np.round(y_pred_prob-binary_threshold+0.5)
        else:
            y_true = np.argmax(y_pred_prob,1)
            y_pred_class = np.argmax(y_pred_prob,1)

        metric = {}
        metric['total_sample'] = len(y_true)
        metric['confusion_matrix'] = confusion_matrix(y_true, y_pred_class)

        if tot_class == 2:
            metric['recall_score']  = recall_score(y_true, y_pred_class, average = 'binary')
            metric['precision_score'] = precision_score(y_true, y_pred_class, average = 'binary')
            metric['auc_score'] = roc_auc_score(y_true, y_pred_prob)
        else:
            metric['recall_score']  = recall_score(y_true, y_pred_class, average = 'micro')
            metric['precision_score'] = precision_score(y_true, y_pred_class, average = 'micro')            

        print('\n***** model metrics *****\n')
        for key in metric.keys():
            print('%s:' % key)
            print(' ',metric[key])

        return metric 
