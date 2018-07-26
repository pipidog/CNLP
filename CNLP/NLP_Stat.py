import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

from CNLP.NLP_Base import nlp_base


class nlp_stat(nlp_base):
    def __init__(self,wkdir = os.getcwd()+'/'):
        '''
        << summary >>
        initialize the class
        << inputs >>
        wkdir: working directory where the data are stored
        << outputs >>
        None
        '''
        super().__init__(wkdir)
        print('working folder: %s' % self.wkdir)

    def load_data(self, read_csv, sep='@', txt_col = 0, label_col = 1, rm_short = 0,
                is_remove_special = True, remove_list = ()):
        '''
        << summary >>
        load scratch data, data must be three columns: [acct_num, label, address]
        << inputs >>
        data_file: a list contains all data files, e.g. ['data1.csv', 'data2.csv']
        sep: symbol used in data file to seperate columns
        is_shuffle: whether to shuffle the data
        << outputs >>
        [file] 
            self.wkdir+'output/stat_data.pkl': (self.data)
        [var] 
            None
        ''' 
        print('\n\n>>>>>>>>>>> load_data <<<<<<<<<<<<<')
        self.data = super().load_data(read_csv, sep, txt_col, label_col, rm_short, is_remove_special, 
                    remove_list, False, self.wkdir+'/output/stat_data.pkl')

        return self.data

    def run_stat(self):
        '''
        << summary >>
        count high frequency words 
        << inputs >>
        None
        << outputs >>
        [file]
            self.wkdir+'output/stat_freq.pkl' (word_freq)
        [var]
        '''
        print('\n\n>>>>>>>>>>> run_stat <<<<<<<<<<<<<')
        if not hasattr(self,'data'):
            self.data = pickle.load(open(self.wkdir+'output/stat_data.pkl','rb'))
        token_list, token_freq = self.build_freq_list(self.data[:,0])
        pickle.dump(token_freq, open(self.wkdir+'output/stat_freq.pkl','wb'))

        return token_freq

    def vis_stat(self, n_show = 60, ex_words = (), lang = 'tc', is_show = True):
        '''
        << summary >>
        visualize high frequency words
        << input >>
        fontdir: chinese font directory to be used for matplotlib, file name must be zh-font.ttf
        n_show: top-n words to show
        ex_words: a list contains the words to be excluded in the plot
        << outputs >>
        [file]
            self.wkdir+'output/stat_freq.png', plot of the high frequency data
        [var]
            None
        ''' 
        print('\n\\n>>>>>>>>>>> vis_stat <<<<<<<<<<<<<')
        assert n_show < 500, 'max of n_show is 500'
        # set font to show chinese characters correctly
        if lang == 'tc':
            fontfile = os.path.dirname(os.path.abspath(__file__))+'/tc-font.ttf'
        elif lang == 'sc':
            fontfile = os.path.dirname(os.path.abspath(__file__))+'/sc-font.ttf'

        myfont = FontProperties(fname= fontfile)  
        matplotlib.rcParams['axes.unicode_minus']=False  

        # load data
        token_freq = pickle.load(open(self.wkdir+ 'output/stat_freq.pkl','rb'))
        top_list = token_freq.most_common(500)

        # convert ex_words from numbers to actual words if use number labels
        if len(ex_words)!=0:
            if type(ex_words[0])==type(1):
                ex_words = [top_list[n][0] for n in ex_words] 
        print('===== show high-frequency tokens ======')
        print('  excluded words: ')
        print('  ',ex_words)

        # build keys and values 
        keys = []; val = []
        print('  >> top-500 words (with exclusions)')
        for n in range(500):
            if (top_list[n][0] not in ex_words):
                keys += [top_list[n][0]+'   '] # add space to prevent wrong display
                val += [top_list[n][1]]
                print('    rank-%d: %s (%d)' % (n, keys[-1], val[-1]))
        # visuzlize frequency statiscs 
        sns.set(style="whitegrid")
        f, ax = plt.subplots(figsize=(8,8.5))
        sns.barplot(x=val[:n_show], y=keys[:n_show])
        ax.set_yticklabels(keys, fontproperties=myfont) 
        plt.title('word frequency')
        plt.savefig(self.wkdir+'output/stat_freq.png')
        if is_show:
            plt.show()