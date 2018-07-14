# 歡迎來到Project CNLP
CNLP是一個基於Python以及深度學習來完成文本分類以及情緒預測的中文自然語言處理包（同時支援繁體與簡體）。 它能夠幫助使用者使用極少的命令(3~5行)就快速的完成高頻詞統計，建構深度學習模型，以及預測未知數據。這個包總共含有三個主要的功能：   
* 高頻詞統計(NLP_Stat)：這個模組可以幫助使用者快速的載入數據，清洗文本資料，進行中文分詞，並進行高頻詞統計。並以將高頻詞以圖表方式呈現，幫助使用者快速的對文本數據的特性有個大致的了解。
* 深度學習建模型(NLP_Model)：這個模組能夠幫助使用者透過簡單的輸入幾個參數就快速進行數據切割以及轉換成相應的輸入向量。此模組還能夠幫助使用者快速的建構一個深度學習RNN模型。目前支援多種典型的RNN模型，包括單向以及雙向的RNN。另外模型支援多種RNN cell，包括Simple RNN, GRU,以及 LSTM。在模型建構完成後亦可輕易地調用多種優化器(SGD, PRMprop, Adam)進行模型訓練以及測試。
* 未知數據預測(NLP_Pred): 這個模組能夠幫助使用者快速的調用已經訓練好的深度學習模型，並使用該模型對未知的數據進行數據清洗以及結果預測。

# 背景介紹
文本分類以及情緒分析是自然語言處理以及深度學習中常見的基本問題。目前已經有多種針對英文文本數據的自然語言處理工具包，但是對於中文的工具包仍然十分缺乏。因此建構了這個簡單的工具包希望能方便使用者快速的搭建模型做文本分析。要強調的是，本工具包的目的並不在提供強大的擴展性，而是希望在不失基本的彈性的情況下，讓使用者盡可能使用少量的命令來處理大多數的文本分類問題。    

中文數據和英文數據很大的不同在於切詞(tokenize)，因為英文裡面，每個單字會以一個空格做區隔，但是在中文裡面，中文詞彙之間並沒有類似的區隔符，這導致了要把中文詞彙做出正確的分割十分困難。為了解決這個問題，在CNLP中我們使用了jieba作為中文切詞引擎(https://github.com/fxsjy/jieba)。jieba是一個相當優秀的python中文切詞工具，除了能自動分詞之外，使用者也可依據自己的需求來增加或刪除詞彙，並調整詞頻高低。在CNLP中，你能夠使用簡單的函數就調用jieba處理大多數的中文文本場景。

在深度學習方面，CNLP能夠對文本數據進行預處理，將文本數據轉換成訓練所需的向量形式，並使用了Keras以及Tensorflow來進行深度學習。裡面已經預先架構好了一個能適用大多數場景的深度學習RNN模型，使用者能夠依據自己的需求來調整模型的方向性(單向或雙向RNN)，模型的深度，以及相關的超參數等。在訓練完畢之後，CNLP亦可輕易的調用已經訓練好的深度學習模型來對未知的數據進行預測。

此外，由於中文自然語言處理的數據極其缺乏，在CNLP裡面也整理了兩個數據集供使用者測試。第一個數據集是來自攜程網的酒店評論數據，這個數據量偏少，正負評各約一千則（如果有更大的中文情緒分析數據集也請來信告知，讓我加入其中），作為深度學習之用太小了，主要是供大家練習以及測試CNLP之用。第二個數據集是來自北京清華大學的THUCNews，這個數據集包含十四個類別，數十萬則新聞，整個數據超過1.5G，太大了，不適合做為練習測試之用，因此我們採用了網路上提供的另一個精簡化版本(https://github.com/gaussic/text-classification-cnn-rnn)，在這個簡化版本中，共用十個類別，共50000條訓練數據，5000條驗證數據，以及10000條測試數據。以上兩個數據集已經完成了基本的數據清洗，並且做好了相應的標籤。使用者可以在程式裡面直接使用不用再做任何處理。此外為了滿足不同中文使用者的需求，兩個數據集都提供了簡體以及繁體中文的版本，方便不同的使用者使用。

考量到使用者可能會有自己的需求，因此CNLP的設計是，每一步執行完後都可以把結果儲存成pickle檔，以供日後調用。因此除了使用CNLP來進行完整的分析外，使用者亦可使用CNLP來進行部分的預處理，搭配自己的程式使用，增加了CNLP的彈性。

# Welcome to Project CNLP
 CNLP is a python-based deep learning natural language processing toolkit for Chinese text classification and sentiment analysis (both Traditional and Simplified Chinese). It helps users use only few commands (usually 3~5 lines) to finish a job such as analysis of high-frequency words, building deep learning models, and prediction of unlabeled data. CNLP consists of three pars:   
 * count high-frequency words (NLP_stat): this module helps users quick load, clean and tokenize your Chinese text data. Then perform word counting on the whole dataset with beautiful word frequency plot, so the users can quickly have an impression of the data. 
 * build deep learning models: this module helps user preprocessing (tokenize, convert to word vectors, split to training and test dataset) the text data, build deep learning RNN models and test the performance of the model. CNLP has a pre-designed RNN framework that should be compatiable with most scenarios. Users can still fine-tuned several hyper-parameters such as depth of the RNN, cell of the RNN (Simple RNN, LSTM, GRU), direction of the RNN (unidirectional, bidirectional). Once the model has been built, CNLP can train the model using various optimizers (e.g., SGD, RPMprop, Adam) and evaulate its performance automatically. 
 * prediction of unlabeled data: This module helps users to quickly analyze unlabeled data using the pretrained model. 

 # Background
 Text classification and sentiment analysis are very common problems in the field of Natural Language Processing and deep learning. There are already tons of libraries that helps users quickly do the jobs. However, such library for Chinese NLP remains lacking. To fill the gap is the purpose of this project. This project doesn't intend to provide a scalable library for general purpose. Instead, this project aims to help users use as less commands as possible to do text classification and sentiment analysis for Chinese text data without loss of basic scalability.

 A very difference between English and Chinese text data is "tokenize". In English, each word is separated by a space, so tokenize a English text is straightforward. However, there is no such structure in Chinese. Therefore, how to correctly tokenize a Chinese text become a difficult issue. To this end, we introduce jieba (https://github.com/fxsjy/jieba) as our tokenization engine. This is an excellent open-source tokenization tool with flexible features such as add words, delete words, change word rank, etc. In CNLP, you can control jieba thruogh a few inputs to deal with most Chinese texts. 

 As for deep learning, CNLP can help usrs quickly preprocessing the data and use Keras and Tensorflow as backends to perform deep learning calculations. Users can build and train the model with just a fews inputs.

 Due to the lack of Chinese text dataset, CNLP prepered two dataset for users to test. The Hotel review dataset is from Ctrip. It contains about 1000 positive comments and 1000 negative comments. This is certainly too small for a deep learning model. Therefore the purpose of this dataset is just to let users know how to perform sentiment analysis using CNLP (if you have better Chinese dataset, please let me know). The other dataset is New10. This dataset was provided by THUCNews (https://github.com/thunlp/THUCTC). The original dataset contains 740,000 news with 14 classes which is too large for testing. Therefore, CNLP provides a rather small version (https://github.com/gaussic/text-classification-cnn-rnn). This version has 50000 samples for training, 5000 samples for validation, and 10000 samples for testing. All the data are cleaned and propely labled. Users can directly use them without any further processing. Both dataset have Simplified Chinese version and Traditional Chinese version to satisify the need of different users.   

 CNLP allows users to store the results in Pickle format for every processing step, so users can reload the results for other calculations which makes CNLP not only a toolkit for text classification but also for data preprocessing.  

 # Requirement:
 To use CNLP, please make sure you have the follow python package installed:  
 * numpy, matplotlib, seaborn, pandas, sklearn, nltk (should have inculded in Anaconda). 
 * jieba, tensorflow, keras (not inculded in Anaconda). 

 # Instillation:
 Download the project, unzip it, add /CNLP to your python path.     

 # Examples:
 Check /CNLP/EXample for have a quick access.

 # Usage:
 * High-Frequency:

 * build deep learning model:

 * predict unlabel data:
