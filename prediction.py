import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from xgboost import XGBRFClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

def process_text(text):
    text = text.lower().replace('\n',' ').replace('\r','').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'[0-9]','',text)
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    text = " ".join(filtered_sentence)
    return text


def TF_IDF_ML(X,y,tokenizer):

    tokenizer.fit_on_texts(X)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    
    #print("Vocabulary Size :", vocab_size)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42)
    
    
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train),
                        maxlen = max_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test),
                       maxlen = max_len)
    return X_train, X_test, y_train, y_test
