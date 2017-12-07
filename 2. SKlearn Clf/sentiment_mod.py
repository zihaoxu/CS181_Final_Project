import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from collections import defaultdict



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def predict(self, features):
        pred_list = []
        for c in self._classifiers:
            pred_list.append(c.predict(features))
        
        res = []
        for i in range(len(pred_list[0])):
            a = pred_list[0][i]
            b = pred_list[1][i]
            c = pred_list[2][i]
            if (a+b+c)< 2:
                res.append(0)
            else:
                res.append(1)
        return res

    def confidence(self, features):
        pred_list = []
        for c in self._classifiers:
            pred_list.append(c.predict(features))
        
        votes = pred_list[0][0] + pred_list[1][0] + pred_list[2][0]
        if votes< 2:
            return 1-votes/3
        else:
            return votes/3


# documents_f = open("pickled_algos/documents.pickle", "rb")
# documents = pickle.load(documents_f)
# documents_f.close()




# word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
# word_features = pickle.load(word_features5k_f)
# word_features5k_f.close()


# ps = PorterStemmer()

# def find_features(document):
#     words = set(document.split())
#     features = {}
#     for w in words:
#         w = w.lower().replace('.', '').replace(',', '').replace('!', '')
#         w = ps.stem(w)
#         features[w] = (w in word_features)
#     return features
open_file = open("pickled_algos/vect.pickle", "rb")
vect = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/clf_LogisticRegression.pickle", "rb")
clf_LogisticRegression = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/clf_LinearSVC.pickle", "rb")
clf_LinearSVC = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/clf_SGDClassifier.pickle", "rb")
clf_SGDClassifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(clf_LogisticRegression,
                                  clf_LinearSVC, 
                                  clf_SGDClassifier) 




def sentiment(text):
    return (voted_classifier.predict(vect.transform(text))[0], voted_classifier.confidence(vect.transform(text)))











