import spacy
import os
import re
import pandas as pd
from spacy.symbols import *
import numpy as np
import pandas as pd
import spacy
from stemming.porter2 import stem
import os
import shorttext
import re
import csv
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from gensim import matutils, models
import scipy.sparse
from nltk import downloader
import nltk.downloader; nltk.download('stopwords')
import gensim.corpora as corpora
import random
from nltk import word_tokenize, pos_tag
from spacy import displacy
from nltk.corpus import stopwords
from pywsd.lesk import adapted_lesk
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet

os.chdir("C:/Users/Ruben/PycharmProjects/untitled1")
print(os.getcwd())
nlp = spacy.load('en_core_web_sm')


all_sentences = []
all_all_sentences = []
for filename in os.listdir("./d84082_pages"):



    if filename.endswith(".txt") and filename in ['d84084.txt', 'd84083.txt', 'd84118.txt', 'd84101.txt']:
        print("Starting File:", filename)
        filename = "./parsed_pages/" + str(filename)

        reviews = []
        sentences = []

        with open(filename, encoding="utf8") as f:
            for line in f:
                x, y = line.strip().split('\t ')
                y = y.lower()
                reviews.append(y.replace('<br/>', ' ')) 

        for x in reviews:
            y = nlp(x).sents
            for z in y:
                sentences.append(z)

        all_sentences.append(sentences)


all_selected_sentences = []

random.seed(123)
for x in all_sentences:
    x = np.array(x)
    print(x, len(x))
    random_list = random.sample(range(1, len(x)), 20)
    all_selected_sentences.append(x[random_list])


with open("all_selected_sentences_84083_random.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences[0])

with open("all_selected_sentences_84084_random.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences[1])

with open("all_selected_sentences_84101_random.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences[2])

with open("all_selected_sentences_84118_random.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences[3])
