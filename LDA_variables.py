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

os.chdir("C:/Users/Ruben/PycharmProjects/untitled1")
# print(os.getcwd())

file = open('./all_Xb_Nb_Load1', 'rb')

all_Xb_Nb = pickle.load(file)

file.close()

all_data = []

for filename in os.listdir("./d84082_pages"):
    if filename.endswith(".txt"):
        all_data.append(pd.read_csv('./parsed_pages/' + str(filename), sep = "\t", engine = 'python', header = None, names = ['index', 'review']))

# print(all_data[0])

formatted_data = []
docs_lda = [" ".join(list(all_data[0]['review']))]

for hotel in all_data[1:]:
    docs_lda.append(" ".join(list(hotel['review'])))

docs_lda = pd.DataFrame(docs_lda)
docs_lda.columns = ['review']

# print(docs_lda.review)
# print(type(docs_lda.review))

vec = CountVectorizer()
transfer_vec = vec.fit_transform(docs_lda.review)
dtm = pd.DataFrame(transfer_vec.toarray(), columns=vec.get_feature_names())
dtm.index = docs_lda.index

tdm = dtm.transpose()

corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(tdm))

id2word = dict((v, k) for k, v in vec.vocabulary_.items())

np.random.seed(123)
lda = models.LdaModel(corpus=corpus, num_topics = 40, id2word = id2word, passes = 10)

corpus_transformed = lda[corpus]

topic_distributions = []

dropped_documents = []

for i in range(0,len(all_data)):
    corpus_single_hotel = corpus[i]
    vector = lda[corpus_single_hotel]
    topic_distributions.append(vector)

test = []

print(len(topic_distributions))

for i in range(0,len(topic_distributions)):

    topics_present = [item[0] for item in topic_distributions[i]]
    topic_prevalence = [item[1] for item in topic_distributions[i]]

    count = 0
    test_array = np.zeros(40)
    for topic in topics_present:
        test_array[topic] = topic_prevalence[count]
        count += 1

    test.append(test_array)


file = open('LDA_arrays_Load1', 'wb')

pickle.dump(test, file)

file.close()




