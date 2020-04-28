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


nlp = spacy.load('en_core_web_sm')



os.chdir("C:/Users/Ruben/PycharmProjects/untitled1")
print(os.getcwd())

all_Xb_Nb = []

all_sentences = []

for filename in os.listdir("./d84082_pages"):



    if filename.endswith(".txt") and filename in ['d84084.txt', 'd217767.txt', 'd84068.txt', 'd84067.txt', 'd225100.txt']:
        print("Starting File:", filename)
        filename = "./parsed_pages/" + str(filename)

        reviews = []
        sentences = []

        # Import code copied from original author TipSelector
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
        sentences_tagged = []
        sentence = 1



        # Add POS tags to all the sentences and extracting the dependencies
        verbs = set()
        children = []

        # spacy.displacy.serve(sentences[25], style='dep')
        for x in sentences:
            word = 0
            sentence_data = []
            for token in x:
                synset = adapted_lesk(str(x), token.text)
                synset = str(synset)

                if synset != "None":
                    token_synset = synset.split('(', 1)[1].split(')')[0]
                    token_synset = token_synset[1: -1]
                else:
                    token_synset = "None"
                sentence_data.append([token.text, token.pos_,token.dep_, sentence, token.i, token.head.i, token.head, token.lemma_, token_synset])
                word = word + 1

            sentences_tagged.append(pd.DataFrame(sentence_data, columns = ["token.text", "pos_tag", "dep_tag", "sentence number", "token.i", "token.head.i", "token.head", "token.lemma", "token_synset"]))
            sentence = sentence + 1

            if sentence % 500 == 0:
                print(sentence)

        # all_sentences.append(sentences_tagged)

        print("All sentences are tagged")

        # Het extracten van de tokens
        Nb_word = []
        Nb_frequency = []
        Xb = []

        # Token extraction

        # singleton nouns
        sentence = 1
        for x in sentences_tagged:
            c = 0
            for y in x['pos_tag']:
                if y == "NOUN":
                    Xb.append([x["token.lemma"][c], sentence, "NOUN"])
                c = c + 1
            sentence = sentence + 1

        print("Singleton Nouns")

        # negator POS-gram
        sentence = 1
        list_modifiers = ['amod', 'npmod', "acl", 'advcl', "advmod", "appos", "meta", "neg", "nn", "noumod", "npmod",
                           "nummod", "poss", "quantmod", "relcl"]
        for x in sentences_tagged:
            c = 0
            start_index = x["token.i"][0]
            for y in x['dep_tag']:

                if y == "neg":

                    negator_index = x['token.i'][c] - start_index
                    head_index = x['token.head.i'][c] - start_index

                    third_index = (x["token.head.i"][head_index]) - start_index  # head of head

                    indices = [negator_index, head_index]

                    if x['dep_tag'][head_index] in list_modifiers:
                        third_word = x['token.head'][head_index]
                        indices.append(third_index)

                    indices.sort()

                    negator_pos_token = ""
                    neg_synset = []
                    sentiment = []

                    for z in indices:
                        if negator_pos_token == "":
                            negator_pos_token = x["token.lemma"][z]
                            neg_synset = [x["token_synset"][z]]


                        else:
                            negator_pos_token = negator_pos_token + " " + x['token.lemma'][z]
                            neg_synset.append(x["token_synset"][z])

                    for syn in neg_synset:
                        if syn == "None":
                            sentiment = "None"
                        else:
                             sentiment = sentiwordnet.senti_synset(syn)
                             sentiment = sentiment.pos_score() - sentiment.neg_score()

                    tags = []

                    for z in indices:
                        tags.append(str(x["pos_tag"][z]))

                    tags = ''.join(tags)
                    print(sentiment)

                    Xb.append([negator_pos_token, sentence, "NEGPOS", tags])
                    Xb.append([neg_synset, sentence, "NEGSYN", sentiment])

                c = c + 1
            sentence = sentence + 1

        print("Negators")

        # Adjective modified nouns
        sentence = 1
        for x in sentences_tagged:
            c = 0
            start_index = x["token.i"][0]
            for y in x['dep_tag']:

                if y == "amod":
                    first_word = x["token.lemma"][c]
                    second_word = str(x['token.head'][c])
                    adj_mod_token = first_word + " " + second_word

                    Xb.append([adj_mod_token, sentence, "ADJMOD"])

                c = c + 1

            sentence = sentence + 1

        print("Adj mod Nouns")

        # Compound n-grams
        sentence = 1
        for x in sentences_tagged:
            c = 0
            start_index = x["token.i"][0]
            for y in x['dep_tag']:

                if y == "compound":
                    first_word = x["token.lemma"][c]
                    second_word = str(x['token.head'][c])
                    comp_token = first_word + " " + second_word

                    Xb.append([comp_token, sentence, "COMPOUND"])

                c = c + 1

            sentence = sentence + 1

        print("Compound ngrams")

        # n-grams (moet na stopwoorden verwijderen): !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        stop_words = set(stopwords.words('english'))
        sentence = 1
        for x in sentences_tagged:
            filtered_sentence = []
            for word in x["token.lemma"]:
                if word not in stop_words:
                    filtered_sentence.append(word)

            for y in range(0, len(filtered_sentence) - 1):
                unigram = str(filtered_sentence[y])
                Xb.append([unigram, sentence, "UNIGRAM"])

                if y + 1 <= len(filtered_sentence) - 1:
                    bigram = str(filtered_sentence[y]) + " " + str(filtered_sentence[y + 1])
                    Xb.append([bigram, sentence, "BIGRAM"])

                if y + 2 <= len(filtered_sentence) - 1:
                    trigram = str(filtered_sentence[y]) + " " + str(filtered_sentence[y + 1]) + " " + str(filtered_sentence[y + 2])
                    Xb.append([trigram, sentence, "TRIGRAM"])

                if y + 3 <= len(filtered_sentence) - 1:
                    quadgram = str(filtered_sentence[y]) + " " + str(filtered_sentence[y + 1]) + " " + str(filtered_sentence[y + 2] )+ " " + str(filtered_sentence[y + 3])
                    Xb.append([quadgram, sentence, "QUADGRAM"])

            sentence = sentence + 1

        print("Ngrams")


        # POSGRAMS (BIGRAM)
        sentence = 1
        for x in sentences_tagged:
            for y in range(0, len(x['token.lemma']) - 1):
                if y + 1 <= len(x["token.lemma"]) - 1:
                    posgram = str(x['token.lemma'][y] + " " + x['token.lemma'][y + 1])
                    tags = x['pos_tag'][y] + " " + x["pos_tag"][y + 1]
                    Xb.append([posgram, sentence, "POSGRAM", tags])

            sentence = sentence + 1


        # Sentisynset unigrams
        for x in sentences_tagged:
            for y in range(0, len(x["token.lemma"]) - 1):
                sentiuni = x["token_synset"][y]
                print(sentiuni)
                if sentiuni != "None":
                    sentiment = sentiwordnet.senti_synset(sentiuni)
                    sentiment = sentiment.pos_score() - sentiment.neg_score()
                    Xb.append([sentiuni, sentence, "SENTIUNI", sentiment])


        print("Done constructing all tokens")


        # Verwijderen van duplicates en samenvoegen van de sentences in de tokenlist (Xb)
        Xb = pd.DataFrame(Xb)

        Xb.columns = ['word', 'sentences', 'type', "extra"]
        Xb.word = Xb.word.astype(str)
        Xb.extra = Xb.extra.astype(str)
        Xb_copy = Xb["word"] + "_" + Xb['type'] + "_" + Xb['extra']
        print(Xb_copy)
        Xb_copy.columns = ['word']


        # Het aanmaken van de frequency table Nb
        Nb = pd.DataFrame(Xb_copy.value_counts())

        Nb['actual_word'] = Nb.index.values
        Nb.columns = ['frequency', 'word']

        between_list = [Xb, Nb]
        all_Xb_Nb.append(between_list)
        print(all_Xb_Nb)
        print(filename + " " + 'Has been tokenized and is ready for LDA')


file = open('all_Xb_Nb_Load_84084', "wb")

pickle.dump(all_Xb_Nb, file)

file.close()






