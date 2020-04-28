import spacy
import os
import re
import pandas as pd
from spacy.symbols import *
import numpy as np
import random
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
from nltk.corpus import wordnet
from contextlib import suppress
import csv
import json

os.chdir("C:/Users/Ruben/PycharmProjects/untitled1")
print(os.getcwd())

nltk.download('omw')

nlp = spacy.load('en_core_web_sm')

all_Xb_Nb = []

all_sentences = []

for filename in os.listdir("./d84082_pages"):



    if filename.endswith(".txt") and filename in ["d84084.txt"]:
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
                print(500)

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
                    Xb.append([x["token.lemma"][c], sentence, "NOUN", "",x["token_synset"][c]])
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


                    Xb.append([negator_pos_token, sentence, "NEGPOS", tags, neg_synset])
                    Xb.append([neg_synset, sentence, "NEGSYN", sentiment, ''])

                c = c + 1
            sentence = sentence + 1

        print("Negators")

        # Adjective modified nouns
        sentence = 1
        for x in sentences_tagged:
            c = 0

            for y in x['dep_tag']:

                if y == "amod":
                    first_word = x["token.lemma"][c]
                    second_word = str(x['token.head'][c])
                    adj_mod_token = first_word + " " + second_word

                    lesk = adapted_lesk(str(all_sentences[0][sentence - 1]), second_word)

                    if lesk is not None:
                        lesk = lesk.name()

                    Xb.append([adj_mod_token, sentence, "ADJMOD", '',[x['token_synset'][c], lesk]])

                c = c + 1

            sentence = sentence + 1

        print("Adj mod Nouns")

        # Compound n-grams
        sentence = 1

        for x in sentences_tagged:
            c = 0

            for y in x['dep_tag']:

                if y == "compound":
                    first_word = x["token.lemma"][c]
                    second_word = str(x['token.head'][c])
                    adj_mod_token = first_word + " " + second_word
                    second_word_synset = "None"
                    with suppress(Exception):
                        second_word_synset = list(x['token_synset'][x['token.text'].str.match(second_word)])[0]

                    # print(first_word, second_word, x['token.text'], x['token_synset'], second_word_synset)


                    Xb.append([adj_mod_token, sentence, "COMPOUND", '', [x['token_synset'][c], second_word_synset]])

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
                Xb.append([unigram, sentence, "UNIGRAM", '', x['token_synset'][y]])

                if y + 1 <= len(filtered_sentence) - 1:
                    bigram = str(filtered_sentence[y]) + " " + str(filtered_sentence[y + 1])
                    Xb.append([bigram, sentence, "BIGRAM", '', [x['token_synset'][y], x['token_synset'][y + 1]]])

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
                    Xb.append([posgram, sentence, "POSGRAM", tags, [x['token_synset'][y], x['token_synset'][y + 1]]])

            sentence = sentence + 1


        # Sentisynset unigrams
        sentence = 1
        for x in sentences_tagged:
            for y in range(0, len(x["token.lemma"]) - 1):
                sentiuni = x["token_synset"][y]
                if sentiuni != "None":
                    sentiment = sentiwordnet.senti_synset(sentiuni)
                    sentiment = sentiment.pos_score() - sentiment.neg_score()
                    Xb.append([sentiuni, sentence, "SENTIUNI", sentiment])
            sentence = sentence + 1

        print("Done constructing all tokens")


        # Verwijderen van duplicates en samenvoegen van de sentences in de tokenlist (Xb)
        Xb = pd.DataFrame(Xb)
        Xb.columns = ['word', 'sentences', 'type', "extra", "synsets"]
        Xb.word = Xb.word.astype(str)

        # print(Xb)


with open("./all_selected_tokens_84084_2", "rb") as f:
    a = pickle.load(f)
    plain_selected = a

# print(plain_selected)

all_selected_tokens = []



for x in plain_selected:


    token_split = x[1].rsplit("_", 2)
    # token_split = '_'.join(x[1].rsplit('_', 2)[:2])
    all_selected_tokens.append(token_split)

file = open("84084_all_selected_tokens_split", "wb")
pickle.dump(all_selected_tokens, file)
file.close()


# print('all selected tokens:', all_selected_tokens)


sentence_list = list(set(Xb['sentences']))



all_plain_tagged_sentences = []
all_ngram_tagged_sentences = []
all_pos_tagged_sentences = []
all_senti_tagged_sentences = []
all_all_tagged_sentences = []




plain_tagged_sentences = []
ngram_tagged_sentences = []
pos_tagged_sentences = []
senti_tagged_sentences = []
all_tagged_sentences = []

total_coverage = 0
total_hit_counter = 0


for x in sentence_list:


    targets_hit = []
    targets_hit_senti = []
    targets_hit_pos = []
    targets_hit_ngrams = []
    targets_hit_basic = []
    hit_counter = 0

    temp_Xb = Xb[Xb['sentences'] == x]


    # print(Xb)
    # print(Xb[Xb['sentences'] == 1])

    sentence = all_sentences[0][x-1]
    print('Sentence: ', x)


    synset = []
    for selected_token in all_selected_tokens:

        if selected_token[0] in list(temp_Xb['word']):

            # We skip trigram and quadgram because we will not be using synsets for those because of calculation times
            if selected_token[1] == "NOUN" or selected_token[1] == "UNIGRAM":
                synset_selected = adapted_lesk(str(sentence), str(selected_token[0]))
                if synset_selected is not None:
                    synset_selected = synset_selected.name()
                synset.append(str(synset_selected))
                continue

            if selected_token[1] == "COMPOUND" or selected_token[1] == "ADJMOD" or selected_token[1] == "BIGRAM" or selected_token[1] == "POSGRAM" or selected_token[1] == "NEGPOS":
                split_selected_token = selected_token[0].split(' ')
                synset_selected_1 = adapted_lesk(str(sentence), str(split_selected_token[0]))
                if synset_selected_1 is not None:
                    synset_selected_1 = synset_selected_1.name()
                synset_selected_2 = adapted_lesk(str(sentence), str(split_selected_token[1]))
                if synset_selected_2 is not None:
                    synset_selected_2 = synset_selected_2.name()
                synset_selected = [synset_selected_1, synset_selected_2]
                synset.append(str(synset_selected))
                continue

            else:
                synset.append("No synset")
        else:
            synset.append("No synset")


    temporary_selected_tokens = pd.DataFrame(all_selected_tokens)
    # print(temporary_selected_tokens)
    temporary_selected_tokens[2] = synset
    # print(temporary_selected_tokens)
    # print(temporary_selected_tokens[4][:50])
    temporary_selected_tokens.columns = ['word','type','synsets']
    synset = temporary_selected_tokens[-temporary_selected_tokens['synsets'].str.match("No synset")]
    synset = synset[-synset['synsets'].str.contains("None")]

    indices = list(synset.index)

    # Weird behaviour if you copy indices directly (it drops some indices probably because of the remove funciton used below??
    indices_copy = list(synset.index)

    test = []

    for y in indices_copy:

        temp_synsets = []

        if ',' in synset['synsets'][y]:
            temp_synsets = synset['synsets'][y].split(',')
            temp_synsets[0] = re.sub(r"[^a-zA-Z0-9;^.]+", '', temp_synsets[0])
            temp_synsets[1] = re.sub(r"[^a-zA-Z0-9;^.]+", '', temp_synsets[1])
            # print(temp_synsets[0])
            test_2 = y
            # Exception because wordnet sometimes gives an error because it can't recognize the synset for some reason
            with suppress(Exception):
                temp_synsets = [wordnet.synset(temp_synsets[0]), wordnet.synset(temp_synsets[1])]
                # print(temp_synsets)
                test.append(temp_synsets)
                test_2 = test_2 + 1

            if test_2 == y:
                indices.remove(y)

            # test = wordnet.synset(synset['synsets'][y])
        else:
            temp_synsets = wordnet.synset(synset['synsets'][y])
            test.append(temp_synsets)

    synset = synset.loc[indices, :]
    synset['synsets'] = test



    #ONT COMMENT DE COUNTER HIERONDER OOK GELIJK
    #
    temp_Xb = pd.DataFrame(temp_Xb)

    def create_wordnet_synsets(lists):
        output = []
        for j in lists:
            if isinstance(j, list):


                # longer than 4 checks for None types that somehow doesnt work with a None check
                if len(str(j[0])) > 4 and len(str(j[1])) > 4:


                    if isinstance(j[0], str) and isinstance(j[1], str):

                        output.append([wordnet.synset(j[0]), wordnet.synset(j[1])])
                    # elif's are for weird exceptions where the .name() during the construction of Xb does not work sometimes
                    elif 'Synset' in str(j[0]) and isinstance(j[1], str):
                        output.append([j[0], wordnet.synset(j[1])])

                    elif 'Synset' in str(j[1]) and isinstance(j[0], str):
                        output.append([wordnet.synset(j[0]), j[0]])

            elif len(j) > 4:
                # Once again an exception for wordnet just in case
                with suppress(Exception):
                    output.append(wordnet.synset(j))

        return output

    list_noun = list(temp_Xb[temp_Xb['type'].str.match('NOUN')]['word'])
    list_noun_synsets = list(temp_Xb[temp_Xb['type'].str.match('NOUN')]['synsets'])
    list_noun_synsets = create_wordnet_synsets(list_noun_synsets)

    list_compound = list(temp_Xb[temp_Xb['type'].str.match('COMPOUND')]['word'])
    list_compound_synsets = list(temp_Xb[temp_Xb['type'].str.match('COMPOUND')]['synsets'])
    list_compound_synsets = create_wordnet_synsets(list_compound_synsets)

    list_adjmod = list(temp_Xb[temp_Xb['type'].str.match('ADJMOD')]['word'])
    list_adjmod_synsets= list(temp_Xb[temp_Xb['type'].str.match('ADJMOD')]['synsets'])
    list_adjmod_synsets = create_wordnet_synsets(list_adjmod_synsets)


    list_unigram = list(temp_Xb[temp_Xb['type'].str.match('UNIGRAM')]['word'])
    list_unigram_synsets = list(temp_Xb[temp_Xb['type'].str.match('UNIGRAM')]['synsets'])
    list_unigram_synsets = create_wordnet_synsets(list_unigram_synsets)

    list_bigram = list(temp_Xb[temp_Xb['type'].str.match('BIGRAM')]['word'])
    list_bigram_synsets = list(temp_Xb[temp_Xb['type'].str.match('BIGRAM')]['synsets'])
    list_bigram_synsets = create_wordnet_synsets(list_bigram_synsets)

    list_trigram = list(temp_Xb[temp_Xb['type'].str.match('TRIGRAM')]['word'])

    list_quadgram = list(temp_Xb[temp_Xb['type'].str.match('QUADGRAM')]['word'])


    list_posgram = list(temp_Xb[temp_Xb['type'].str.match('POSGRAM')]['word'])
    list_posgram_synsets = list(temp_Xb[temp_Xb['type'].str.match('POSGRAM')]['synsets'])
    list_posgram_synsets = create_wordnet_synsets(list_posgram_synsets)



    list_negpos = list(temp_Xb[temp_Xb['type'].str.match('NEGPOS')]['word'])
    list_negpos_synsets = list(temp_Xb[temp_Xb['type'].str.match('NEGPOS')]['synsets'])
    list_negpos_synsets = create_wordnet_synsets(list_negpos_synsets)
    # list_negpos_extra = np.array(temp_Xb[temp_Xb['type'].str.match('NEGPOS')]['extra'])


    list_senti = list(temp_Xb[temp_Xb['type'].str.match('SENTIUNI')]['word'])
    # list_senti_extra = np.array(temp_Xb[temp_Xb['type'].str.match('SENTIUNI')]['extra'])

    list_extra = [list(temp_Xb['word']), list(temp_Xb['extra'])]



    counter = 0

    for selected_token in all_selected_tokens:


        if selected_token[1] == 'NOUN':

            if selected_token[0] in list_noun:
                hit_counter = hit_counter + 1
                print("match:", selected_token, selected_token[0])
                targets_hit.append(["match:", selected_token, selected_token[0]])
                targets_hit_basic.append(["match:", selected_token, selected_token[0]])

            elif counter in indices:
                selected_word = synset['synsets'][counter]
                for sentence_word in list_noun_synsets:
                    if sentence_word.wup_similarity(selected_word) is not None:
                        if sentence_word.wup_similarity(selected_word) > 0.8:
                            hit_counter = hit_counter + 1
                            print("match:", selected_token, selected_token[0])
                            targets_hit.append(["match:", selected_token, sentence_word])
                            targets_hit_basic.append(["match:", selected_token, sentence_word])




        if selected_token[1] == 'COMPOUND':

            if selected_token[0] in list_compound:
                hit_counter = hit_counter + 1
                print("match:", selected_token, selected_token[0])
                targets_hit.append(["match:", selected_token, selected_token[0]])
                targets_hit_basic.append(["match:", selected_token, selected_token[0]])

            elif counter in indices:
                selected_word_1 = synset['synsets'][counter][0]
                selected_word_2 = synset['synsets'][counter][1]
                for sentence_words in list_compound_synsets:
                    sentence_word_1 = sentence_words[0]
                    sentence_word_2 = sentence_words[1]
                    similarity_1 = selected_word_1.wup_similarity(sentence_word_1)
                    similarity_2 = selected_word_1.wup_similarity(sentence_word_2)
                    if similarity_1 is not None and similarity_2 is not None:
                        if similarity_1 > 0.8 and similarity_2 > 0.8:
                            hit_counter = hit_counter + 1
                            print("match:", selected_token, sentence_words)
                            targets_hit.append(["match:", selected_token, sentence_words])
                            targets_hit_basic.append(["match:", selected_token, sentence_words])


        if selected_token[1] == 'ADJMOD':

            if selected_token[0] in list_adjmod:
                hit_counter = hit_counter + 1
                print("match:", selected_token, selected_token[0])
                targets_hit.append(["match:", selected_token, selected_token[0]])
                targets_hit_basic.append(["match:", selected_token, selected_token[0]])

            elif counter in indices:
                selected_word_1 = synset['synsets'][counter][0]
                selected_word_2 = synset['synsets'][counter][1]
                for sentence_words in list_adjmod_synsets:
                    sentence_word_1 = sentence_words[0]
                    sentence_word_2 = sentence_words[1]
                    similarity_1 = selected_word_1.wup_similarity(sentence_word_1)
                    similarity_2 = selected_word_1.wup_similarity(sentence_word_2)
                    if similarity_1 is not None and similarity_2 is not None:
                        if similarity_1 > 0.8 and similarity_2 > 0.8:
                            hit_counter = hit_counter + 1
                            print("match:", selected_token, selected_token[0])
                            targets_hit.append(["match:", selected_token, sentence_words])
                            targets_hit_basic.append(["match:", selected_token, sentence_words])


        if selected_token[1] == 'UNIGRAM':

            if selected_token[0] in list_unigram:
                hit_counter = hit_counter + 1
                print("match:", selected_token, selected_token[0])
                targets_hit.append(["match:", selected_token, selected_token[0]])
                targets_hit_ngrams.append(["match:", selected_token, selected_token[0]])

            elif counter in indices:
                selected_word = synset['synsets'][counter]
                for sentence_word in list_unigram_synsets:
                    if sentence_word.wup_similarity(selected_word) is not None:
                        if sentence_word.wup_similarity(selected_word) > 0.8:
                            hit_counter = hit_counter + 1
                            print("match:", selected_token, selected_token[0])
                            targets_hit.append(["match:", selected_token, sentence_word])
                            targets_hit_ngrams.append(["match:", selected_token, sentence_word])



        if selected_token[1] == 'BIGRAM':

            if selected_token[0] in list_adjmod:
                hit_counter = hit_counter + 1
                print("match:", selected_token, selected_token[0])
                targets_hit.append(["match:", selected_token, selected_token[0]])
                targets_hit_ngrams.append(["match:", selected_token, selected_token[0]])

            elif counter in indices:
                selected_word_1 = synset['synsets'][counter][0]
                selected_word_2 = synset['synsets'][counter][1]
                for sentence_words in list_adjmod_synsets:
                    sentence_word_1 = sentence_words[0]
                    sentence_word_2 = sentence_words[1]
                    similarity_1 = selected_word_1.wup_similarity(sentence_word_1)
                    similarity_2 = selected_word_1.wup_similarity(sentence_word_2)
                    if similarity_1 is not None and similarity_2 is not None:
                        if similarity_1 > 0.8 and similarity_2 > 0.8:
                            hit_counter = hit_counter + 1
                            print("match:", selected_token, selected_token[0])
                            targets_hit.append(["match:", selected_token, sentence_words])
                            targets_hit_ngrams.append(["match:", selected_token, sentence_words])


        if selected_token[1] == 'TRIGRAM':
            if selected_token[0] in list_trigram:
                hit_counter = hit_counter + 1
                print("match:", selected_token, selected_token[0])
                targets_hit.append(["match:", selected_token, selected_token[0]])
                targets_hit_ngrams.append(["match:", selected_token, selected_token[0]])

        if selected_token[1] == 'QUADGRAM':
            if selected_token[0] in list_quadgram:
                hit_counter = hit_counter + 1
                print("match:", selected_token, selected_token[0])
                targets_hit.append(["match:", selected_token, selected_token[0]])
                targets_hit_ngrams.append(["match:", selected_token, selected_token[0]])

        if selected_token[1] == 'POSGRAM':

            if selected_token[0] in list_posgram:
                indices_2 = list_extra[0].index(selected_token[0])
                extra_info = list_extra[1][indices_2]

                if selected_token[2] in extra_info:
                    hit_counter = hit_counter + 1
                    print("match:", selected_token, selected_token[0])
                    targets_hit.append(["match:", selected_token, selected_token[0]])
                    targets_hit_pos.append(["match:", selected_token, selected_token[0]])

            elif counter in indices:
                selected_word_1 = synset['synsets'][counter][0]
                selected_word_2 = synset['synsets'][counter][1]
                for sentence_words in list_posgram_synsets:
                    sentence_word_1 = sentence_words[0]
                    sentence_word_2 = sentence_words[1]
                    similarity_1 = selected_word_1.wup_similarity(sentence_word_1)
                    similarity_2 = selected_word_2.wup_similarity(sentence_word_2)
                    if similarity_1 is not None and similarity_2 is not None:
                        indices_2 = list_extra[0].index(selected_token[0])
                        extra_info = list_extra[1][indices_2]
                        if similarity_1 > 0.8 and similarity_2 > 0.8 and selected_token[2] in extra_info:
                            hit_counter = hit_counter + 1
                            print("match:", selected_token, selected_token[0])
                            targets_hit.append(["match:", selected_token, sentence_words])
                            targets_hit_pos.append(["match:", selected_token, sentence_words])


        if selected_token[1] == 'NEGPOS':


            if selected_token[0] in list_negpos:

                indices_2 = list_extra[0].index(selected_token[0])

                extra_info = list_extra[1][indices_2]

                if selected_token[2] in extra_info:
                    hit_counter = hit_counter + 1
                    print("match:", selected_token, selected_token[0])
                    targets_hit.append(["match:", selected_token, selected_token[0]])
                    targets_hit_pos.append(["match:", selected_token, selected_token[0]])


            elif counter in indices:
                selected_word_1 = synset['synsets'][counter][0]
                selected_word_2 = synset['synsets'][counter][1]
                for sentence_words in list_negpos_synsets:
                    sentence_word_1 = sentence_words[0]
                    sentence_word_2 = sentence_words[1]
                    similarity_1 = selected_word_1.wup_similarity(sentence_word_1)
                    similarity_2 = selected_word_2.wup_similarity(sentence_word_2)
                    if similarity_1 is not None and similarity_2 is not None:
                        indices_2 = list_extra[0].index(selected_token[0])
                        extra_info = list_extra[1][indices_2]
                        if similarity_1 > 0.8 and similarity_2 > 0.8 and selected_token[2] in extra_info:
                            hit_counter = hit_counter + 1
                            print("match:", selected_token, selected_token[0])
                            targets_hit.append(["match:", selected_token, sentence_words])
                            targets_hit_pos.append(["match:", selected_token, sentence_words])



        if selected_token[1] == 'SENTIUNI':

            if selected_token[0] in list_senti:
                indices_2 = list_extra[0].index(selected_token[0])
                extra_info = list_extra[1][indices_2]

                # Different from other tokens with an extra tag because it is a number now brackets around extra_info ensure it works even if it is a single floatpoint number
                if str(selected_token[2]) in [extra_info]:
                    hit_counter = hit_counter + 1
                    print("match:", selected_token, selected_token[0])
                    targets_hit.append(["match:", selected_token, selected_token[0]])
                    targets_hit_senti.append(["match:", selected_token, selected_token[0]])

            elif counter in indices:
                selected_word_1 = synset['synsets'][counter][0]
                selected_word_2 = synset['synsets'][counter][1]
                for sentence_words in list_senti:
                    sentence_word_1 = sentence_words[0]
                    sentence_word_2 = sentence_words[1]
                    similarity_1 = selected_word_1.wup_similarity(sentence_word_1)
                    similarity_2 = selected_word_2.wup_similarity(sentence_word_2)
                    if similarity_1 is not None and similarity_2 is not None:
                        indices_2 = list_extra[0].index(selected_token[0])
                        extra_info = list_extra[1][indices_2]
                        if similarity_1 > 0.8 and similarity_2 > 0.8:
                            hit_counter = hit_counter + 1
                            print("match:", selected_token, selected_token[0])
                            targets_hit.append(["match:", selected_token, sentence_words])
                            targets_hit_senti.append(["match:", selected_token, sentence_words])



        counter = counter + 1

    coverage = hit_counter / len(plain_selected)
    # (UN)COMMENT BASED ON THE RUN
    plain_tagged_sentences.append([x - 1, targets_hit_basic, coverage])
    ngram_tagged_sentences.append([x - 1, targets_hit_ngrams, coverage])
    pos_tagged_sentences.append([x - 1, targets_hit_pos, coverage])
    senti_tagged_sentences.append([x - 1, targets_hit_senti, coverage])
    all_tagged_sentences.append([x - 1, targets_hit, coverage])



all_all_tagged_sentences.append(all_tagged_sentences)







# (UN)COMMENT BASED ON THE RUN


# with open("84084_all_tips_tagged_plain_2", 'w') as myfile:
#     wr = csv.writer(myfile)
#     wr.writerows(plain_tagged_sentences)
#
# with open("84084_all_tips_tagged_all_2", 'w') as myfile_2:
#     json.dump(all_tagged_sentences, myfile_2)
#
# with open("84084_all_tips_tagged_ngram_2", 'w') as myfile_3:
#     wr = csv.writer(myfile_3)
#     wr.writerows(ngram_tagged_sentences)
#
# with open("84084_all_tips_tagged_pos_2", 'w') as myfile_4:
#     wr = csv.writer(myfile_4)
#     wr.writerows(pos_tagged_sentences)
#
# with open("84084_all_tips_tagged_senti_2", 'w') as myfile_5:
#     wr = csv.writer(myfile_5)
#     wr.writerows(senti_tagged_sentences)
#



with open("./all_selected_tokens_84084_2", "rb") as f:
    a = pickle.load(f)
    all_selected = a

all_selected_tokens = []

for x in all_selected:
    token_split = x[1].rsplit("_", 2)
    all_selected_tokens.append(token_split)

tokens_to_go = all_selected_tokens

tagged_sentences = all_tagged_sentences


length_list = []
tokens_list = []
for sentence in tagged_sentences:
    if any(isinstance(el, list) for el in sentence[1]):
        length_list.append(len(sentence[1]))
    elif any(isinstance(el, str) for el in sentence[1]):
        length_list.append(1)
    else:
        length_list.append(0)
    tokens_list.append(sentence[1])

tokens_list_copy = []
for sentence in tokens_list:
    if len(sentence) > 0:
        test = []
        for token in sentence:
            test.append(token[1])
        tokens_list_copy.append(test)
    else:
        tokens_list_copy.append("")

tokens_list_copy_temp = tokens_list_copy

final_selection = []
testing_number = 0
for x in length_list:
    if x != 0:
        testing_number = testing_number + 1

while len(tokens_to_go) > 0:
    selected_sentence = [i for i, x in enumerate(length_list) if x == max(length_list)][0]
    # length_list[selected_sentence] = 0
    tokens_list_copy = tokens_list_copy_temp
    # print(tokens_list_copy)
    final_selection.append([tokens_list_copy[selected_sentence], selected_sentence])
    for token in tokens_list_copy[selected_sentence]:
        # print(token)
        index = 0
        for sentence in tokens_list_copy:

            if len(sentence) == 1:
                if sentence == [token]:
                    # print("yes")
                    length_list[index] = length_list[index] - 1
                    # tokens_list_copy_temp[index] = []

            elif len(sentence) != 0:
                if token in sentence:
                    # print("yes")
                    # tokens_list_copy_temp[index].remove(token)
                    length_list[index] = length_list[index] - 1

            index = index + 1
        # print(token)
        with suppress(Exception):
            tokens_to_go.remove(token)
    # print(length_list, sum(length_list))

sentence_list = []
for x in final_selection:
    sentence_list.append(x[1])

print(sentence_list)

all_selected_sentences = []
for i in sentence_list:
    print(i)
    print(all_sentences[0][i])
    all_selected_sentences.append([all_sentences[0][i], i])

with open("all_selected_sentences_84084_all.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences)



with open("./plain_selected_84084", "rb") as f:
    a = pickle.load(f)
    all_selected = a

all_selected_tokens = []

for x in all_selected:
    token_split = x[1].rsplit("_", 2)
    all_selected_tokens.append(token_split)

tokens_to_go = all_selected_tokens

tagged_sentences = plain_tagged_sentences


length_list = []
tokens_list = []
for sentence in tagged_sentences:
    if any(isinstance(el, list) for el in sentence[1]):
        length_list.append(len(sentence[1]))
    elif any(isinstance(el, str) for el in sentence[1]):
        length_list.append(1)
    else:
        length_list.append(0)
    tokens_list.append(sentence[1])

tokens_list_copy = []
for sentence in tokens_list:
    if len(sentence) > 0:
        test = []
        for token in sentence:
            test.append(token[1])
        tokens_list_copy.append(test)
    else:
        tokens_list_copy.append("")

tokens_list_copy_temp = tokens_list_copy

final_selection = []
testing_number = 0
for x in length_list:
    if x != 0:
        testing_number = testing_number + 1

while len(tokens_to_go) > 0:
    selected_sentence = [i for i, x in enumerate(length_list) if x == max(length_list)][0]
    # length_list[selected_sentence] = 0
    tokens_list_copy = tokens_list_copy_temp
    # print(tokens_list_copy)
    final_selection.append([tokens_list_copy[selected_sentence], selected_sentence])
    for token in tokens_list_copy[selected_sentence]:
        # print(token)
        index = 0
        for sentence in tokens_list_copy:

            if len(sentence) == 1:
                if sentence == [token]:
                    # print("yes")
                    length_list[index] = length_list[index] - 1
                    # tokens_list_copy_temp[index] = []

            elif len(sentence) != 0:
                if token in sentence:
                    # print("yes")
                    # tokens_list_copy_temp[index].remove(token)
                    length_list[index] = length_list[index] - 1

            index = index + 1
        # print(token)
        with suppress(Exception):
            tokens_to_go.remove(token)
    # print(length_list, sum(length_list))

sentence_list = []
for x in final_selection:
    sentence_list.append(x[1])

print(sentence_list)

all_selected_sentences = []
for i in sentence_list:
    print(i)
    print(all_sentences[0][i])
    all_selected_sentences.append([all_sentences[0][i], i])

with open("all_selected_sentences_84084_plain.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences)


with open("./ngram_selected_84084", "rb") as f:
    a = pickle.load(f)
    all_selected = a

all_selected_tokens = []

for x in all_selected:
    token_split = x[1].rsplit("_", 2)
    all_selected_tokens.append(token_split)

tokens_to_go = all_selected_tokens

tagged_sentences = ngram_tagged_sentences


length_list = []
tokens_list = []
for sentence in tagged_sentences:
    if any(isinstance(el, list) for el in sentence[1]):
        length_list.append(len(sentence[1]))
    elif any(isinstance(el, str) for el in sentence[1]):
        length_list.append(1)
    else:
        length_list.append(0)
    tokens_list.append(sentence[1])

tokens_list_copy = []
for sentence in tokens_list:
    if len(sentence) > 0:
        test = []
        for token in sentence:
            test.append(token[1])
        tokens_list_copy.append(test)
    else:
        tokens_list_copy.append("")

tokens_list_copy_temp = tokens_list_copy

final_selection = []
testing_number = 0
for x in length_list:
    if x != 0:
        testing_number = testing_number + 1

while len(tokens_to_go) > 0:
    selected_sentence = [i for i, x in enumerate(length_list) if x == max(length_list)][0]
    # length_list[selected_sentence] = 0
    tokens_list_copy = tokens_list_copy_temp
    # print(tokens_list_copy)
    final_selection.append([tokens_list_copy[selected_sentence], selected_sentence])
    for token in tokens_list_copy[selected_sentence]:
        # print(token)
        index = 0
        for sentence in tokens_list_copy:

            if len(sentence) == 1:
                if sentence == [token]:
                    # print("yes")
                    length_list[index] = length_list[index] - 1
                    # tokens_list_copy_temp[index] = []

            elif len(sentence) != 0:
                if token in sentence:
                    # print("yes")
                    # tokens_list_copy_temp[index].remove(token)
                    length_list[index] = length_list[index] - 1

            index = index + 1
        # print(token)
        with suppress(Exception):
            tokens_to_go.remove(token)
    # print(length_list, sum(length_list))

sentence_list = []
for x in final_selection:
    sentence_list.append(x[1])

print(sentence_list)

all_selected_sentences = []
for i in sentence_list:
    print(i)
    print(all_sentences[0][i])
    all_selected_sentences.append([all_sentences[0][i], i])

with open("all_selected_sentences_84084_ngram.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences)


with open("./pos_selected_84084", "rb") as f:
    a = pickle.load(f)
    all_selected = a

all_selected_tokens = []

for x in all_selected:
    token_split = x[1].rsplit("_", 2)
    all_selected_tokens.append(token_split)

tokens_to_go = all_selected_tokens

tagged_sentences = pos_tagged_sentences


length_list = []
tokens_list = []
for sentence in tagged_sentences:
    if any(isinstance(el, list) for el in sentence[1]):
        length_list.append(len(sentence[1]))
    elif any(isinstance(el, str) for el in sentence[1]):
        length_list.append(1)
    else:
        length_list.append(0)
    tokens_list.append(sentence[1])

tokens_list_copy = []
for sentence in tokens_list:
    if len(sentence) > 0:
        test = []
        for token in sentence:
            test.append(token[1])
        tokens_list_copy.append(test)
    else:
        tokens_list_copy.append("")

tokens_list_copy_temp = tokens_list_copy

final_selection = []


while len(tokens_to_go) > 0:
    selected_sentence = [i for i, x in enumerate(length_list) if x == max(length_list)][0]
    # length_list[selected_sentence] = 0
    tokens_list_copy = tokens_list_copy_temp
    # print(tokens_list_copy)
    final_selection.append([tokens_list_copy[selected_sentence], selected_sentence])
    for token in tokens_list_copy[selected_sentence]:
        # print(token)
        index = 0
        for sentence in tokens_list_copy:

            if len(sentence) == 1:
                if sentence == [token]:
                    # print("yes")
                    length_list[index] = length_list[index] - 1
                    # tokens_list_copy_temp[index] = []

            elif len(sentence) != 0:
                if token in sentence:
                    # print("yes")
                    # tokens_list_copy_temp[index].remove(token)
                    length_list[index] = length_list[index] - 1

            index = index + 1
        # print(token)
        with suppress(Exception):
            tokens_to_go.remove(token)
    # print(length_list, sum(length_list))

sentence_list = []
for x in final_selection:
    sentence_list.append(x[1])

print(sentence_list)

all_selected_sentences = []
for i in sentence_list:
    print(i)
    print(all_sentences[0][i])
    all_selected_sentences.append([all_sentences[0][i], i])

with open("all_selected_sentences_84084_pos.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences)


with open("./sent_selected_84084", "rb") as f:
    a = pickle.load(f)
    all_selected = a

all_selected_tokens = []

for x in all_selected:
    token_split = x[1].rsplit("_", 2)
    all_selected_tokens.append(token_split)

tokens_to_go = all_selected_tokens

tagged_sentences = senti_tagged_sentences

length_list = []
tokens_list = []
for sentence in tagged_sentences:
    print(sentence[1])
    if any(isinstance(el, list) for el in sentence[1]):
        length_list.append(len(sentence[1]))
    elif any(isinstance(el, str) for el in sentence[1]):
        length_list.append(1)
    else:
        length_list.append(0)
    tokens_list.append(sentence[1])

tokens_list_copy = []
for sentence in tokens_list:
    if len(sentence) > 0:
        test = []
        for token in sentence:
            test.append(token[1])
        tokens_list_copy.append(test)
    else:
        tokens_list_copy.append("")

tokens_list_copy_temp = tokens_list_copy

final_selection = []
testing_number = 0
for x in length_list:
    if x != 0:
        testing_number = testing_number + 1

while len(tokens_to_go) > 0:
    selected_sentence = [i for i, x in enumerate(length_list) if x == max(length_list)][0]
    # length_list[selected_sentence] = 0
    tokens_list_copy = tokens_list_copy_temp
    # print(tokens_list_copy)
    final_selection.append([tokens_list_copy[selected_sentence], selected_sentence])
    for token in tokens_list_copy[selected_sentence]:
        # print(token)
        index = 0
        for sentence in tokens_list_copy:

            if len(sentence) == 1:
                if sentence == [token]:
                    # print("yes")
                    length_list[index] = length_list[index] - 1
                    # tokens_list_copy_temp[index] = []

            elif len(sentence) != 0:
                if token in sentence:
                    # print("yes")
                    # tokens_list_copy_temp[index].remove(token)
                    length_list[index] = length_list[index] - 1

            index = index + 1
        # print(token)
        with suppress(Exception):
            tokens_to_go.remove(token)
    # print(length_list, sum(length_list))

sentence_list = []
for x in final_selection:
    sentence_list.append(x[1])

print(sentence_list)

all_selected_sentences = []
for i in sentence_list:
    print(i)
    print(all_sentences[0][i])
    all_selected_sentences.append([all_sentences[0][i], i])

with open("all_selected_sentences_84084_sent.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(all_selected_sentences)