import os
import pickle
import scipy.stats as stats
from scipy.stats import fisher_exact
import pandas as pd
from textblob import TextBlob




os.chdir("C:/Users/Ruben/PycharmProjects/untitled1")




with open("./tokens_freq_selected_84084", "rb") as f:
    a = pickle.load(f)
    tokens_freq_selected = a

with open("./deletion_list", "rb") as f:
     a = pickle.load(f)
     deletion_list = a




tokens_freq_selected_copy = []
for hotel in tokens_freq_selected:
    hotel_copy = []
    for token in tokens_freq_selected[0]:

        # EXTRA RULES
        if ',' not in token[1]:

            if token[1] not in deletion_list:

                if 'SENTIUNI' not in token[1]:
                    print(token[1])
                    if '.' in str(token[1]) or "'" in str(token[1]) or '-' in str(token[1]) or "(" in str(token[1]) or ")" in str(token[1]):
                        print(token[1])
                        continue

                if 'SENTIUNI' in token[1]:
                    if token[0] < 54:
                        continue

                if 'POSGRAM' in token[1] or 'NEGPOS' in token[1]:
                    if token[0] <42:
                        continue

                if 'UNIGRAM' in token[1] or 'BIGRAM' in token[1] or 'TRIGRAM' in token[1] or 'QUADGRAM' in token[1]:
                    if token[0] <37:
                        continue

            hotel_copy.append(token)

        tokens_freq_selected_copy.append(hotel_copy)



frequency_list = []
for hotel in tokens_freq_selected_copy:
    frequency_plain = 0
    frequency_ngrams = 0
    frequency_pos = 0
    frequency_sentiment = 0
    frequency_all = 0
    for token in hotel:

        frequency_all = frequency_all + token[0]

        if "NOUN" in token[1] or "ADJMOD" in token[1] or "COMPOUND" in token[1]:
           frequency_plain = frequency_plain + token[0]
        elif "UNIGRAM" in token[1] or "BIGRAM" in token[1] or "TRIGRAM" in token[1] or "QUADGRAM" in token[1]:
            frequency_ngrams = frequency_ngrams + token[0]
        elif "NEGPOS" in token[1] or "POSGRAM" in token[1]:
            frequency_pos = frequency_pos + token[0]
        elif "SENTIUNI" in token[1] or "NEGSYN" in token[1]:
            frequency_sentiment = frequency_sentiment + token[0]
        else:
            print(token[1])

    frequency_ngrams = frequency_plain + frequency_ngrams
    frequency_pos = frequency_plain + frequency_pos
    frequency_sentiment = frequency_plain + frequency_sentiment

    frequency_list.append([frequency_plain, frequency_ngrams, frequency_pos, frequency_sentiment, frequency_all])

print(frequency_list)

# counter is 0 because we are only interested in the first (0 index) hotel (which is the hotel we want to select tips for, the others are the most similar hotels)
counter = 0
other_hotel_indices = [0, 1, 2, 3, 4]
other_hotels = []
other_hotels_indices = list(filter(lambda x: x != counter, other_hotel_indices))

for other_hotel_index in other_hotels_indices:
    other_hotel = tokens_freq_selected[other_hotel_index]
    other_hotel = pd.DataFrame(other_hotel)
    other_hotel.columns = ['frequency', 'token']
    other_hotels.append(other_hotel)
    #PLAIN

plain_selected = []
pos_selected = []
sent_selected = []
ngram_selected = []
all_selected_tokens = []
hotel = tokens_freq_selected_copy[0]

for token in hotel:

        #AANPASSEN NAAR NOUN IN HET PAPER
        if "ADJMOD" in token[1] or "COMPOUND" in token[1] or 'NOUN' in token[1]:
            frequency_token = token[0]
            frequency_similar = 0
            index = 0
            for other_hotel in other_hotels:

                frequency_similar = other_hotel['frequency'][other_hotel['token'] == token[1]]

                if len(frequency_similar) == 0:
                    frequency_similar = 0
                else:
                    frequency_similar = int(frequency_similar)

                tot_freq_hotel = frequency_list[counter][0]
                tot_freq_other_hotel = frequency_list[index][0]

                contingency_table = [[frequency_token, frequency_similar],[tot_freq_hotel - frequency_token, tot_freq_other_hotel - frequency_similar]]
                p_value = fisher_exact(contingency_table)[1]
                index = index + 1
                if p_value < 0.05:
                    plain_selected.append(token)
                    all_selected_tokens.append(token)
                    break

        elif "NEGPOS" in token[1] or "POSGRAM" in token[1]:
            frequency_token = token[0]
            frequency_similar = 0
            index = 0
            for other_hotel in other_hotels:
                frequency_similar = other_hotel['frequency'][other_hotel['token'] == token[1]]

                #POTENTIAL SENTIMENT FILTER
                # sentiment = TextBlob(str(other_hotel['token'][other_hotel['token'] == token[1]]).split('_')[0]).sentiment
                # if sentiment[0] == 0 and sentiment[1] == 0:
                #     break

                if len(frequency_similar) == 0:
                    frequency_similar = 0
                else:
                    frequency_similar = int(frequency_similar)


                tot_freq_hotel = frequency_list[counter][2]
                tot_freq_other_hotel = frequency_list[index][2]

                contingency_table = [[frequency_token, frequency_similar],[tot_freq_hotel - frequency_token, tot_freq_other_hotel - frequency_similar]]
                p_value = fisher_exact(contingency_table)[1]
                index = index + 1
                if p_value < 0.05 and 'NEGPOS' or 'POSGRAM' in token[1]:
                    pos_selected.append(token)
                    all_selected_tokens.append(token)
                    break
                # THIS IS THE CASE OF AN ADJMOD
                elif p_value < 0.05 and 'ADJMOD' in token[1]:
                    plain_selected.append(token)
                    all_selected_tokens.append(token)
                    break



        elif "SENTIUNI" in token[1] or "NEGSYN" in token[1]:
            frequency_token = token[0]
            frequency_similar = 0
            index = 0
            for other_hotel in other_hotels:
                if token[1] in other_hotel['token']:
                    frequency_similar = other_hotel['frequency'][other_hotel['token'] == token]
                    print(other_hotel['token'][other_hotel['token'] == token])
                else:
                    frequency_similar = 0

                tot_freq_hotel = frequency_list[counter][3]
                tot_freq_other_hotel = frequency_list[index][3]

                contingency_table = [[frequency_token, frequency_similar],
                                     [tot_freq_hotel - frequency_token, tot_freq_other_hotel - frequency_similar]]
                p_value = fisher_exact(contingency_table)[1]
                index = index + 1
                if p_value < 0.05:
                    sent_selected.append(token)
                    all_selected_tokens.append(token)
                    break


        elif "UNIGRAM" in token[1] or "BIGRAM" in token[1] or "TRIGRAM" in token[1] or "QUADGRAM" in token[1]:

            # EXTRA RULES
            if len(token[1]) > 12: #(This gets rid of the UNIGRAMS with length 2 or lower and bigrams smaller than 2x2:
                frequency_token = token[0]
                frequency_similar = 0
                index = 0
                for other_hotel in other_hotels:
                    if token[1] in other_hotel['token']:
                        frequency_similar = other_hotel['frequency'][other_hotel['token'] == token]
                        print(other_hotel['token'][other_hotel['token'] == token])
                    else:
                        frequency_similar = 0

                    tot_freq_hotel = frequency_list[counter][1]
                    tot_freq_other_hotel = frequency_list[index][1]

                    contingency_table = [[frequency_token, frequency_similar],
                                         [tot_freq_hotel - frequency_token, tot_freq_other_hotel - frequency_similar]]
                    p_value = fisher_exact(contingency_table)[1]
                    index = index + 1
                    if p_value < 0.05:
                        ngram_selected.append(token)
                        all_selected_tokens.append(token)
                        break



print(len(plain_selected))
print(plain_selected)
print(len(pos_selected))
print(pos_selected)
print(len(sent_selected))
print(sent_selected)
print(len(ngram_selected))
print(ngram_selected)
print(len(all_selected_tokens))
print(all_selected_tokens)

file = open("plain_selected_84084", "wb")
pickle.dump(plain_selected, file)
file.close()

file = open("sent_selected_84084", "wb")
pickle.dump(sent_selected, file)
file.close()

file = open("ngram_selected_84084", "wb")
pickle.dump(ngram_selected, file)
file.close()

file = open("all_selected_tokens_84084_2", "wb")
pickle.dump(all_selected_tokens, file)
file.close()

file = open("pos_selected_84084", "wb")
pickle.dump(pos_selected, file)
file.close()


# print(token_list)
# print(len(token_list))
# all_selected_tokens.append(token_list)
# file = open('all_selected_tokens', "wb")
# pickle.dump(all_selected_tokens, file)
# file.close()
#
#






