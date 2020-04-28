import pickle
import os
import numpy as np
from scipy.spatial import distance


os.chdir("C:/Users/Ruben/PycharmProjects/untitled1")
# print(os.getcwd())

file = open('./amenity_arrays_Load1', 'rb')

amenity_arrays = pickle.load(file)

file.close()

file = open('./LDA_arrays_Load1', 'rb')

LDA_arrays = pickle.load(file)

file.close()

# print(len(amenity_arrays[1]))
# print(len(LDA_arrays))

filelist = []
for filename in os.listdir("./d84082_pages"):
    filelist.append(str(filename))

test = [5,6,7,8,9,0]

similarity_input = {}

length = range(0, len(LDA_arrays))

length_amenity = range(0, len(amenity_arrays))


for i in length:

    similarity_input[filelist[i]] = LDA_arrays[i]

    for j in length_amenity:

        np.append(similarity_input[filelist[i]], [amenity_arrays[j][i]])

chosen_hotel_sets = []

print(similarity_input)

for i in filelist:


    stored_distances = []

    for j in filelist:

        stored_distances.append(distance.euclidean(similarity_input[i], similarity_input[j]))

    chosen_distances_index = []
    for k in range(0,5):
        chosen_distances_index.append(np.argmin(stored_distances))
        del stored_distances[np.argmin(stored_distances)]

    grouped_hotels = []

    for l in chosen_distances_index:
        grouped_hotels.append(filelist[l])

    chosen_hotel_sets.append(grouped_hotels)

file = open('hotel_sets', "wb")

pickle.dump(chosen_hotel_sets, file)

file.close()