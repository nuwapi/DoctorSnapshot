#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:41:01 2017

@author: nwang
"""

import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
import statistics
from collections import Counter
import numpy as np

reviews = pd.read_csv("/home/nwang/proj/data/intermediate/all_reviews.csv")
review_main_text = reviews["review main text"]
review_rating = reviews["review rating"]
one_star_reviews = reviews.loc[reviews["review rating"] == 1.0]
five_star_reviews = reviews.loc[reviews["review rating"] == 5.0]
one_star_reviews.to_csv("/home/nwang/proj/data/intermediate/one_star_reviews.csv")
five_star_reviews.to_csv("/home/nwang/proj/data/intermediate/five_star_reviews.csv")
data = pd.read_csv("/home/nwang/proj/data/final/doctor_speadsheet_step1.csv")

 
'''
!! Rating distribution !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
rating_count = [0, 0, 0, 0, 0]
five_star_reviews = []
for i in review_rating:
    if i == 1.0:
        rating_count[0] += 1
    elif i == 2.0:
        rating_count[1] += 1
    elif i == 3.0:
        rating_count[2] += 1
    elif i == 4.0:
        rating_count[3] += 1
    elif i == 5.0:
        rating_count[4] += 1
            
# Plotting raw review rating distribution.
plt.figure(figsize=(4,4), dpi=150)
plt.style.use("ggplot")
plt.bar([1,2,3,4,5], rating_count)
plt.xlabel("Rating")
plt.ylabel("Count")

    
'''
!! Word count distribution !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
tokenizer = RegexpTokenizer(r'\w+')
word_count = []
for i in review_main_text:
    word_count.append(len(tokenizer.tokenize(i)))
average_word_count = sum(word_count)/len(word_count)
median_word_count = statistics.median(word_count)

plt.figure(figsize=(4,4), dpi=150)
plt.style.use("ggplot")
plt.hist(word_count, 100, histtype='bar')
legend1 = plt.axvline(x=average_word_count, color='k', linestyle='--', linewidth = 1, label="average")
legend2 = plt.axvline(x=median_word_count, color='k', linestyle='-.', linewidth = 1, label="median")
plt.legend(handles=[legend1, legend2], framealpha=0)
plt.xlabel("Word count")
plt.ylabel("Occurrences")
plt.xlim((0,1000))

'''
!! Dcotor review count distribution !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
review_counts = [21, 58, 53, 6, 16, 2, 0, 0, 0, 1, 0, 2, 0, 0, 23, 14, 16, 61, 12, 13, 14, 0, 1, 5, 1, 9, 101, 1, 33, 33, 14, 2, 5, 5, 6, 4, 38, 3, 5, 11, 5, 3, 9, 20, 0, 0, 1, 2, 1, 1, 19, 2, 6, 6, 15, 1, 41, 30, 1, 1, 1, 6, 4, 12, 6, 28, 3, 14, 2, 35, 4, 14, 8, 11, 2, 6, 66, 45, 10, 0, 49, 27, 6, 27, 8, 37, 18, 0, 1, 5, 70, 7, 6, 24, 2, 10, 1, 8, 9, 6, 13, 0, 193, 0, 8, 29, 4, 15, 3, 5, 7, 24, 31, 9, 0, 2, 64, 30, 35, 10, 7, 5, 43, 40, 13, 16, 15, 11, 2, 2, 30, 28, 13, 9, 15, 12, 38, 2, 6, 5, 22, 11, 70, 82, 2, 8, 67, 39, 60, 6, 3, 188, 12, 11, 4, 7, 81, 68, 22, 2, 7, 3, 3, 3, 34, 7, 4, 4, 12, 57, 46, 17, 4, 23, 27, 19, 25, 108, 33, 25, 1, 18, 14, 17, 80, 30, 5, 9, 19, 11, 71, 23, 5, 4, 30, 11, 32, 43, 13, 27, 25, 4, 17, 3, 0, 39, 10, 32, 16, 9, 35, 25, 49, 7, 16, 20, 30, 3, 39, 12, 546, 46, 4, 29, 1, 35, 92, 5, 9, 108, 3, 13, 14, 105, 19, 6, 54, 26, 31, 3, 66, 45, 28, 18, 2, 17, 5, 64, 7, 6, 2, 11, 6, 35, 19, 5, 54, 11, 9, 15, 40, 6, 1, 58, 28, 44, 11, 4, 59, 7, 52, 14, 170, 22, 18, 43, 63, 1, 75, 46, 42, 15, 9, 13, 20, 13, 41, 9, 3, 1, 3, 7, 8, 3, 13, 32, 10, 45, 20, 19, 46, 3, 12, 19, 13, 46, 7, 21, 6, 11, 25, 50, 19, 68, 17, 20, 3, 26, 14, 62, 15, 90, 13, 10, 34, 4, 19, 11, 8, 40, 4, 3, 27, 14, 0, 4, 24, 5, 21, 78, 33, 30, 21, 67, 22, 39, 55, 14, 22, 33, 10, 108, 50, 8, 9, 87, 17, 157, 4, 1, 4, 27, 26, 43, 13, 1, 3, 33, 16, 45, 16, 6, 73, 25, 22, 10, 26, 6, 54, 14, 115, 29, 19, 9, 4, 13, 4, 23, 49, 8, 14, 7, 13, 21, 55, 144, 12, 1, 37, 42, 8, 4, 0, 16, 10, 14, 3, 8, 2, 5, 4, 11, 8, 10, 1, 30, 118, 62, 8, 72, 17, 1, 8, 26, 16, 8, 65, 76, 45, 28, 14, 89, 9, 22, 8, 21, 16, 5, 10, 6, 3, 10, 46, 15, 46, 20, 12, 27, 39, 74, 6, 33, 27, 2, 35, 19, 30, 26, 3, 12, 1, 1, 1, 19, 33, 1, 40, 5, 9, 24, 0, 0, 1, 25, 8, 50, 6, 9, 49, 0, 7, 24, 89, 22, 7, 7, 16, 23, 48, 1, 10, 2, 4, 1, 0, 21, 3, 48, 5, 17, 0, 5, 6, 6, 1, 13]
review_counts_numerical = []
for i in review_counts:
    review_counts_numerical.append(int(i))
    
a = ["aa", "aa", "bb"]
plt.figure(figsize=(4,4), dpi=150)
plt.style.use("ggplot")
plt.hist(review_counts_numerical, 50, histtype='bar')
plt.xlabel("Doctor review count")
plt.ylabel("Occurrences")
plt.xlim((0,300))


'''
!! Dcotor specialty count distribution !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
specialty_count = []
for i in data.index:
    if not pd.isnull(data.loc[i]["specialty"]):
        string = data.loc[i]["specialty"]
        string = string.replace("Procedural Dermatology", "Dermatology")
        string = string.replace("Dermatopathology", "Dermatology")
        string = string.replace("General Dentistry", "Dentistry")
        string = string.replace("Orthopedic Surgery of the Spine", "Orthopedic Surgery")
        string = string.replace("Plastic and Reconstructive Surgery", "Plastic Surgery")
        string = string.replace("Facial Plastic Surgery", "Plastic Surgery")
        string = string.replace("Foot & Ankle Surgery", "Foot and Ankle Surgery")
        specialty_count += string.split(", ")
string_counts = Counter(specialty_count)
specialty_df = pd.DataFrame.from_dict(string_counts, orient='index').sort_values([0], ascending=[1])
pd_plot = specialty_df.plot.barh(stacked=True, figsize=(4,35), fontsize=20, legend=False);
#pd_plot = specialty_df.plot(kind='bar', figsize=(20,8), fontsize=20, legend=False)
pd_plot.set_xlabel("Count", fontsize=20)


'''
!! Get all doctor names !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
fn = data["first_name"].tolist()
ln = data["last_name"].tolist()
all_names = list(set(fn + ln))


'''
!! Matching doctor to revie records !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
review_url = []
for i in range(0, 11694):
    review_url.append(reviews.loc[i]["page url"].split("?")[0])
review_url = list(set(review_url))

data_url = []
for i in range(0, 714):
    data_url.append(data.loc[i]["yelp"].split("?")[0])
data_url = list(set(data_url))

pairs = []
p1 = []
p2 = []
for indexi, i in enumerate(review_url):
    r_url = i.split("/")[2].split("-")
    top_count = 0
    top_indexj = -1
    for indexj, j in enumerate(data_url):
        this_count = 0
        d_url = j.split("/")[2].split("-")
        for ii in r_url:
            for jj in d_url:
                if ii == jj:
                    this_count += 1
        if this_count > top_count:
            top_count = this_count
            top_indexj = indexj
            
    if top_indexj != -1:
        print(review_url[indexi], data_url[top_indexj])
        pairs.append([review_url[indexi], data_url[top_indexj]])
        p1.append(review_url[indexi])
        p2.append(data_url[top_indexj])
print(len(set(p1)))
print(len(set(p2)))
set_p2 = set(p2)
new_ids = []
for i in p2:
    for index, j in enumerate(set_p2):
        if i == j:
            new_ids.append(index)
            
master_string = ""
data['phone'] = data['phone'].astype(str)
for i in range(0, 714):
    doc_url1 = data.loc[i]["yelp"].split("?")[0]
    full = "http://www.yelp.com" + doc_url1
    
    checked = False
    for index, j in enumerate(p2):
        if doc_url1 == j:
            checked = True
            data.set_value(i, "doctor ID", str(new_ids[index]))
            break        
    if checked == False:
        data.set_value(i, "doctor ID", "-100")
        
    #
    phone_number = str(data.loc[i]["phone"])
    fi = phone_number[0:3]
    mi = phone_number[3:6]
    la = phone_number[6:10]
    formatted_phone = "({0}) {1}-{2}".format(fi, mi, la)
    data.set_value(i, "phone", formatted_phone)
    
    #
    address = str(data.loc[i]["address"]).replace(" ,", ",")
    data.set_value(i, "address", address)
    
    #np.nanz
    if pd.isnull(data.loc[i]["website"]):
        data.set_value(i, "website", "NONE")

# Save
data.loc[data["doctor ID"] != "-100"].set_index('doctor ID').to_csv("/home/nwang/proj/data/final/doctor_speadsheet_step2.csv")

'''
!! Plot doctor locations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
plt.figure(figsize=(4,4), dpi=150)
plt.scatter(data['loc_lon'], data['loc_lat'], s=0.1)

'''
!! Grab SF and SD reviews !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
# Require doctor step 2 to be ran first!!!
data2 = pd.read_csv("/home/nwang/proj/data/final/doctor_speadsheet_step2.csv")
reviews = pd.read_csv("/home/nwang/proj/data/intermediate/all_reviews.csv")

# reindex reviews
for i in range(0, len(reviews)):
    page = reviews.loc[i]["page url"].split("?")[0]
    check = False
    for index, j in enumerate(p1):
        if page == j:
            reviews.set_value(i, "doctor ID", str(new_ids[index]))
            check = True
            break
    if check == False:
        print("Error")
    print(i/11694)
reviews.to_csv("/home/nwang/proj/data/intermediate/all_reviews.csv")

# Grab SD, SF
for i in range(0, len(reviews)):
    ID = reviews.loc[i]["doctor ID"]
    lon = data2.loc[int(ID)]["loc_lon"]
    if lon > -121 and lon < -118:
        reviews.set_value(i, "page url", "XXX")
    print(i/11694)



'''
!! Plot sentiment-rating correlations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
import pickle

sentiment_list = pickle.load(open("/home/nwang/proj/plots/sentiment_rating_tuples_vader.pickle", "rb"))
one = []
two = []
three = []
four = []
five = []
for item in sentiment_list:
    if item[1] == 1:
        one.append(item[0])
    elif item[1] == 2:
        two.append(item[0])
    elif item[1] == 3:
        three.append(item[0])
    elif item[1] == 4:
        four.append(item[0])
    elif item[1] == 5:
        five.append(item[0])

print(sum(one)/len(one))
print(sum(two)/len(two))
print(sum(three)/len(three))
print(sum(four)/len(four))
print(sum(five)/len(five))

#plt.scatter(*zip(*sentiment_list))
plt.style.use("ggplot")
plt.figure(figsize=(4,4), dpi=150)
h1 = plt.hist(one, 20, histtype='bar',alpha=0.6, lw = 2,normed=True, label='One star')
h2 = plt.hist(two, 20, histtype='bar',alpha=0.6,lw = 2,normed=True, label='Two stars')
h3 = plt.hist(three, 20, histtype='bar',alpha=0.6,lw = 2,normed=True, label='Three stars')
h4 = plt.hist(four, 20, histtype='bar',alpha=0.6,lw = 2,normed=True, label='Four stars')
h5 = plt.hist(five, 20, histtype='bar',alpha=0.6,lw = 2, normed=True,label='Five stars')
plt.ylabel("Normalized count")
plt.xlabel("Sentiment")
plt.xlim((-1,1))
plt.ylim((0,4))
plt.legend(framealpha=0)










