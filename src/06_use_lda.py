#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:07:01 2017

@author: nwang
"""

import gensim
import pickle
import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np

# Set up tokenizer.
tokenizer = RegexpTokenizer(r'\w+')
# Set up stop words.
stop = set(stopwords.words('english'))
# Set up stemmer.
p_stemmer = PorterStemmer()
# For sentiment analysis
sid = SentimentIntensityAnalyzer()

# Load saved models.
ldamodel1 = pickle.load(open("/home/nwang/proj/data/lda_model/ldamodel_v2.pickle", "rb"))
ldamodel2 = pickle.load(open("/home/nwang/proj/data/lda_model/ldamodel_v3.pickle", "rb"))
ldamodel3 = pickle.load(open("/home/nwang/proj/data/lda_model/ldamodel_v4.pickle", "rb"))
dict1 = pickle.load(open("/home/nwang/proj/data/lda_model/dictionary_v2.pickle", "rb"))
dict2 = pickle.load(open("/home/nwang/proj/data/lda_model/dictionary_v3.pickle", "rb"))
dict3 = pickle.load(open("/home/nwang/proj/data/lda_model/dictionary_v4.pickle", "rb"))
corpus1 = pickle.load(open("/home/nwang/proj/data/lda_model/corpus_v2.pickle", "rb"))
corpus2 = pickle.load(open("/home/nwang/proj/data/lda_model/corpus_v3.pickle", "rb"))
corpus3 = pickle.load(open("/home/nwang/proj/data/lda_model/corpus_v4.pickle", "rb"))
mod_num = 3
category = 15
ldamodel = [ldamodel1, ldamodel2, ldamodel3]
dictionary = [dict1, dict2, dict3]
corpus = [corpus1, corpus2, corpus3]

reviews = pd.read_csv("/home/nwang/proj/data/intermediate/all_reviews.csv")
review_main_text = reviews["review main text"]
review_counts = [21, 58, 53, 6, 16, 2, 0, 0, 0, 1, 0, 2, 0, 0, 23, 14, 16, 61, 12, 13, 14, 0, 1, 5, 1, 9, 101, 1, 33, 33, 14, 2, 5, 5, 6, 4, 38, 3, 5, 11, 5, 3, 9, 20, 0, 0, 1, 2, 1, 1, 19, 2, 6, 6, 15, 1, 41, 30, 1, 1, 1, 6, 4, 12, 6, 28, 3, 14, 2, 35, 4, 14, 8, 11, 2, 6, 66, 45, 10, 0, 49, 27, 6, 27, 8, 37, 18, 0, 1, 5, 70, 7, 6, 24, 2, 10, 1, 8, 9, 6, 13, 0, 193, 0, 8, 29, 4, 15, 3, 5, 7, 24, 31, 9, 0, 2, 64, 30, 35, 10, 7, 5, 43, 40, 13, 16, 15, 11, 2, 2, 30, 28, 13, 9, 15, 12, 38, 2, 6, 5, 22, 11, 70, 82, 2, 8, 67, 39, 60, 6, 3, 188, 12, 11, 4, 7, 81, 68, 22, 2, 7, 3, 3, 3, 34, 7, 4, 4, 12, 57, 46, 17, 4, 23, 27, 19, 25, 108, 33, 25, 1, 18, 14, 17, 80, 30, 5, 9, 19, 11, 71, 23, 5, 4, 30, 11, 32, 43, 13, 27, 25, 4, 17, 3, 0, 39, 10, 32, 16, 9, 35, 25, 49, 7, 16, 20, 30, 3, 39, 12, 546, 46, 4, 29, 1, 35, 92, 5, 9, 108, 3, 13, 14, 105, 19, 6, 54, 26, 31, 3, 66, 45, 28, 18, 2, 17, 5, 64, 7, 6, 2, 11, 6, 35, 19, 5, 54, 11, 9, 15, 40, 6, 1, 58, 28, 44, 11, 4, 59, 7, 52, 14, 170, 22, 18, 43, 63, 1, 75, 46, 42, 15, 9, 13, 20, 13, 41, 9, 3, 1, 3, 7, 8, 3, 13, 32, 10, 45, 20, 19, 46, 3, 12, 19, 13, 46, 7, 21, 6, 11, 25, 50, 19, 68, 17, 20, 3, 26, 14, 62, 15, 90, 13, 10, 34, 4, 19, 11, 8, 40, 4, 3, 27, 14, 0, 4, 24, 5, 21, 78, 33, 30, 21, 67, 22, 39, 55, 14, 22, 33, 10, 108, 50, 8, 9, 87, 17, 157, 4, 1, 4, 27, 26, 43, 13, 1, 3, 33, 16, 45, 16, 6, 73, 25, 22, 10, 26, 6, 54, 14, 115, 29, 19, 9, 4, 13, 4, 23, 49, 8, 14, 7, 13, 21, 55, 144, 12, 1, 37, 42, 8, 4, 0, 16, 10, 14, 3, 8, 2, 5, 4, 11, 8, 10, 1, 30, 118, 62, 8, 72, 17, 1, 8, 26, 16, 8, 65, 76, 45, 28, 14, 89, 9, 22, 8, 21, 16, 5, 10, 6, 3, 10, 46, 15, 46, 20, 12, 27, 39, 74, 6, 33, 27, 2, 35, 19, 30, 26, 3, 12, 1, 1, 1, 19, 33, 1, 40, 5, 9, 24, 0, 0, 1, 25, 8, 50, 6, 9, 49, 0, 7, 24, 89, 22, 7, 7, 16, 23, 48, 1, 10, 2, 4, 1, 0, 21, 3, 48, 5, 17, 0, 5, 6, 6, 1, 13]
review_counts = [2, 64, 10, 9, 26, 144, 33, 31, 5, 4, 89, 14, 22, 21, 1, 4, 13, 8, 3, 17, 14, 3, 37, 12, 124, 11, 24, 2, 17, 37, 34, 5, 17, 0, 37, 4, 4, 41, 12, 6, 3, 3, 9, 1, 7, 6, 44, 17, 84, 50, 6, 43, 54, 56, 19, 1, 7, 30, 14, 68, 41, 29, 8, 0, 17, 87, 49, 22, 52, 12, 4, 1, 42, 17, 30, 19, 32, 1, 7, 4, 14, 18, 46, 25, 4, 0, 20, 131, 7, 42, 2, 15, 13, 101, 24, 13, 1, 36, 46, 13, 9, 25, 6, 3, 3, 81, 11, 8, 13, 4, 15, 2, 22, 6, 36, 13, 38, 14, 26, 11, 4, 19, 19, 16, 6, 26, 6, 10, 23, 65, 3, 5, 12, 256, 31, 7, 78, 3, 37, 31, 59, 37, 14, 9, 94, 100, 94, 50, 9, 4, 36, 78, 124, 37, 18, 7, 4, 96, 3, 87, 0, 3, 5, 15, 11, 0, 13, 82, 8, 29, 17, 11, 33, 55, 278, 21, 12, 4]
ratings = -1 * np.ones((len(review_counts), mod_num, category))

# Article
text = "This doctor is great, I would go back to his clinic again. But the one thing that I don't like is that the wait time being too long, it took me 50 minutes to be seen!"
text = "This doctor is great, I would go back to his clinic again. The wait time being too long, it took me 50 minutes to be seen!"

counter = 0
for doctor_id in range(0, len(review_counts)):
    doctor_review_count = int(review_counts[doctor_id])
    all_review_ratings = -1 * np.ones((mod_num, category, doctor_review_count))
    
    # Calculate rationgs
    for review_id in range(0, doctor_review_count):
        sentences = nltk.tokenize.sent_tokenize(review_main_text[counter])
        sent_num = len(sentences)
        all_sent_ratings = -1 * np.ones((mod_num, category, sent_num))

        #print(all_sent_ratings, "\n")
        for i, sentence in enumerate(sentences):
            pos_or_neg = sid.polarity_scores(sentence)["compound"]
            sent_rating = 0
            if pos_or_neg > 1:
                sent_rating = 1
                
            tokens = tokenizer.tokenize(str(sentence).lower())
            cleaned_text = [p_stemmer.stem(i) for i in tokens]
            for mod_id in range(0, mod_num):
                m = ldamodel[mod_id]
                d = dictionary[mod_id]
                lda_score = m[d.doc2bow(cleaned_text)]
                highest_relevance = 0
                highest_relevance_id = -1
                for j in lda_score:
                    if j[1] > highest_relevance:
                        highest_relevance = j[1]
                        highest_relevance_id = j[0]
                all_sent_ratings[mod_id, highest_relevance_id, i] = sent_rating
            
        for mod_id in range(0, mod_num):
            for cat_id in range(0, category):
                sents_have_rating = 0
                sents_total_rating = 0
                for i, sentence in enumerate(sentences):
                    if all_sent_ratings[mod_id, cat_id, i] != -1:
                        sents_have_rating += 1
                        sents_total_rating += all_sent_ratings[mod_id, cat_id, i]
                if sents_have_rating != 0:
                    all_review_ratings[mod_id, cat_id, review_id] = 100*sents_total_rating/sents_have_rating
        counter += 1
        
    # Compiling calculations
    for mod_id in range(0, mod_num):
        for cat_id in range(0, category):
            reviews_have_rating = 0
            reviews_total_rating = 0
            for review_id in range(0, doctor_review_count):
                if all_review_ratings[mod_id, cat_id, review_id] != -1:
                    reviews_have_rating += 1
                    reviews_total_rating += all_review_ratings[mod_id, cat_id, review_id]
            if reviews_have_rating != 0:
                ratings[doctor_id, mod_id, cat_id] = int(reviews_total_rating/reviews_have_rating)
        

# Combining across models.    
# recommend   8	  5  3
# appointment 9	  4  8
# insurance   7 13  5
doctor_info = pd.read_csv("/home/nwang/proj/data/final/doctor_speadsheet_step2.csv")
r_i = [8,5,3]
a_i = [9,4,8]
i_i = [7,13,5]
r = [0]*len(review_counts)
a = [0]*len(review_counts)
ii = [0]*len(review_counts)
for doctor_id in range(0, len(review_counts)):
    count = 0
    total = 0
    for i, item in enumerate(r_i):
        if ratings[doctor_id][i][item] != -1:
            count += 1
            total += ratings[doctor_id][i][item]
    if count > 0:
        doctor_info.set_value(doctor_id, "rating1", int(round(total/count)))
    else:
        doctor_info.set_value(doctor_id, "rating1", 0)

    count = 0
    total = 0
    for i, item in enumerate(a_i):
        if ratings[doctor_id][i][item] != -1:
            count += 1
            total += ratings[doctor_id][i][item]
    if count > 0:
        doctor_info.set_value(doctor_id, "rating2", int(round(total/count)))
    else:
        doctor_info.set_value(doctor_id, "rating2", 0)
        
    count = 0
    total = 0
    for i, item in enumerate(i_i):
        if ratings[doctor_id][i][item] != -1:
            count += 1
            total += ratings[doctor_id][i][item]
    if count > 0:
        doctor_info.set_value(doctor_id, "rating3", int(round(total/count)))
    else:
        doctor_info.set_value(doctor_id, "rating3", 0)
    
    doctor_info.set_value(doctor_id, "rating4", 0)
    doctor_info.set_value(doctor_id, "rating5", 0)
    
doctor_info.to_csv("/home/nwang/proj/data/final/doctor_speadsheet_step3.csv")

    