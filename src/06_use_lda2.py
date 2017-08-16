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
from nltk import tokenize
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
sia = SentimentIntensityAnalyzer()

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

reviews = pd.read_csv("/home/nwang/proj/data/intermediate/SF_reviews.csv")
review_main_text = reviews["review main text"]
review_counts = [2, 64, 10, 9, 26, 144, 33, 31, 5, 4, 89, 14, 22, 21, 1, 4, 13, 8, 3, 17, 14, 3, 37, 12, 124, 11, 24, 2, 17, 37, 34, 5, 17, 0, 37, 4, 4, 41, 12, 6, 3, 3, 9, 1, 7, 6, 44, 17, 84, 50, 6, 43, 54, 56, 19, 1, 7, 30, 14, 68, 41, 29, 8, 0, 17, 87, 49, 22, 52, 12, 4, 1, 42, 17, 30, 19, 32, 1, 7, 4, 14, 18, 46, 25, 4, 0, 20, 131, 7, 42, 2, 15, 13, 101, 24, 13, 1, 36, 46, 13, 9, 25, 6, 3, 3, 81, 11, 8, 13, 4, 15, 2, 22, 6, 36, 13, 38, 14, 26, 11, 4, 19, 19, 16, 6, 26, 6, 10, 23, 65, 3, 5, 12, 256, 31, 7, 78, 3, 37, 31, 59, 37, 14, 9, 94, 100, 94, 50, 9, 4, 36, 78, 124, 37, 18, 7, 4, 96, 3, 87, 0, 3, 5, 15, 11, 0, 13, 82, 8, 29, 17, 11, 33, 55, 278, 21, 12, 4]
doctor_info = pd.read_csv("/home/nwang/proj/data/intermediate/SF_doctors.csv")


# Current topics
### Topic name, the list of models ID and topic ID pairs.
### These correspond exactly to the models loaded above.
topics=[
["Positive comments", [[0,8], [1,5], [2,3]]],
["Payment", [[0,7], [1,13], [2,5]]],
["Appointment and office visits", [[0,9], [1,4], [2,8]]],
["Dental care", [[0,1], [0,4], [1,12], [2,14]]],
["Women's health", [[0,2], [1,14], [2,13]]],
["Surgery", [[0,13], [1,8], [2,4]]],
["Allergy treatments", [[0,11], [1,9], [2,0]]],
["Skin procedures"	, [[0,3], [1,6]]],
["Eye care", [[0,10]]],
["Reconstructive surgery", [[1,2]]],
["Urology treatments", [[2,12]]],
["Internet and media", [[0,14]]]
]
topic_num = 12
topics_matrix = [ [ -1 for i in range(category) ] for j in range(mod_num) ]
for index, item in enumerate(topics):
    for item2 in item[1]:
        topics_matrix[item2[0]][item2[1]] = index

# Main loop!
### Logic for every doctor, find two things
###     !. The more mentioned FIVE topics then-> their sentiments.
###     2. The 3 most positive sentences and the 3 most negative sentences.
###       2.1 Rank all sentences according to sentiment analysis.
counter = 0
big_str = []
#sentiment_list = []
for doctor_id in range(0, len(review_counts)):
    doctor_review_count = int(review_counts[doctor_id])
    is_dentist = False
    if str(doctor_info.loc[doctor_id]["specialty"]).lower().find("dent") > -1:
        is_dentist = True
        
    # DID NOT keep info about individual reviews. All sentences are stored in a
    # long list regardless of whether they are from the same reviews or not!
    ###########################################################################
    # Build sentence data frame
    ###########################################################################
    this_doc = pd.DataFrame(columns = ["sentence", "sentiment", "topic", "topic_score"])
    sent_count = 0
    for review_id in range(0, doctor_review_count):
        real_review_id = review_id + counter
        review_rating = reviews.loc[real_review_id]["review rating"]
        sentences = tokenize.sent_tokenize(review_main_text[real_review_id])
        
        ### ###
        #total_sentiment = 0
        for sentence in sentences:
            sentiment = sia.polarity_scores(sentence)["compound"]
            this_topic = -1
            sent_tokens = tokenizer.tokenize(str(sentence).lower())
            cleaned_sent = [p_stemmer.stem(i) for i in sent_tokens]
            
            sent_topics = []
            for mod_id in range(0, mod_num):
                model = ldamodel[mod_id]
                dicti = dictionary[mod_id]
                lda_score = model[dicti.doc2bow(cleaned_sent)]
                for item in lda_score:
                    sent_topics.append((mod_id, item[0], item[1]))
            sent_topics =  sorted(sent_topics, key=lambda x: x[2], reverse=True)
            if sent_topics[0][2] > 0.7:  # only if there is a strong topic
                this_topic = topics_matrix[sent_topics[0][0]][sent_topics[0][1]]
            #print(sentiment, this_topic, " ||| ", sentence)
            #total_sentiment += sentiment
        #average_sentiment = total_sentiment/len(sentences)
        #sentiment_list.append((average_sentiment, review_rating))
        ### ###
            
            this_doc.loc[sent_count] = [sentence, sentiment, this_topic, sent_topics[0][2]]
            sent_count += 1
    
    ###########################################################################
    # Compiling results
    ###########################################################################
    # Top and bottom sentiments!
    this_doc2 = this_doc.sort_values(["sentiment"], ascending=[0]).reset_index(drop=True)
    this_doc2 = this_doc2.loc[this_doc2["topic"] != -1].reset_index(drop=True)
    this_doc2 = this_doc2.loc[this_doc2["topic_score"] > 0.5].reset_index(drop=True)
    sent_count_2 = len(this_doc2)
    composite = "NONE"
    if sent_count_2 > 5:
        sent1 = sent2 = sent3 = sent4 = sent5 = sent6 = ""
        if this_doc2.loc[0]["sentiment"] > 0.4:
            sent1 = this_doc2.loc[0]["sentence"]
        if this_doc2.loc[1]["sentiment"] > 0.4:
            sent2 = this_doc2.loc[1]["sentence"]
        if this_doc2.loc[2]["sentiment"] > 0.4:
            sent3 = this_doc2.loc[2]["sentence"]
        if this_doc2.loc[sent_count_2-1]["sentiment"] < -0.2:
            sent4 = this_doc2.loc[sent_count_2-1]["sentence"]
        if this_doc2.loc[sent_count_2-2]["sentiment"] < -0.2:
            sent5 = this_doc2.loc[sent_count_2-2]["sentence"]
        if this_doc2.loc[sent_count_2-3]["sentiment"] < -0.2:
            sent6 = this_doc2.loc[sent_count_2-3]["sentence"]
        composite = sent1 + "SSEEPP" + sent2 + "SSEEPP" + sent3 + "SSEEPP" + sent4 + "SSEEPP" + sent5 + "SSEEPP" + sent6 + "SSEEPP" + str(sent_count)
    doctor_info.set_value(doctor_id, "summary", composite)

    # Top topics!
    doc_topics = [ [ 0 for i in range(2) ] for j in range(topic_num) ]  # [total count, count positive]
    for index2 in range(0, len(this_doc2)):
        topic_index = this_doc2.loc[index2]["topic"]
        if topic_index != -1:
            doc_topics[topic_index][0] += 1
            topic_sentiment = this_doc2.loc[index2]["sentiment"]
            
            if topic_index == 5 and doctor_info.loc[doctor_id]["last_name"] == "Morganroth":
                print(doctor_info.loc[doctor_id]["first_name"], doctor_info.loc[doctor_id]["last_name"], topic_sentiment, this_doc2.loc[index2]["sentence"])
                
            if topic_sentiment > 0.1:  # !!! Deciding what is positive
                doc_topics[topic_index][1] += 1
    # Do not display dentist stuff for non-dentist
    if not is_dentist:
        doc_topics[3][0] = 0
    # Do not output "positive comment" info
    doc_topics[0][0] = 0
    
    # Turn into tuples
    doc_topic_tuples = []
    for index3, item in enumerate(doc_topics):
        doc_topic_tuples.append((index3, item[0], item[1]))
    doc_topic_tuples =  sorted(doc_topic_tuples, key=lambda x: x[1], reverse=True)
    for index4 in range(0, 5):
        if doc_topic_tuples[index4][1] >= 10:
            topic_name = topics[doc_topic_tuples[index4][0]][0]
            percent_positive = str(int(doc_topic_tuples[index4][2]/doc_topic_tuples[index4][1] * 100))
            composite = topic_name + "SSEEPP" + percent_positive + "SSEEPP" + str(doc_topic_tuples[index4][1])
            doctor_info.set_value(doctor_id, "percent{0}".format(str(index4+1)), composite)
            
            print(topic_name, "XXXXXX", doctor_info.loc[doctor_id]["specialty"])
            big_str.append(topic_name + "XXXXXX" + str(doctor_info.loc[doctor_id]["specialty"]))
        else:
            doctor_info.set_value(doctor_id, "percent{0}".format(str(index4+1)), "NONE")

    
    print(counter/5088)
    counter += doctor_review_count
    del this_doc
    del this_doc2
    

doctor_info.to_csv("/home/nwang/proj/data/final/SF_doctors_added.csv")


###############################################################################
###############################################################################
###############################################################################


### Plot some useful things
import matplotlib.pyplot as plt
cumulation = [[] for i in range(0,topic_num)]
#plt.style.use("ggplot")
plt.figure(figsize=(4,4), dpi=150)
for ii in range(0, topic_num):
    phrase = topics[ii][0]
    for i in range(0, len(doctor_info)):
        if doctor_info.loc[i]["percent1"].find(phrase) != -1:
            cumulation[ii].append(int(doctor_info.loc[i]["percent1"].split("SSEEPP")[1]))
    
    if cumulation[ii] != []:
        plt.hist(cumulation[ii], 20, histtype='step',alpha=10, lw = 2, label=phrase)


plt.ylabel("Count")
plt.xlabel("Percent positive across doctors")
plt.xlim((0,100))
plt.legend(framealpha=0)

### Topic and doctor labels
import matplotlib.pyplot as plt
dental_count_list = []
non_dental_count_list = []
for doctor_id in range(0, len(review_counts)):
    specialty = str(doctor_info.loc[doctor_id]["specialty"]).lower().find("dent")
    
    count = 0
    for i in range(0, 5):
        yesno = str(doctor_info.loc[doctor_id]["percent{0}".format(str(i+1))]).lower().find("dent")
        if yesno != -1:
            count = int(str(doctor_info.loc[doctor_id]["percent{0}".format(str(i+1))]).split("SSEEPP")[2])
            break

    if specialty != -1:
        dental_count_list.append(count)
    else:
        non_dental_count_list.append(count)

plt.figure(figsize=(4,4), dpi=150)
plt.hist(dental_count_list, 20, histtype='step',alpha=10, lw = 2, label="Dentist")
plt.hist(non_dental_count_list, 20, histtype='step',alpha=10, lw = 2, label="Not dentist")    
plt.ylabel("Doctor count")
plt.xlabel("Dental mentions (no. of sentences)")
plt.xlim((0,100))
plt.legend(framealpha=0)

###############################################################################
###############################################################################
###############################################################################
# VALIDATION!!!! and PLOTING!!!
# All: dental, surgery, skin, eye care, women, allergy
specialty_pairs = [('Naturopath', 'none'),
 ('Dermatopathology', 'Skin procedures'),
 ('nan', 'none'),
 ('Orthopedic Sports Medicine', 'Surgery'),
 ('Otolaryngology', 'Surgery'),
 ('Endocrinology', 'none'),
 ('Physical Medicine & Rehabilitation', 'none'),
 ('Facial Plastic Surgery', 'Surgery'),
 ('Interventional Pain Medicine', 'none'),
 ('Urology', 'none'),
 ('Otolaryngology/Facial Plastic Surgery', 'Surgery'),
 ('Orthopedic Surgery', 'Surgery'),
 ('Internal Medicine', 'none'),
 ('Family Medicine', 'none'),
 ('MOHS-Micrographic Surgery', 'Surgery'),
 ('Pain Medicine', 'none'),
 ('Procedural Dermatology', 'Skin procedures'),
 ('Orthopedic Surgery of the Spine', 'Surgery'),
 ('Foot & Ankle Surgery', 'Surgery'),
 ('Allergy & Immunology', 'Allergy treatments'),
 ('Foot and Ankle Surgery', 'Surgery'),
 ('Endodontics', 'none'),
 ('Dentistry', 'Dental care'),
 ('Gynecology', "Women's health"),
 ('Plastic and Reconstructive Surgery', 'Surgery'),
 ('Oral & Maxillofacial Surgery', 'Surgery'),
 ('Psychiatry', 'none'),
 ('Surgery', 'Surgery'),
 ('Anesthesiology', 'none'),
 ('Gastroenterology', "Women's health"),
 ('Family Medicine Adult Medicine', 'none'),
 ('Obstetrics', "Women's health"),
 ('Diabetes & Metabolism', 'none'),
 ('Hand Surgery', 'Surgery'),
 ('General Dentistry', 'Dental care'),
 ('Dermatology', 'Skin procedures'),
 ('Optometry', 'Eye care'),
 ('Pediatrics', 'none'),
 ('Neurology', 'none'),
 ('Cardiovascular Disease', 'none'),
 ('Obstetrics & Gynecology', "Women's health"),
 ('Nephrology', 'none'),
 ('Podiatry',  'Surgery'),
 ('Chiropractics', 'none'),
 ('Hospitalist', 'none')]

# % dentist has dental, % non-dentist has dental, total no of dentists, total nu of non-dentist
specialty_dict = {"Dental care": [0, 0, 0, 0], "Surgery": [0, 0, 0, 0], "Skin procedures": [0, 0, 0, 0], "Eye care": [0, 0, 0, 0], "Women's health": [0, 0, 0, 0], "Allergy treatments": [0, 0, 0, 0]}
specialty_name_dict = {'Allergy treatments': 'Allergy & Immunology',
 'Dental care': 'DentistryXXXXXXGeneral Dentistry',
 'Eye care': 'Optometry',
 'Skin procedures': 'DermatopathologyXXXXXXProcedural DermatologyXXXXXXDermatology',
 'Surgery': 'Orthopedic Sports MedicineXXXXXXOtolaryngologyXXXXXXFacial Plastic SurgeryXXXXXXOtolaryngology/Facial Plastic SurgeryXXXXXXOrthopedic SurgeryXXXXXXMOHS-Micrographic SurgeryXXXXXXOrthopedic Surgery of the SpineXXXXXXFoot & Ankle SurgeryXXXXXXFoot and Ankle SurgeryXXXXXXPlastic and Reconstructive SurgeryXXXXXXOral & Maxillofacial SurgeryXXXXXXSurgeryXXXXXXHand SurgeryXXXXXXPodiatry',
 "Women's health": 'GynecologyXXXXXXGastroenterologyXXXXXXObstetricsXXXXXXObstetrics & Gynecology'}
 #'none': 'NaturopathXXXXXXnanXXXXXXEndocrinologyXXXXXXPhysical Medicine & RehabilitationXXXXXXInterventional Pain MedicineXXXXXXUrologyXXXXXXInternal MedicineXXXXXXFamily MedicineXXXXXXPain MedicineXXXXXXEndodonticsXXXXXXPsychiatryXXXXXXAnesthesiologyXXXXXXFamily Medicine Adult MedicineXXXXXXDiabetes & MetabolismXXXXXXPediatricsXXXXXXNeurologyXXXXXXCardiovascular DiseaseXXXXXXNephrologyXXXXXXChiropracticsXXXXXXHospitalist'}

for item in big_str:
    cat = item.split("XXXXXX")[0]
    spe = item.split("XXXXXX")[1].split(", ")

    try:
        specialties_under = specialty_name_dict[cat]
        is_special_doctor = False
        for every_specialty in spe:
            if specialties_under.find(every_specialty) > -1:
                is_special_doctor = True
                break
            
        if is_special_doctor:
            specialty_dict[cat][0] += 1
        else:
            specialty_dict[cat][1] += 1
    except:
        pass

# Check total number of doctors
for doctor_id in range(0, len(review_counts)):
    specialty_check = {"Dental care": 0, "Surgery": 0, "Skin procedures": 0, "Eye care": 0, "Women's health": 0, "Allergy treatments": 0}

    all_specialty = str(doctor_info.loc[doctor_id]["specialty"]).split(", ")
    for item in all_specialty:
        for cat in specialty_name_dict:
            if specialty_name_dict[cat].find(item) > -1 and specialty_check[cat] == 0:
                specialty_dict[cat][2] += 1
                specialty_check[cat] = 1
                break
            
    for cat in specialty_name_dict:
        if specialty_check[cat] == 0:
            specialty_dict[cat][3] += 1

for cat in specialty_dict:
    #specialty_dict[cat][0] = specialty_dict[cat][0]/specialty_dict[cat][2]
    #specialty_dict[cat][1] = specialty_dict[cat][1]/specialty_dict[cat][3]
    print(specialty_dict[cat][1])


import matplotlib.pyplot as plt
# data to plot
n_groups = 6
means_frank = (100, 87.5, 66.6, 37.5, 30, 30.4)
means_guido = (0.56, 12.3, 0, 0, 0, 4.5)
labels = ('Skin procedures', 'Allergy treatments', "Women's health", 'Eye care', "Surgery", 'Dental care')
means_frank = (30.4, 30, 37.5, 66.6, 87.5, 100)
means_guido = (4.5, 0, 0, 0, 12.3, 0.56)
labels = ('Dental care', 'Surgery', 'Eye care', "Women's health", 'Allergy treatments', 'Skin procedures')

# create plot
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

plt.style.use("ggplot")

plt.figure(figsize=(4,4), dpi=150)
rects1 = plt.barh(index, means_guido, bar_width,
                 alpha=opacity,
                 label='Topic matches specialty')
 
rects2 = plt.barh(index + bar_width, means_frank, bar_width,
                 alpha=opacity,
                 label='Topic mismatches specialty')
plt.yticks(index + bar_width/2, labels)
plt.xlabel('% of doctors having the topic mentioned in their reviews')
plt.ylabel('Topics')
plt.xlim((0,100))
#plt.legend()



