#!/usr/bin/env python3
# Created by Nuo Wang.
# Last modified on 8/17/2017.

# Required libraries.
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

### Step 1: Initialization.

# Set up tokenizer.
tokenizer = RegexpTokenizer(r'\w+')
# Set up stop words.
stop = set(stopwords.words('english'))
# Set up stemmer.
p_stemmer = PorterStemmer()
# For sentiment analysis
sia = SentimentIntensityAnalyzer()

# Load saved models.
# I trained three LDA models from step 5, they have slightly different topics.
# I trained three models because there is randomness in LDA training.
ldamodel1 = pickle.load(open("PATH/model/lda_1.pickle", "rb"))
ldamodel2 = pickle.load(open("PATH/model/lda_2.pickle", "rb"))
ldamodel3 = pickle.load(open("PATH/model/lda_3.pickle", "rb"))
dict1 = pickle.load(open("PATH/model/dictionary_1.pickle", "rb"))
dict2 = pickle.load(open("PATH/model/dictionary_2.pickle", "rb"))
dict3 = pickle.load(open("PATH/model/dictionary_3.pickle", "rb"))
corpus1 = pickle.load(open("PATH/model/corpus_1.pickle", "rb"))
corpus2 = pickle.load(open("PATH/model/corpus_2.pickle", "rb"))
corpus3 = pickle.load(open("PATH/model/corpus_3.pickle", "rb"))
mod_num = 3
category = 15
ldamodel = [ldamodel1, ldamodel2, ldamodel3]
dictionary = [dict1, dict2, dict3]
corpus = [corpus1, corpus2, corpus3]

# Load my dataset.
doctor_info = pd.read_csv("PATH/data/yelp_doctors.csv")
reviews = pd.read_csv("PATH/data/yelp_reviews.csv")
review_main_text = reviews["review main text"]
# This is review count for SF doctors, copied from step 4.
review_counts = [2, 64, 10, 9, 26, 144, 33, 31, 5, 4, 89, 14, 22, 21, 1, 4, 13, 8, 3, 17, 14, 3, 37, 12, 124, 11, 24, 2, 17, 37, 34, 5, 17, 0, 37, 4, 4, 41, 12, 6, 3, 3, 9, 1, 7, 6, 44, 17, 84, 50, 6, 43, 54, 56, 19, 1, 7, 30, 14, 68, 41, 29, 8, 0, 17, 87, 49, 22, 52, 12, 4, 1, 42, 17, 30, 19, 32, 1, 7, 4, 14, 18, 46, 25, 4, 0, 20, 131, 7, 42, 2, 15, 13, 101, 24, 13, 1, 36, 46, 13, 9, 25, 6, 3, 3, 81, 11, 8, 13, 4, 15, 2, 22, 6, 36, 13, 38, 14, 26, 11, 4, 19, 19, 16, 6, 26, 6, 10, 23, 65, 3, 5, 12, 256, 31, 7, 78, 3, 37, 31, 59, 37, 14, 9, 94, 100, 94, 50, 9, 4, 36, 78, 124, 37, 18, 7, 4, 96, 3, 87, 0, 3, 5, 15, 11, 0, 13, 82, 8, 29, 17, 11, 33, 55, 278, 21, 12, 4]

# Generate the "topic matrix".
# This list is completely manually generated. The names of the topics are assigned by me.
# Data structure: Topic name, the list of models ID and topic ID pairs.
# There are three models (model ID 0 to 2) and each model contains 15 topics (topic ID 0 to 14).
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
# There are 12 topics that I can rationalize, i.e. that meant something.
topic_num = 12
# Assign topic names to the topic matrix.
# If a topic does not exist for a model-ID topic ID pair, its "name" is set to -1.
topics_matrix = [ [ -1 for i in range(category) ] for j in range(mod_num) ]
for index, item in enumerate(topics):
    for item2 in item[1]:
        topics_matrix[item2[0]][item2[1]] = index

### Step 2: Generate the contents of the doctors' snapshots.

counter = 0
# The temporary string that stores all of the review highlights in each round of the for loop below.
big_str = []
# For every doctor, find two things:
#     1. The most mentioned FIVE topics in their reviews.
#         1.1 The sentiments of these topics.
#     2. The 3 most positive sentences and the 3 most negative sentences.
#         2.1 Rank all sentences according to sentiment analysis.
for doctor_id in range(0, len(review_counts)):
    doctor_review_count = int(review_counts[doctor_id])
    # The dentist's topic is highly contaminated by other topics.
    # So I only assign dentist topic to dentists.
    is_dentist = False
    # Detect if someone is a dentist.
    if str(doctor_info.loc[doctor_id]["specialty"]).lower().find("dent") > -1:
        is_dentist = True
        
    # I do NOT keep info about individual reviews. All sentences are stored in a
    # long list regardless of whether they are from the same reviews or not!
    ###########################################################################
    # Build sentence dataframe for the current doctor.
    ###########################################################################
    this_doc = pd.DataFrame(columns = ["sentence", "sentiment", "topic", "topic_score"])
    sent_count = 0

    # For every review.
    for review_id in range(0, doctor_review_count):
        real_review_id = review_id + counter
        review_rating = reviews.loc[real_review_id]["review rating"]
        sentences = tokenize.sent_tokenize(review_main_text[real_review_id])
        
        # For every sentence in each review.
        for sentence in sentences:
            # Assess sentiment.
            sentiment = sia.polarity_scores(sentence)["compound"]
            
            # Assign topic.
            # Default topic to -1.
            this_topic = -1
            # Preprocess sentence.
            sent_tokens = tokenizer.tokenize(str(sentence).lower())
            cleaned_sent = [p_stemmer.stem(i) for i in sent_tokens]
            # Evaluate for topic.
            sent_topics = []
            for mod_id in range(0, mod_num):
                model = ldamodel[mod_id]
                dicti = dictionary[mod_id]
                lda_score = model[dicti.doc2bow(cleaned_sent)]
                for item in lda_score:
                    sent_topics.append((mod_id, item[0], item[1]))
            sent_topics =  sorted(sent_topics, key=lambda x: x[2], reverse=True)
            # Assign the most relevant topic to a sentence only if the topic is more than 70% dominant. 
            if sent_topics[0][2] > 0.7:
                this_topic = topics_matrix[sent_topics[0][0]][sent_topics[0][1]]
            
            # Add procressed sentence and its meta information to the sentence dataframe.
            this_doc.loc[sent_count] = [sentence, sentiment, this_topic, sent_topics[0][2]]
            sent_count += 1
    
    ###########################################################################
    # Compiling results for a doctor.
    ###########################################################################
    # Review highlights.
    # Save the most positive and negative sentiments.
    this_doc2 = this_doc.sort_values(["sentiment"], ascending=[0]).reset_index(drop=True)
    this_doc2 = this_doc2.loc[this_doc2["topic"] != -1].reset_index(drop=True)
    this_doc2 = this_doc2.loc[this_doc2["topic_score"] > 0.5].reset_index(drop=True)
    sent_count_2 = len(this_doc2)
    composite = "NONE"
    # Save the most polarizing sentiments only if there are at least 6 sentences.
    if sent_count_2 > 5:
        sent1 = sent2 = sent3 = sent4 = sent5 = sent6 = ""
        # Only keep positive sentiment if its score is above 0.4 (within [-1, 1]).
        if this_doc2.loc[0]["sentiment"] > 0.4:
            sent1 = this_doc2.loc[0]["sentence"]
        if this_doc2.loc[1]["sentiment"] > 0.4:
            sent2 = this_doc2.loc[1]["sentence"]
        if this_doc2.loc[2]["sentiment"] > 0.4:
            sent3 = this_doc2.loc[2]["sentence"]
        # Only keep positive sentiment if its score is below -0.2 (within [-1, 1]).
        if this_doc2.loc[sent_count_2-1]["sentiment"] < -0.2:
            sent4 = this_doc2.loc[sent_count_2-1]["sentence"]
        if this_doc2.loc[sent_count_2-2]["sentiment"] < -0.2:
            sent5 = this_doc2.loc[sent_count_2-2]["sentence"]
        if this_doc2.loc[sent_count_2-3]["sentiment"] < -0.2:
            sent6 = this_doc2.loc[sent_count_2-3]["sentence"]
        composite = sent1 + "SSEEPP" + sent2 + "SSEEPP" + sent3 + "SSEEPP" + sent4 + "SSEEPP" + sent5 + "SSEEPP" + sent6 + "SSEEPP" + str(sent_count)
    # Add review highlights to the doctor dataframe.
    doctor_info.set_value(doctor_id, "summary", composite)

    # Top topics and their ratings.
    # Ratings are the percent positive sentences belonging to a topic.
    doc_topics = [ [ 0 for i in range(2) ] for j in range(topic_num) ]  # [total count, count positive]
    for index2 in range(0, len(this_doc2)):
        topic_index = this_doc2.loc[index2]["topic"]
        if topic_index != -1:
            doc_topics[topic_index][0] += 1
            topic_sentiment = this_doc2.loc[index2]["sentiment"]
            # A topic sentence if positive if its sentiment is bigger than 0.1.
            if topic_sentiment > 0.1:
                doc_topics[topic_index][1] += 1
    # Do not display dentist stuff for non-dentist
    if not is_dentist:
        doc_topics[3][0] = 0
    # Do not output "positive comment" as a topic. It is non-informative.
    doc_topics[0][0] = 0
    
    # Putting the results into a format to be sparsed by the webapp.
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

    # Print progress.
    print(counter/5088)
    counter += doctor_review_count
    del this_doc
    del this_doc2
    
# Save the updated doctor dataframe containing snapshot information.
doctor_info.to_csv("PATH/result/all_doctors.csv")
