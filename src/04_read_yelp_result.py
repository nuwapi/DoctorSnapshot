#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 22:35:46 2017

@author: nwang
"""

import urllib
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Set up usually lists
# Yelp_SF folder, 178 doctors.
# Index 33 and 165 are anormalis, they have 0 reviews so that side bar review count got picked up. (need to fix this in previous scripts!)
review_counts_SF = ['2', '64', '10', '9', '26', '144', '33', '31', '5', '4', '89', '14', '22', '21', '1', '4', '13', '8', '3', '17', '14', '3', '37', '12', '124', '11', '24', '2', '17', '37', '34', '5', '17', '0', '37', '4', '4', '41', '12', '6', '3', '3', '9', '1', '7', '6', '44', '17', '84', '50', '6', '43', '54', '56', '19', '1', '7', '30', '14', '68', '41', '29', '8', '0', '17', '87', '49', '22', '52', '12', '4', '1', '42', '17', '30', '19', '32', '1', '7', '4', '14', '18', '46', '25', '4', '0', '20', '131', '7', '42', '2', '15', '13', '101', '24', '13', '1', '36', '46', '13', '9', '25', '6', '3', '3', '81', '11', '8', '13', '4', '15', '2', '22', '6', '36', '13', '38', '14', '26', '11', '4', '19', '19', '16', '6', '26', '6', '10', '23', '65', '3', '5', '12', '256', '31', '7', '78', '3', '37', '31', '59', '37', '14', '9', '94', '100', '94', '50', '9', '4', '36', '78', '124', '37', '18', '7', '4', '96', '3', '87', '0', '3', '5', '15', '11', '0', '13', '82', '8', '29', '17', '11', '33', '55', '278', '21', '12', '4']
# Yelp_SDLA folder, 506 doctors.
zeros_SDLA = [205,102,88,495,115,80,501,471,403,22,472,14,480,46,45]  # index+1 !!!
review_counts_SDLA = ['21', '58', '53', '6', '16', '2', 0, 0, 0, '1', 0, '2', 0, '111', '23', '14', '16', '61', '12', '13', '14', '4', '1', '5', '1', '9', '101', '1', '33', '33', '14', '2', '5', '5', '6', '4', '38', '3', '5', '11', '5', '3', '9', '20', '94', '39', '1', '2', '1', '1', '19', '2', '6', '6', '15', '1', '41', '30', '1', '1', '1', '6', '4', '12', '6', '28', '3', '14', '2', '35', '4', '14', '8', '11', '2', '6', '66', '45', '10', '3', '49', '27', '6', '27', '8', '37', '18', '20', '1', '5', '70', '7', '6', '24', '2', '10', '1', '8', '9', '6', '13', '16', '193', 0, '8', '29', '4', '15', '3', '5', '7', '24', '31', '9', '12', '2', '64', '30', '35', '10', '7', '5', '43', '40', '13', '16', '15', '11', '2', '2', '30', '28', '13', '9', '15', '12', '38', '2', '6', '5', '22', '11', '70', '82', '2', '8', '67', '39', '60', '6', '3', '188', '12', '11', '4', '7', '81', '68', '22', '2', '7', '3', '3', '3', '34', '7', '4', '4', '12', '57', '46', '17', '4', '23', '27', '19', '25', '108', '33', '25', '1', '18', '14', '17', '80', '30', '5', '9', '19', '11', '71', '23', '5', '4', '30', '11', '32', '43', '13', '27', '25', '4', '17', '3', '10', '39', '10', '32', '16', '9', '35', '25', '49', '7', '16', '20', '30', '3', '39', '12', '546', '46', '4', '29', '1', '35', '92', '5', '9', '108', '3', '13', '14', '105', '19', '6', '54', '26', '31', '3', '66', '45', '28', '18', '2', '17', '5', '64', '7', '6', '2', '11', '6', '35', '19', '5', '54', '11', '9', '15', '40', '6', '1', '58', '28', '44', '11', '4', '59', '7', '52', '14', '170', '22', '18', '43', '63', '1', '75', '46', '42', '15', '9', '13', '20', '13', '41', '9', '3', '1', '3', '7', '8', '3', '13', '32', '10', '45', '20', '19', '46', '3', '12', '19', '13', '46', '7', '21', '6', '11', '25', '50', '19', '68', '17', '20', '3', '26', '14', '62', '15', '90', '13', '10', '34', '4', '19', '11', '8', '40', '4', '3', '27', '14', 0, '4', '24', '5', '21', '78', '33', '30', '21', '67', '22', '39', '55', '14', '22', '33', '10', '108', '50', '8', '9', '87', '17', '157', '4', '1', '4', '27', '26', '43', '13', '1', '3', '33', '16', '45', '16', '6', '73', '25', '22', '10', '26', '6', '54', '14', '115', '29', '19', '9', '4', '13', '4', '23', '49', '8', '14', '7', '13', '21', '55', '144', '12', '1', '37', '42', '8', '4', '16', '16', '10', '14', '3', '8', '2', '5', '4', '11', '8', '10', '1', '30', '118', '62', '8', '72', '17', '1', '8', '26', '16', '8', '65', '76', '45', '28', '14', '89', '9', '22', '8', '21', '16', '5', '10', '6', '3', '10', '46', '15', '46', '20', '12', '27', '39', '74', '6', '33', '27', '2', '35', '19', '30', '26', '3', '12', '1', '1', '1', '19', '33', '1', '40', '5', '9', '24', '17', '31', '1', '25', '8', '50', '6', '9', '49', '<spanitemprop="Count">20</span>', '7', '24', '89', '22', '7', '7', '16', '23', '48', '1', '10', '2', '4', '1', '37', '21', '3', '48', '5', '17', '149', '5', '6', '6', '1', '13']
for i in zeros_SDLA:
    review_counts_SDLA[i-1] = "0"
for i, item in enumerate(review_counts_SF):
    review_counts_SF[i] = int(review_counts_SF[i])
for i, item in enumerate(review_counts_SDLA):
    review_counts_SDLA[i] = int(review_counts_SDLA[i])
    
# Stats:
# 5088+11705 = 16793 reviews
# 657 doctors.
# 2322327 = 2.3 million words, 2.14 times of the Harry Potter corpus

# Generating file names and the number of reviews in each file.
file_names = []
file_reviews = []
doctor_id = []
#for i, review_count in enumerate(review_counts_SF):
#    if review_count > 0:
#        doc_id = "SF_{0}".format(i+1)
#        max_num_page = int((review_count-1)/20 + 1)     
#        total = 0
#        for j in range(0, max_num_page):
#            name = "file:///home/nwang/proj/data/yelp_SF/" + str(i+1) + "_" + str(j+1) + ".html"
#            file_names.append(name)
#            count_this_page = 20
#            if j == max_num_page - 1 and review_count % 20 != 0:
#                count_this_page = review_count % 20
#            file_reviews.append(count_this_page)
#            doctor_id.append(doc_id)
#            total += count_this_page
for i, review_count in enumerate(review_counts_SDLA):
    if review_count > 0:
        doc_id = "doc_{0}".format(i+1)
        max_num_page = int((review_count-1)/20 + 1)        
        total = 0
        for j in range(0, max_num_page):
            name = "file:///home/nwang/proj/data/yelp_SDLA/" + str(i+1) + "_" + str(j+1) + ".html"
            file_names.append(name)
            count_this_page = 20
            if j == max_num_page - 1 and review_count % 20 != 0:
                count_this_page = review_count % 20
            file_reviews.append(count_this_page)
            doctor_id.append(doc_id)
            total += count_this_page
# Fixing special deletion cases
file_reviews[512] = 2


# Read html back by BeatifulSoup
# review ID
# review main text
# review rating
# review date
# review useful count
# review funny count
# review cool count
# page url
# reviewer ID
# reviewer location
# reviewer friend count
# reviewer review count
# reviewer photo count
count = 0  # The total review count
reviews = pd.DataFrame(columns = ["review main text","review rating","review date","review useful count","review funny count","review cool count","page url","reviewer url","reviewer location","reviewer friend count","reviewer review count","reviewer photo count","doctor ID", "doctor rating"])
for i in range(0, len(file_names)):
    if file_reviews[i] > 0:
        page = urllib.request.urlopen(file_names[i])
        soup = BeautifulSoup(page)
        
        url_loc = soup.find_all("script")[3].text.find("full_url")
        if url_loc == -1:
            url_loc = soup.find_all("script")[4].text.find("full_url")
            page_url0 = soup.find_all("script")[4].text[url_loc:].split("\"")[2]
        else:
            page_url0 = soup.find_all("script")[3].text[url_loc:].split("\"")[2]
        doc_id = doctor_id[i]
        
        doctor_rating = str(soup.find_all("div", class_="biz-rating biz-rating-very-large clearfix")[0].find("img")).split("\"")[1].split(" ")[0]
    
        for j in range(0, file_reviews[i]):
            try:
                review_main_text = soup.find_all("p", itemprop="description")[j].text
                review_rating = float(str(soup.find_all("div", itemprop="reviewRating")[j].find("meta")).split("\"")[1])
                review_date = soup.find_all("div", class_="review-content")[j].find("span").text.split("\n")[1].replace(" ", "")
                review_useful_count = str(soup.find_all("a", class_="ybtn ybtn--small useful js-analytics-click")[j].find_all("span", class_="count")).replace("<", ">").split(">")[2]
                review_funny_count = str(soup.find_all("a", class_="ybtn ybtn--small funny js-analytics-click")[j].find_all("span", class_="count")).replace("<", ">").split(">")[2]
                review_cool_count = str(soup.find_all("a", class_="ybtn ybtn--small cool js-analytics-click")[j].find_all("span", class_="count")).replace("<", ">").split(">")[2]
                page_url = page_url0
                reviewer_url = str(soup.find_all("div", class_="review review--with-sidebar")[j].find("a", class_="js-analytics-click")).split("\"")[5]
                reviewer_location = str(soup.find_all("div", class_="review review--with-sidebar")[j].find("li", class_="user-location responsive-hidden-small").text).replace("\n", "")
                reviewer_friend_count = soup.find_all("div", class_="review review--with-sidebar")[j].find("li", class_="friend-count responsive-small-display-inline-block")
                reviewer_review_count = soup.find_all("div", class_="review review--with-sidebar")[j].find("li", class_="review-count responsive-small-display-inline-block")
                reviewer_photo_count = soup.find_all("div", class_="review review--with-sidebar")[j].find("li", class_="photo-count responsive-small-display-inline-block")
                
                if review_useful_count == "":
                    review_useful_count = 0
                else:
                    review_useful_count = int(review_useful_count)
                if review_funny_count == "":
                    review_funny_count = 0
                else:
                    review_funny_count = int(review_funny_count)
                if review_cool_count == "":
                    review_cool_count = 0
                else:
                    review_cool_count = int(review_cool_count)
                    
                if reviewer_friend_count == None:
                    reviewer_friend_count = 0
                else:
                    reviewer_friend_count = int(str(reviewer_friend_count.text).split(" ")[0].replace("\n", ""))
                if reviewer_review_count == None:
                    reviewer_review_count = 0
                else:
                    reviewer_review_count = int(str(reviewer_review_count.text).split(" ")[0].replace("\n", ""))
                if reviewer_photo_count == None:
                    reviewer_photo_count = 0
                else:
                    reviewer_photo_count = int(str(reviewer_photo_count.text).split(" ")[0].replace("\n", ""))
                    
                reviews.loc[count] = [review_main_text,review_rating,review_date,review_useful_count,review_funny_count,review_cool_count,page_url,reviewer_url,reviewer_location,reviewer_friend_count,reviewer_review_count,reviewer_photo_count, doc_id, doctor_rating]
            except:
                print(i, j, page_url0)
                
            print(count/16793 * 100, "%")
            count += 1

reviews.to_csv("/home/nwang/proj/data/intermediate/all_reviews.csv")