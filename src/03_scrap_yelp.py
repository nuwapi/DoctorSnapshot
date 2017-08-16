#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:35:46 2017

@author: nwang
"""

import pickle

import urllib
from bs4 import BeautifulSoup
import time
import random
import math

# Open the yelp links.
with open ('/home/nwang/proj/data/intermediate/yelp_links.pickle', 'rb') as fp:
    yelp_links = pickle.load(fp)
# SD, LA, doctors with no reviews:
# 334, 495, 501
# For SD, LA
yelp_links[46] = "http://www.yelp.com/biz/marc-k-rubenzik-m-d-san-diego"
yelp_links[102] = "http://www.yelp.com/biz/celebrity-laser-spa-and-surgery-center-los-angeles"
# For SF doctors.
#yelp_links[124] = "http://www.yelp.com/biz/kram-a-jerrold-md-oakland"

# This sections scraps only the front page (first 20 reviews)
review_counts = []
current = 0
for index, url in enumerate(yelp_links[current:]):
    try:
        #page = requests.get(url, auth=('user', 'pass'))
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
        all_no_of_reviews = soup.find_all('span', class_='review-count rating-qualifier')
        no_of_reviews = str(all_no_of_reviews[0]).split("\n")[1].replace(" ", "").replace("reviews", "").replace("review", "")
        review_counts.append(no_of_reviews)
        print(index+current, no_of_reviews)

        #text = str(soup.find_all('script')[6]).replace("<script type=\"application/ld+json\">        ", "").replace("\n</script>", "")
        #j = json.loads(text)
        #with open("/home/nwang/1.json", 'w') as fp:
        #    json.dump(j, fp)
            
        # Save the webpages
        data = str(soup.find_all("html"))
        file = open("/home/nwang/proj/data/yelp_SDLA/{0}.html".format(index+current+1),"w") # open file in binary mode
        file.writelines(data)
        file.close()
        
        random_no = (random.random()-0.5) * 2 * 15
        time.sleep(30+random_no)
    except:
        print("ERROR:", url)
print (review_counts)

# Get review counts again
review_counts = []
review_counts_id = []
for index in range(1, 507):
    try:
        filename = "file:///home/nwang/proj/data/yelp_SDLA/{0}.html".format(str(index))
        page = urllib.request.urlopen(filename)
        soup = BeautifulSoup(page)
        all_no_of_reviews = soup.find_all('span', class_='review-count rating-qualifier')
        no_of_reviews = str(all_no_of_reviews[0]).split("\n")[1].replace(" ", "").replace("reviews", "").replace("review", "")
        review_counts.append(no_of_reviews)
        review_counts_id.append(index)
        print(index)
    except:
        print("ERROR:", index)
    

# Generate links for the rest of the review pages if more than 20 reviews.
extra_links = []
extra_links_index1 = []
extra_links_index2 = []
for i in range(0, len(review_counts)):
    if int(review_counts[i]) > 20:
        extra_no_of_pages = math.floor((int(review_counts[i])-1)/20)
        for j in range(1, extra_no_of_pages+1):
            extra_link = "{0}?start={1}".format(yelp_links[review_counts_id[i]-1], j*20)
            extra_links.append(extra_link)
            extra_links_index1.append(review_counts_id[i])
            extra_links_index2.append(j+1)
print(extra_links)

# Grab the extra links.
# 279_3.html starts to go wrong, that corresponds to index 223.
for i, url in enumerate(extra_links):
    try:
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
            
        # Save the webpages
        data = str(soup.find_all("html"))
        file = open("/home/nwang/proj/data/yelp_SDLA/{0}_{1}.html".format(extra_links_index1[i], extra_links_index2[i]), "w")
        file.writelines(data)
        file.close()
        
        random_no = (random.random()-0.5) * 2 * 20
        time.sleep(40+random_no)
    except:
         print("ERROR:", index)    
    
               
# NOTE: 28 our of 178 docs/practices are not in SF city, according to their yelp link.
'''
http://www.yelp.com/biz/edward-c-sun-md-san-mateo
http://www.yelp.com/biz/wendy-p-liao-dds-alameda
http://www.yelp.com/biz/scotthyver-visioncare-daly-city
http://www.yelp.com/biz/meghan-trojnar-do-los-gatos
http://www.yelp.com/biz/jade-schechter-md-larkspur
http://www.yelp.com/biz/nathan-ehmer-r-do-santa-rosa
http://www.yelp.com/biz/lamont-j-cardon-md-berkeley
http://www.yelp.com/biz/joshua-richards-md-mph-san-ramon
http://www.yelp.com/biz/saito-david-md-mountain-view
http://www.yelp.com/biz/hernandez-raul-md-urology-daly-city
http://www.yelp.com/biz/david-s-chang-md-oakland
http://www.yelp.com/biz/grady-brian-p-md-daly-city
http://www.yelp.com/biz/dr-warren-chang-daly-city
http://www.yelp.com/biz/amanda-z-chen-dds-daly-city
http://www.yelp.com/biz/barzman-anita-j-mountain-view
http://www.yelp.com/biz/sima-stein-md-mountain-view
http://www.yelp.com/biz/sverdlov-dina-md-san-mateo
http://www.yelp.com/biz/jennifer-a-baron-md-san-jose
http://www.yelp.com/biz/mullens-jonah-dpm-redwood-city-2
http://www.yelp.com/biz/juliana-cinque-md-pleasanton-2
http://www.yelp.com/biz/michael-f-dillingham-md-redwood-city
http://www.yelp.com/biz/paul-cortez-md-alamo
http://www.yelp.com/biz/kram-a-jerrold-md-oakland
http://www.yelp.com/biz/flynn-darragh-md-santa-rosa
http://www.yelp.com/biz/xilin-xiang-dds-sunnyvale
http://www.yelp.com/biz/lau-glen-k-md-oakland
http://www.yelp.com/biz/california-skin-institute-mountain-view
http://www.yelp.com/biz/matthew-mynsberge-dds-larkspur
'''