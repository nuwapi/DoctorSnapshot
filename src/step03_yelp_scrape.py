#!/usr/bin/env python3
# Created by Nuo Wang.
# Last modified on 8/16/2017.

# For every doctor in my dataset (of doctors that have a Yelp page), I need to scrape their Yelp reviews.

# Required libraries.
import pickle
import urllib
from bs4 import BeautifulSoup
import time
import random
import math

### Part 1: Scrape Yelp front page.

# Load the pickled Yelp links.
with open ('PATH/data/yelp_links.pickle', 'rb') as fp:
    yelp_links = pickle.load(fp)

# Later I found out that 1 doctor has a wrong Yelp link on BetterDoctor.
# So I am fixing it here manually.
yelp_links[124] = "http://www.yelp.com/biz/kram-a-jerrold-md-oakland"

# This sections scraps only the front page of a doctor's Yelp page.
# This front page contains only up to 20 reviews but a doctor can have more than 20 reviews.
# The review count information is on the front page, getting this number will let me know how many total pages to scrape.
review_counts = []
current = 0
# For every doctor in my dataset.
for index, url in enumerate(yelp_links[current:]):
    try:
        # Get the page from Yelp.
        page = urllib.request.urlopen(url)
        # Turn that into a BeatifulSoup object for parsing.
        soup = BeautifulSoup(page)
        # Find the total number of reviews for a doctor.
        all_no_of_reviews = soup.find_all('span', class_='review-count rating-qualifier')
        no_of_reviews = str(all_no_of_reviews[0]).split("\n")[1].replace(" ", "").replace("reviews", "").replace("review", "")
        review_counts.append(no_of_reviews)
        
        # Print some information for progress monitoring.
        print(index+current, no_of_reviews)
            
        # Save the whole Yelp webpage and parse them later.
        data = str(soup.find_all("html"))
        file = open("PATH/data/yelp_pages/{0}.html".format(index+current+1),"w") # open file in binary mode
        file.writelines(data)
        file.close()
        
        # Wait for some time between each request and add some randomness to it.
        # But it turns out that this is not good enough to by pass Yelp's detector.
        random_no = (random.random()-0.5) * 2 * 15
        time.sleep(30+random_no)
    # If a Yelp link does not exist.
    except:
        print("ERROR:", url)

### Part 2: Scrape the rest of Yelp pages (2nd+) for all reviews.

# Generate webpage links for the rest of the review pages if a doctor has more than 20 reviews.
extra_links = []
extra_links_index1 = []
extra_links_index2 = []
for i in range(0, len(review_counts)):
    if int(review_counts[i]) > 20:
        # Calculate the number of extra pages there are.
        extra_no_of_pages = math.floor((int(review_counts[i])-1)/20)
        for j in range(1, extra_no_of_pages+1):
            # Generate the extra links based on the observed Yelp url patterns.
            extra_link = "{0}?start={1}".format(yelp_links[review_counts_id[i]-1], j*20)
            extra_links.append(extra_link)
            extra_links_index1.append(review_counts_id[i])
            extra_links_index2.append(j+1)
print(extra_links)

# Scrape Yelp for the extra pages.
for i, url in enumerate(extra_links):
    try:
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
            
        # Save the webpages.
        data = str(soup.find_all("html"))
        file = open("PATH/data/yelp_pages/{0}_{1}.html".format(extra_links_index1[i], extra_links_index2[i]), "w")
        file.writelines(data)
        file.close()
        
        # Wait for some time between each request and add some randomness to it.
        # But it turns out that this is not good enough to by pass Yelp's detector.
        random_no = (random.random()-0.5) * 2 * 20
        time.sleep(40+random_no)
    # If a page doesn't exist.
    except:
         print("ERROR:", index)
