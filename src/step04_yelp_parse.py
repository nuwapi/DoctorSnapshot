#!/usr/bin/env python3
# Created by Nuo Wang.
# Last modified on 8/16/2017.

# Required libraries.
import urllib
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# The review counts here is the list generated from step 3. I just manually copied it over.
review_counts_SF = ['2', '64', '10', '9', '26', '144', '33', '31', '5', '4', '89', '14', '22', '21', '1', '4', '13', '8', '3', '17', '14', '3', '37', '12', '124', '11', '24', '2', '17', '37', '34', '5', '17', '0', '37', '4', '4', '41', '12', '6', '3', '3', '9', '1', '7', '6', '44', '17', '84', '50', '6', '43', '54', '56', '19', '1', '7', '30', '14', '68', '41', '29', '8', '0', '17', '87', '49', '22', '52', '12', '4', '1', '42', '17', '30', '19', '32', '1', '7', '4', '14', '18', '46', '25', '4', '0', '20', '131', '7', '42', '2', '15', '13', '101', '24', '13', '1', '36', '46', '13', '9', '25', '6', '3', '3', '81', '11', '8', '13', '4', '15', '2', '22', '6', '36', '13', '38', '14', '26', '11', '4', '19', '19', '16', '6', '26', '6', '10', '23', '65', '3', '5', '12', '256', '31', '7', '78', '3', '37', '31', '59', '37', '14', '9', '94', '100', '94', '50', '9', '4', '36', '78', '124', '37', '18', '7', '4', '96', '3', '87', '0', '3', '5', '15', '11', '0', '13', '82', '8', '29', '17', '11', '33', '55', '278', '21', '12', '4']
for i, item in enumerate(review_counts_SF):
    review_counts_SF[i] = int(review_counts_SF[i])

# Generate saved Yelp page file names and the number of reviews in each file (between 1 to 20).
file_names = []
file_reviews = []
doctor_id = []    # Doctor ID in my dataset is the index of doctors in review_counts_SF.
for i, review_count in enumerate(review_counts_SF):
    if review_count > 0:
        doc_id = "doc_{0}".format(i+1)
        max_num_page = int((review_count-1)/20 + 1)        
        total = 0
        for j in range(0, max_num_page):
            name = "file:///PATH/data/yelp_pages/" + str(i+1) + "_" + str(j+1) + ".html"
            file_names.append(name)
            count_this_page = 20
            if j == max_num_page - 1 and review_count % 20 != 0:
                count_this_page = review_count % 20
            file_reviews.append(count_this_page)
            doctor_id.append(doc_id)
            total += count_this_page
# Fix special deletion cases manually (after running in to erry).
# It happens that reviews on Yelp are sometimes deleted when I am scrapping and it leads to wrong number of total review counts.
file_reviews[512] = 2

# Read in the html pages using BeatifulSoup.
# There are the fields to be retrieved from the Yelp pages.
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

count = 0
# Initialize the dataframe to save all reviews an their key meta data.
reviews = pd.DataFrame(columns = ["review main text","review rating","review date","review useful count","review funny count","review cool count","page url","reviewer url","reviewer location","reviewer friend count","reviewer review count","reviewer photo count","doctor ID", "doctor rating"])

# Each page first get the above information for the reviews and save them in the review dataframe.
# This section is highly customized and manually designed with respect to the html source code of Yelp.
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

# Save the reviews to the hard drive.
reviews.to_csv("PATH/data/yelp_reviews.csv")
