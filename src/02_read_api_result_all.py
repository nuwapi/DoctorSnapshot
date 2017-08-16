#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:24:14 2017

@author: nwang
"""

import json
import numpy as np
import pickle
import pandas as pd
import random

# Retrieving file names.
with open("/home/nwang/proj/data/intermediate/better_doctor_json_all.txt") as file:
    content = file.readlines()
file_list = ["/home/nwang/proj/data/" + x.strip() for x in content] 
total_files = len(file_list)

# Reading in files one by one. Append the ones that has an associated yelp link. Not all files exist.
master_list = []
uids = []
for index, filename in enumerate(file_list):
    print(index, total_files)
    try:
        with open(filename,'r') as fp:
            this_file = json.load(fp)["data"]
            if this_file != []:
                for i in range(0, len(this_file)):
                    try:
                        link = this_file[i]['ratings'][1]['provider_url']
                        uids.append(this_file[i]['uid'])
                        master_list.append(this_file[i]);
                    except:
                        pass
        fp.close()
    except:
        pass
unique_uids = list(set(uids))

# Deleting duplicated files.
unique_master_list = []
yelp_links = []
uid_check = [0]*len(unique_uids)
for index1, doctor in enumerate(master_list):
    for index2, uid in enumerate(unique_uids):
        if doctor["uid"] == uid and uid_check[index2] == 0:
            unique_master_list.append(doctor)
            yelp_links.append(doctor['ratings'][1]['provider_url'])
            uid_check[index2] = 1

# Saving.
with open("/home/nwang/proj/data/intermediate/yelp_link.pickle", "wb") as fp:
    pickle.dump(yelp_links, fp)    
with open("/home/nwang/proj/data/intermediate/yelp_docs.json", "w") as fp:
    json.dump(unique_master_list, fp)    


# Obtain doctor full informtion!
with open("/home/nwang/proj/data/intermediate/yelp_docs.json",'r') as fp:
    doc_n_yelp = json.load(fp)

doctor_fn = []
doctor_ln = []
doctor_title = []
doctor_gender = []
doctor_specialty_list = []
doctor_addr = []
doctor_phone = []
doctor_lat= []
doctor_lon = []
doctor_website = []
doctor_yelp = []
doctor_yelp_rating = []
doctor_rating1 = []  # category to be determined
doctor_rating2 = []  # category to be determined
doctor_rating3 = []  # category to be determined
doctor_rating4 = []  # category to be determined
doctor_rating5 = []  # category to be determined
doctor_key_sent = [] # category to be determined

for doctor in doc_n_yelp:
    doctor_fn.append(doctor["profile"]["first_name"])
    doctor_ln.append(doctor["profile"]["last_name"])
    doctor_title.append(doctor["profile"]["title"])
    try:
        doctor_gender.append(doctor["profile"]["gender"])
    except:
        doctor_gender.append("")
    doctor_specialty = ""
    try:
        specialties = doctor["specialties"]
        for i in specialties:
            doctor_specialty += i["name"] + ", "
    except KeyError:
        pass
    doctor_specialty_list.append(doctor_specialty[:-2])
    
    practices = doctor["practices"]
    doctor_st1 = ""
    doctor_st2 = ""
    doctor_city = ""
    doctor_state = ""
    doctor_zip = ""
    for i in practices:
        if i["within_search_area"] == True or i == practices[len(practices)-1]:
            try:
                doctor_website.append(i["website"])
            except KeyError:
                doctor_website.append("")
                
            number = ""
            try:
                for j in i["phones"]:
                    if j["type"] == "landline":
                        number = j["number"]
                        break
            except KeyError:
                pass
            doctor_phone.append(number)
            doctor_lat.append(i["visit_address"]["lat"])
            doctor_lon.append(i["visit_address"]["lon"])
            doctor_st1 = i["visit_address"]["street"]
            try:
                doctor_st2 = i["visit_address"]["street2"]
            except KeyError:
                doctor_st2 = ""
            doctor_city = i["visit_address"]["city"]
            doctor_state = i["visit_address"]["state"]
            doctor_zip = i["visit_address"]["zip"]
            break
    doctor_addr.append(doctor_st1 + " " + doctor_st2 + ", " + doctor_city + ", " + doctor_state + " " + doctor_zip)
    
    doctor_yelp_rating.append(0)
    doctor_yelp.append(doctor['ratings'][1]['provider_url'].split(".com")[1])
    
    doctor_rating1.append(random.randrange(6))
    doctor_rating2.append(random.randrange(6))
    doctor_rating3.append(random.randrange(6))
    doctor_rating4.append(random.randrange(6))
    doctor_rating5.append(random.randrange(6))
    doctor_key_sent.append("Empty doctor smart summary.")
    #print(doctor_name, doctor_title, doctor_gender, doctor_specialty, doctor_addr, yelp_rating, yelp_link)

doctor_info = pd.DataFrame({"first_name": doctor_fn, "last_name": doctor_ln,
                           "title": doctor_title, "gender": doctor_gender,
                           "specialty": doctor_specialty_list, "address": doctor_addr, "phone": doctor_phone,
                           "loc_lat": doctor_lat, "loc_lon": doctor_lon,
                           "website": doctor_website, "yelp": doctor_yelp, "yelp_rating": doctor_yelp_rating,
                           "rating1": doctor_rating1, "rating2": doctor_rating2, "rating3": doctor_rating3,
                           "rating4": doctor_rating4, "rating5": doctor_rating5, "summary": doctor_key_sent})
# Manually correcting wrong entries.
doctor_info.set_value(2, "yelp", "/biz/celebrity-laser-spa-and-surgery-center-los-angeles")
doctor_info.set_value(497, "yelp", "/biz/marc-k-rubenzik-m-d-san-diego")
doctor_info.set_value(544, "yelp", "/biz/parkside-endodontics-san-francisco")
# Save.
doctor_info.to_csv("/home/nwang/proj/data/final/doctor_speadsheet_step1.csv")




 