#!/usr/bin/env python3
# Created by Nuo wang.
# Last modified on 8/16/2017.

# Required libraries.
import json
import numpy as np
import pickle
import pandas as pd
import random

### Part 1: Find and save unique doctors.

# Variable to hold file names.
file_list = []

# Generate the names of the BetterDoctor JSON files by using lattitudes and longitudes.
# And append the filenames to a list.
# Example for SF.
for i in np.arange(37.709024, 37.805807+0.0032261, 0.0032261):
    for j in np.arange(-122.513732,-122.351042+0.0045192, 0.0045192):
        file_list.append("PATH/data/better_doctor_SF/{0}_{1}.json".format(i, j))

# I actually did not end up using the doctors from LA and SD in DoctorSnapshot's model.
# This is because I made some indexing mistakes in a later step when I scrappped Yelp for their reviews.
# I couldn't scrape Yelp again for a long time because I was banned quickly.
# Example for LA.
# for i in np.arange(33.927934, 34.315607+0.0032261, 0.0032261):
#     for j in np.arange(-118.656211,-118.150840+0.0045192, 0.0045192):
#         file_list.append("PATH/data/better_doctor_LA/{0}_{1}.json".format(i, j))

# Example for SD.
# for i in np.arange(32.785121799999963, 32.999479+0.0032261, 0.0032261):
#     for j in np.arange(-117.303937,-116.972974+0.0045192, 0.0045192):
#             file_list.append("PATH/data/better_doctor_SD/{0}_{1}.json".format(i, j))

# Reading in files one by one.
# Append the doctors to the master_list if that doctor has an associated yelp page in their profile.
master_list = []
# Unique doctor UID on BetterDoctor.
uids = []
for filename in file_list:
    # Try to open a file.
    try: 
        with open(filename,'r') as fp:
            this_file = json.load(fp)["data"]
            if this_file != []:
                # For every doctor in this file.
                for i in range(0, len(this_file)):
                    # Use try here because not all fields exist for all doctors.
                    try:
                        # If Yelp link does not exist, the program will go to except.
                        # If Yelp link exists, append doctor profile and UID to lists.
                        link = this_file[i]['ratings'][1]['provider_url']
                        uids.append(this_file[i]['uid'])
                        master_list.append(this_file[i]);
                    except:
                        pass
        fp.close()
    # Except for when a file doesn't exsit, ignore it.
    except:
        pass

# Get the set of unique doctor UIDs.
# Note that repetitive doctor profiles exist because of the way I retrieved them in step 1.
unique_uids = list(set(uids))

# Deleting duplicated doctor profiles.
# Storing all unique doctor profiles.
unique_master_list = []
# Storing the doctors Yelp links for each doctor in unique_master_list.
yelp_links = []
# A temporary array for unique ID checking. 0 means not yet appended to unique_master_list, 1 means yes.
uid_check = [0]*len(unique_uids)
# For every doctor in the non-unique list.
for index1, doctor in enumerate(master_list):
    # Go through the uniqueness array to see if that doctor has been added.
    for index2, uid in enumerate(unique_uids):
        # If the doctor hasn't been added, add doctor to the unique list and update the checker.
        if doctor["uid"] == uid and uid_check[index2] == 0:
            unique_master_list.append(doctor)
            yelp_links.append(doctor['ratings'][1]['provider_url'])
            uid_check[index2] = 1

# Saving the unique doctors.
with open("PATH/data/yelp_links.pickle", "wb") as fp:
    pickle.dump(yelp_links, fp)    
with open("PATH/data/yelp_docs.json", "w") as fp:
    json.dump(unique_master_list, fp)    


### Part 2: Parse the BetterDoctor JSON files.

# Load the previously saved unique doctor profiles.
with open("PATH/data/yelp_docs.json",'r') as fp:
    doc_n_yelp = json.load(fp)

# Lists to be turned into columns in the final pandas dataframe.
doctor_fn = []             # A doctor's first name.
doctor_ln = []             # A doctor's last name.
doctor_title = []          # A doctor's degree.
doctor_gender = []         # A doctor's gender.
doctor_specialty_list = [] # A doctor's specialty/specialties.
doctor_addr = []           # A doctor's physical address.
doctor_phone = []          # A doctor's phone number.
doctor_lat= []             # The lattitude of a doctor's physical address.
doctor_lon = []            # The longitude of a doctor's physical address.
doctor_website = []        # A doctor's website (not Yelp).
doctor_yelp = []           # A doctor's Yelp link.
doctor_yelp_rating = []    # A doctor's rating on Yelp (missing from BetterDoctor).
doctor_rating1 = []        # A doctor's rating (0-100%) on topic 1 (to be determined).
doctor_rating2 = []        # A doctor's rating (0-100%) on topic 2 (to be determined).
doctor_rating3 = []        # A doctor's rating (0-100%) on topic 3 (to be determined).
doctor_rating4 = []        # A doctor's rating (0-100%) on topic 4 (to be determined).
doctor_rating5 = []        # A doctor's rating (0-100%) on topic 5 (to be determined).
doctor_key_sent = []       # A doctor's review highlights (up to 6 sentences).

# For every unique doctor in the dataset, parse and retrieve for all of the above fields.
# When they don't exist, pass.
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
    doctor_yelp.append(doctor['ratings'][1]['provider_url'])
    doctor_rating1.append(random.randrange(6))
    doctor_rating2.append(random.randrange(6))
    doctor_rating3.append(random.randrange(6))
    doctor_rating4.append(random.randrange(6))
    doctor_rating5.append(random.randrange(6))
    doctor_key_sent.append("Empty doctor smart summary.")
    #print(doctor_name, doctor_title, doctor_gender, doctor_specialty, doctor_addr, yelp_rating, yelp_link)

# Save the parsed doctor information and extra fields (to be filled in in the future).
doctor_info = pd.DataFrame({"first_name": doctor_fn, "last_name": doctor_ln,
                           "title": doctor_title, "gender": doctor_gender,
                           "specialty": doctor_specialty_list, "address": doctor_addr, "phone": doctor_phone,
                           "loc_lat": doctor_lat, "loc_lon": doctor_lon,
                           "website": doctor_website, "yelp": doctor_yelp, "yelp_rating": doctor_yelp_rating,
                           "rating1": doctor_rating1, "rating2": doctor_rating2, "rating3": doctor_rating3,
                           "rating4": doctor_rating4, "rating5": doctor_rating5, "summary": doctor_key_sent})
doctor_info.to_csv("PATH/data/yelp_doctors.csv")
