# Created by Nuo Wang.
# Last modified on 8/16/2017.
# BetterDoctor is a database that host a large number of doctor profiles in the US.
# I got my initial round of doctor profiles for doctors in SF, LA and SF through the BetterDoctor API.
# Here I have left out my BetterDoctor API key, if you want to use this code, you'll have to supply your own key.

# Required libraries.
import requests
import json
import numpy as np
import time

'''
The latitudinal and longitudinal dimensions of three California cities:

San Francisco
           top-bot    left-right
top left : 37.805807, -122.513732
top right: 37.805807, -122.351042
bot left : 37.709024, -122.513732
bot right: 37.709024, -122.351042
width : 0.162690, 9.0mi; 0.0045192, 0.25mi
height: 0.096783, 7.5mi; 0.0032261, 0.25mi

San Diego
top left : 32.999479, -117.303937
bot right: 32.662530, -116.972974
width : 0.330963
height: 0.336949

Los Angeles
top left : 34.315607, -118.656211
bot right: 33.927934, -118.150840
width : 0.505371
height: 0.387673
'''

# Here I search doctors on by their geographical location.
# For example, I draw a box around San Francisco and retrieve all of the doctors in that box.
# Better doctor API only return at most 100 search results at a time, so I converted lattitude and longitude into miles
# and search in 0.5mi circles spaced out by 0.5mi.
# Some doctors are repetitively retrieved, and the repetition will be handled in later steps.

# Example for SD doctors.
counter  = 0
# Drawing circles around the box.
for i in np.arange(32.785121799999963, 32.999479+0.0032261, 0.0032261):
    for j in np.arange(-117.303937,-116.972974+0.0045192, 0.0045192):
        # HTML query.
        link = "https://api.betterdoctor.com/2016-03-01/doctors?location={0}%2C{1}%2C0.5&user_location={0}%2C{1}&skip=0&limit=100&user_key=________________________________".format(i, j)
        # Get HTML.
        r = requests.get(link, auth=('user', 'pass'))
        # Get the JSON part in the HTML.
        fname = "PATH/data/better_doctor_SD/{0}_{1}.json".format(i, j)
        # Save the JSON file to local directory for further processing.
        with open(fname,'w') as fp:
            json.dump(r.json(),fp)
        # Comply to the BetterDoctor API access frequency limit.
        time.sleep(1.6)
        
        counter += 1
        print(counter)
        
# Example for LA doctors.
counter = 0
for i in np.arange(33.927934, 34.315607+0.0032261, 0.0032261):
    for j in np.arange(-118.656211,-118.150840+0.0045192, 0.0045192):
        link = "https://api.betterdoctor.com/2016-03-01/doctors?location={0}%2C{1}%2C0.5&user_location={0}%2C{1}&skip=0&limit=100&user_key=________________________________".format(i, j)
        r = requests.get(link, auth=('user', 'pass'))
        fname = "PATH/data/better_doctor_LA/{0}_{1}.json".format(i, j)
        with open(fname,'w') as fp:
            json.dump(r.json(),fp)
        time.sleep(1.6)
        
        counter += 1
        print(counter)
