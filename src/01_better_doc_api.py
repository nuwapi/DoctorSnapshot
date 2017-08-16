# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
import json
import numpy as np
import time

# San Francisco
#            top-bot    left-right
# top left : 37.805807, -122.513732
# top right: 37.805807, -122.351042
# bot left : 37.709024, -122.513732
# bot right: 37.709024, -122.351042
# width : 0.162690, 9.0mi; 0.0045192, 0.25mi
# height: 0.096783, 7.5mi; 0.0032261, 0.25mi

# San Diego
# top left : 32.999479, -117.303937
# bot right: 32.662530, -116.972974
# width : 0.330963
# height: 0.336949

# Los Angeles
# top left : 34.315607, -118.656211
# bot right: 33.927934, -118.150840
# width : 0.505371
# height: 0.387673

# For SD.
counter  = 0
#for i in np.arange(32.662530, 32.999479+0.0032261, 0.0032261):
for i in np.arange(32.785121799999963, 32.999479+0.0032261, 0.0032261):
    for j in np.arange(-117.303937,-116.972974+0.0045192, 0.0045192):
        link = "https://api.betterdoctor.com/2016-03-01/doctors?location={0}%2C{1}%2C0.5&user_location={0}%2C{1}&skip=0&limit=100&user_key=39a7c36427d4f05c14c96c607fff5147".format(i, j)
        r = requests.get(link, auth=('user', 'pass'))
        fname = "/home/nwang/proj/data/better_doctor_SD/{0}_{1}.json".format(i, j)
        with open(fname,'w') as fp:
            json.dump(r.json(),fp)
        time.sleep(1.6)
        
        counter += 1
        print(counter)
        
# For LA.
counter = 0
for i in np.arange(33.927934, 34.315607+0.0032261, 0.0032261):
    for j in np.arange(-118.656211,-118.150840+0.0045192, 0.0045192):
        link = "https://api.betterdoctor.com/2016-03-01/doctors?location={0}%2C{1}%2C0.5&user_location={0}%2C{1}&skip=0&limit=100&user_key=6257ab18d7cb76e204ebec1bdebaff39".format(i, j)
        r = requests.get(link, auth=('user', 'pass'))
        fname = "/home/nwang/proj/data/better_doctor_LA/{0}_{1}.json".format(i, j)
        with open(fname,'w') as fp:
            json.dump(r.json(),fp)
        time.sleep(1.6)
        
        counter += 1
        print(counter)

# For SF, over-written.... >.<



