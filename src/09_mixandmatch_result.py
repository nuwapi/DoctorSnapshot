#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 23:10:30 2017

@author: nwang
"""

import pandas as pd

r1 = pd.read_csv("/home/nwang/proj/heroku/doctor_app5/data/all_doctors.csv")
r2 = pd.read_csv("/home/nwang/proj/heroku/doctor_app5/data/all_doctors2.csv")

for i in range(len(r2)):
    print(i)
    value = r2.loc[i, "summary"]
    print(value)
    r2.loc[i, "summary"] = value
    value = r2.loc[i, "summary"]
    print(value)
    
r2.drop("summary")
r2.drop("Unnamed: 0.1.1")
r2.drop("Unnamed: 0")

r2.to_csv("/home/nwang/proj/heroku/doctor_app/data/all_doctors.csv")

r3 = pd.read_csv("/home/nwang/proj/heroku/doctor_app/data/all_doctors.csv")


