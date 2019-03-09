from __future__ import division
from collections import defaultdict

import os, sys
import random
import copy
import math
import multiprocessing
import time
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import os

from pandas import datetime
from pandas import rolling_median

new_df = pd.read_csv('371_WOOD_BF_2018.csv', header=None)
new_df.columns = ['datetime', 'unix', 'weight', 'temp'] 


##Data Cleaning
#suppress FutureWarning:
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

##KPT cleaning rolling_median, https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/

thresh_lo = 0.5 #0.2 #played around with this #70000 ADC
new_df['weight_1'] = rolling_median(new_df['weight'], window = 3, center=True).fillna(method='bfill').fillna(method='ffill')
difference = np.abs(new_df['weight']-new_df['weight_1'])

i_outlier= difference > thresh_lo

weight_spikes = new_df[new_df['weight']-new_df['weight_1'] > thresh_lo].index.values

kw = dict(marker = 'o', linestyle='none', color='r', alpha=0.3)

#before cleaning
if len(weight_spikes) >= 1:
    new_df['weight'].plot()
    new_df['weight'][i_outlier].plot(**kw)
    plt.show()

weight_median = rolling_median(new_df['weight'], window = 3, center=True).fillna(method='bfill').fillna(method='ffill')
w_spikeless = [] 
weight_spike = new_df['weight']
weight_std = pd.rolling_std(weight_spike,20)
#weight_smooth = pd.rolling_mean(weight_spike, 20)

for i in range(0, len(new_df['weight'])):
    if difference[i] > thresh_lo: 
        #weight_spike[i] = weight_smooth[i]
        weight_spike[i] = new_df['weight'].iat[i+2] #replacing with a nearby point
        w_spikeless.append(weight_spike[i]) 
    else:
        weight_spike[i] = weight_spike[i]
        w_spikeless.append(weight_spike[i])
new_df['weight'] = w_spikeless

#after cleaning
if len(weight_spikes) >= 1:
    plt.plot(new_df['unix'], w_spikeless)
    plt.show()


days = new_df['unix']
new_df.to_csv('new_practice_data.csv', index=False) 
##convert unix time to date
new_df = pd.read_csv('new_practice_data.csv')
new_df['unix'] = pd.to_datetime(new_df['unix'], unit ='s')
new_df.to_csv('new_practice_data.csv') 

diff = new_df['weight'].diff()

#pd.set_option('display.max_rows', 600) #to show all rows of printed array in Python shell

##weight change
delta_pos = 1 #kg threshold- see 9_short.xlsx
delta_neg = -0.2 #-0.1 #-0.2 #-0.1 #kg threshold
add = []
time_add = []
add_and_time = []
take = []
time_take = []
time = new_df['unix']
take_verify = []
time_take_verify = []
##temp change
delta_pos_temp = 5 #deg C, determine threshold
temps = [] #list of negative weights, verified by temps

time_temp = []
temp_day = []
time_day = []

##weight change
for i, row in enumerate(new_df.values):  
    if diff[i]>=delta_pos: #weight added
        diff[i] = '{0:.2f}'.format(diff[i]) 
        add.append(diff[i])
        time_add.append(time[i])
        add_and_time.append([time[i], round(diff[i], 3)]) 

    if diff[i] <= delta_neg: 
        diff[i] = '{0:.2f}'.format(diff[i])
        take.append(diff[i]) 
        time_take.append(time[i])
        #print ([time_take, take])

#verify weight taken, keep this list separate from take list bc we may want to compare
    data_pts = 80 #25 #played around with this value, number of data points it takes to detect a significant change in temp at the start of a cooking event, could also do it based on time 
    amb = 0 #based on Burkina Faso data 
    temp = new_df['temp']
    diff_temp = new_df['temp']-amb 

#accounting for temp
    if diff[i]<=delta_neg and diff_temp[i] >= delta_pos_temp or diff[i] <= delta_neg and diff_temp[i+data_pts] >= delta_pos_temp: #second part accounting for initial wood taken out/heat up time at start of cooking
        take_verify.append(diff[i]) 
        time_take_verify.append(time[i])
        temps.append([time[i], round(diff[i],3)]) #list of time, weight (verified by temperature)
        #print take_verify

total_weight = round(abs(sum(take_verify)),2) #sum to compare to KPT total data
print 'total weight:', total_weight, 'kg'

##to store data in Excel:
##list of added wood, separated by day
add_df = pd.DataFrame(list(zip(time_add, add)), columns = ['unix', 'add'])
add_df.index = pd.to_datetime(add_df['unix'], unit = 's') 
add_df= add_df.resample('D').sum()
add_df.to_csv('add_practice_data.csv', index=False, float_format = '%g') 

##list of wood taken, separated by day
take_verify_df = pd.DataFrame(list(zip(time_take, take_verify)), columns = ['time_take', 'take_verify'])   
take_verify_df.index = pd.to_datetime(take_verify_df['time_take'], unit = 's')
take_verify_df = take_verify_df.resample('D').sum()

take_verify_df.to_csv('take_practice_data.csv', index=False, float_format = '%g') 
print take_verify_df
take_verify_df.columns = ['total weight']
#take_verify_df.columns = ['day', 'total weight']

 
#use is defined as a day that weight is being taken (not number of times, just that it happens at all), concurrent with Wilson 2016 article 
total_days = len(take_verify_df['total weight'])
take_verify_df.dropna(axis=0, how='any', inplace=True)  
days_used = take_verify_df['total weight']

#Temperature Corroboration: if temp change but no weight change (per day)
for i, row in enumerate(new_df.values): #checking that a cooking event occurred that day 
    if diff_temp[i] > delta_pos_temp:
        temp_day.append(diff_temp[i])
        time_day.append(time[i])
        
temp_check_df = pd.DataFrame(list(zip(time_day, temp_day)), columns = ['time_day', 'temp_day']).set_index('time_day')
temp_check_df.index = pd.to_datetime(temp_check_df.index, unit = 's')
temps = temp_check_df.ix[:,['temp_day']]
temp_check_df = temps.resample('D').first()
temp_day = temp_check_df['temp_day']

temp_day.index #days where there was cooking activity -- want days where there was *not* cooking activity 

##checking for days where cooking occurred but no weight change was detected
for i, row in enumerate (temp_check_df.values):
    if temp_day[i] > 0:
        if len(temp_day) != len(days_used):
            var = 0
        else:
            var = 1
    else:
        var = 1
if var == 0:
    print 'error - cooking event recorded with no weight change' #, ',', temp_day.index - want days that are NOT in temp_day.index


total_days = len(temp_day)
days_used = take_verify_df['total weight']
days_use = len(days_used)
percent_days_use = (days_use/total_days)*100
ave_use = sum(days_used)/(len(days_used))
#print percent_days_use,"%"
#print days_use


###identify and remove outlier days with high weight - probably not necessary here
##w_spikeless = []
##w_threshold = -16.2 #based off small data sample
##count = 0
##w = take_verify_df['total weight'] #should be list of wood use per day
##for i in range(0, len(w)):
##    if w[i] < w_threshold: #if daily wood use is more than threshold
##        w[i] = nan
##        count = count+1 
##    else:
##        w[i] = w[i] #otherwise, wood weight stays the same 
###w.replace('-', np.nan) #replaces the 0 with nan so it doesn't count towards average 
##ave_use = sum(w)/(len(w)-count)






    



