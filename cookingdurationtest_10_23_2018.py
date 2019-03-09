from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import rolling_median
import datetime
from datetime import date
import time

#from datetime import datetime
#https://stackoverflow.com/questions/30857680/pandas-resampling-error-only-valid-with-datetimeindex-or-periodindex

##ENTER FILENAME HERE
#df = pd.read_csv('downsample_test.csv', header=None, names = ['timestamp', 'temp']).set_index('timestamp')
df = pd.read_csv('81_KPT_test.csv', header=None, names = ['timestamp', 'weight', 'temp', 'amb', 'battery']).set_index('timestamp')


#converting ADC to kg
slope_t = 0.00008318
intcpt_t = 15.27
units_temp = df['temp'] = df['temp'].apply(lambda x:(x*slope_t)+intcpt_t)

#indexing by time, resampling
df.index = pd.to_datetime(df.index, unit = 's')
temps = df.ix[:,['temp']]

##THIS IS GROUPING BY 10 OR 1 MINUTE (10T or 1T), COMMENT OUT (#) TO IGNORE
temp_clean = temps.resample('10T').first()
#temp_clean = temps.resample('1T').first()#group by 10 minute periods and keep only the first value, https://stackoverflow.com/questions/51133531/decimate-data-in-python


#identifying spikes
thresh_lo = 4#-used 4 for FUEL paper#5#7 #10 - try playing around with thresh_lo to capture more or less points
spikes = temp_clean.rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill')
difference = temp_clean-spikes #difference = np.abs(temp_clean-what) #maybe remove abs for positive peaks only

#pd.set_option('display.max_rows', 600) #shows entirety of list in python shell

time_spikes = []
i_outlier= difference > thresh_lo
#print i_outlier
#outlier_temp = i_outlier['temp']
# print outlier_temp

###capture points at startup time -- worry about later 9/26
##for i, row in enumerate(temp_clean_df.values):
##    print i 
##    if i >= 30:
##        print temp_clean_df[i-30]
##        if temp_clean_df[i-30] <= 29 and temp_clean_df[i] > 35: # 29 = high end of amb temp
##            outlier_temp[i] == True
##            #print outlier_temp[i]
          
# print type(outlier_temp)

time_spikes = temp_clean[temp_clean['temp']-spikes['temp'] > thresh_lo].index.values

  
#time_spikes = temp_clean [temp_clean-spikes > thresh_lo].index
#print time_spikes
#print(type(time_spikes[0]))

##for i, row in enumerate (i_outlier.values):
##    if outlier_temp[i] == True:
##        #time_spikes.append(df.index[i]) #want to print the timestamp index in outlier_temp list 
##        time_spikes.append(i)
##        #print df.index[i]
##        #df.index[i].to_csv('downsample.csv')
#print time_spikes #issue number 1: only printing first day 
 
kw = dict(marker = 'o', linestyle='none', color='r', alpha=0.3)

plt.plot(temp_clean)
plt.plot(temp_clean[i_outlier], **kw)
plt.show()


#3 hour groupings 
#time_elapse = []
#event_time = []

time_spikes = pd.Series(time_spikes)
event_dates = sorted(list(set((value.date() for value in time_spikes))))
for day in event_dates:
    # extract time spike data for current day
    current_day = time_spikes[time_spikes.apply(lambda x: x.date()) == day]

    # get all time deltas and ignore NaT
    current_day = current_day.diff().dropna() / np.timedelta64(1, 'h')

    event_times = []
    skip = False
    t_event = 0
    t_low = 3
    t_high = 5

    for loc, dt in enumerate(current_day.values):
        if not skip:
            t_event += dt
            
            if t_low <= t_event <= t_high:
                # current event is at its limit, append it and
                # reset everything for next event
                event_times.append(round(t_event,2))
                t_event = 0
                skip = True

            elif t_event > t_high and t_event != dt:
                # make sure event is recorded if it is below
                # t_low and the next delta would make it be
                # greater than t_high
                val = t_event - dt
                event_times.append(round(val,2))
                t_event = 0

            elif loc == len(current_day.values)-1:
                # make sure last event is recorded even if it is
                # not longer than t_low
                event_times.append(round(t_event,2))

            else:
                # keep summing within the current event
                pass

        else:
            # this step is being skipped
            # make sure next step is counted
            skip = False

    print(day, event_times)
 
##for i, row in enumerate(time_spikes):
##    if i > 0:
##        time_elapsed = time_spikes[i] - time_spikes[i-1] #need to disregard first value or use .diff()       
##        time_elapsed = float(time_elapsed)*2.78e-13 #converts to hours
##        #time_elapsed = time_elapsed.total_seconds()
##        time_elapse.append([time_spikes[i], round(time_elapsed,2)])
##        
##time_elapse_df = pd.DataFrame(list(time_elapse), columns = ['datetime', 'time'])

#print time_elapse_df

##time_elapse_df.to_csv('downsample.csv', index = False, float_format = '%g')
###time_elapse_df = pd.read_csv('downsample.csv', skiprows = 1) #deleting first row 
##time_elapse_df.columns = ['datetime','time']
##diff_time = time_elapse_df['time'] 
##end = time_elapse_df['datetime']
##
###print diff_time
##end_time = []
##groups = diff_time[0]
##
##event_dates = sorted(list(set(time_elapse_df['datetime'].apply(lambda x: x.date()))))
##current_day = 2
##print(event_dates[current_day])
##print(time_elapse_df[time_elapse_df['datetime'].apply(lambda x: x.date()) == event_dates[current_day]]['time'])
##test = time_elapse_df[time_elapse_df['datetime'].apply(lambda x: x.date()) == event_dates[current_day]]['time'][1:].cumsum()
##print(test)
##print(test.floordiv(3))
##print(diff_time[diff_time < 5].index.values)
##
##t_low = 3
##t_high = 5
##
##
##
##for i, row in enumerate(time_elapse_df.values): 
##    if i < len(time_elapse_df)-2:
##        groups = groups + diff_time[i+1] #sum times
##        if t_low <= groups <= t_high: #more than 3 hours
##            event_time.append(groups)
##            groups = diff_time[i+2]
##            i = i+2
##            end_time.append(end[i+1])
##        elif groups > t_high: #and groups<10:
##            groups = groups-diff_time[i+1]
##            event_time.append(groups)
##            groups = diff_time[i+2]
##            i = i+1
##            end_time.append(end[i+1])
##        if diff_time[i] > 12:#signals end of day
##            #groups = groups - diff_time[i+1] #don't include final value in the sum
##            #event_time.append(groups)
##            #groups = diff_time[i+2]
##            groups = diff_time[i+1]
##            i = i+1
##        if groups >=10:
##            groups = groups-diff_time[i+1]
##            event_time.append(groups)
##            groups = diff_time[i+2]
####            #not appending time group so the "event" between two days isn't counted 
##        else:
##            groups = groups

#print event_time
#print end_time
##end_time_df = pd.DataFrame(list(end_time), columns = ['end_time'])
##end_time_df.to_csv('end_time.csv', index = False, float_format = '%g')
##end_time_df.columns = ['end_time']
#print end_time_df
##forreal = end_time_df['end_time'].value_counts()
##pd.set_option('display.max_rows', 600)
#print forreal


##duration_event_df = pd.DataFrame(list(event_time), columns = ['duration'])
#print duration_event_df
#time_elapse_df.to_csv('downsample.csv', index = False, float_format = '%g')
#time_elapse_df.columns = ['time']

##no_events = []
##events = 0
##for i, row in enumerate(duration_event_df.values):
##    if duration_event_df['duration'].iat[i] < 10:
##        events = events+1
##    else:
##        no_events.append(events)
##        events = 0
##        i = i+1
##print no_events #counts number of events per day except for the last day         



##for i, row in enumerate (duration_event_df.values):
##    hours_per_event = round((event_time[i]/hr_to_sec),2)
    #print hours_per_event


##kw = dict(marker = 'o', linestyle='none', color='r', alpha=0.3)
##
##plt.plot(temp_clean,'.')
##plt.plot(temp_clean[i_outlier], **kw)
##plt.show()


##using slope approach
##pos_spike = []
##neg_spike = []
##time_diff = 20 #from '2-T'
##temp_diff = temp_clean.diff()
##der = temp_diff/time_diff
##
##for i, row in enumerate(difference.values): #may need to append a list earlier
##    if difference[i] > thresh_lo:
##        if der[i-1] > 0:
##            pos_spike.append(difference[i])
##    elif difference[i] > thresh_lo:
##        if der[i-1] < 0:
##            neg_spike.append(difference[i])
##    
##print pos_spike
##print neg_spike
