#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 08:42:46 2020

@author: ShonaCW
"""
import numpy as np
import random
import statistics as stat
        

def Oslo(L, grains, p_a, p_b):
    """
    Function to model the Oslo model for a 1D rice pile, where L is the system
    size, grains is the total number of grains to be added to the system, and
    where p_a and p_b are the probability of the threshhold values having value
    of 1 and 2 respectively. 
    """
    h = 0 #initiate counter, will store the height of the first pile/ site
    counts = [] #a list to keep track of avalanche size per grain addition
    count = 0 #a counter for the number of relaxations per grain addition
    h_l = [] #a list for the height of the first pile after each loop
    crossover = [] #a list, the first input of which will be the crossover time
    av_z = []

    z = np.zeros((L)) #z will contain the current slope at each site
    z_thresh = [] #threshold z for each pile
    for i in np.linspace(1, L, L): #assign random threshold value of 1 or 2
        z_thresh.append(random.choices([1,2], [p_a, p_b])[0]) #[0] so as to just add the number, not a list of the number
    
    
    list_1 = [] #list storing the indices of the sites to relax 
    list_2 = [] #indices of the sites which will need further relaxation

    
    for j in range(grains): #loop until we have added required number of grains
        counts.append(count)
        count = 0 #reset avalanche-size counter to zero when new grain is added
        

        h += 1 #keep track of height of first pile
        z[0] += 1 #update slope of first site
        
        if z[0] > z_thresh[0]:
            list_1.append(0) #store index of first site to list_1 if need relax
            count +=1

        #will only enter loop if the first site needs relaxing, otherwise will
        #add another grain.
        while len(list_1) > 0: #relax all sites indicated in list 1
            for i in list_1:
                #relax the sites 
                
                if i==0:
                    h -= 1
                    z[i] -= 2
                    z[i + 1] += 1
                    
                    #site i definitely relaxed, so update threshold value
                    z_thresh[i] = random.choices([1, 2], [p_a, p_b])[0]
                    
                    #check if surrouding sites require relaxing
                    #if they do, append their index to list 2
                    if z[i] == z_thresh[i] + 1:
                        list_2.append(i)
                        count += 1 
                        
                    if z[i + 1] == z_thresh[i + 1] + 1:
                        list_2.append(i + 1)
                        count +=1
                    
                elif i > 0 and i <= (L - 2):
                    z[i] -= 2
                    z[i - 1] += 1
                    z[i + 1] += 1
                    
                    #site i definitely relaxed, so update threshold value
                    z_thresh[i] = random.choices([1, 2], [p_a, p_b])[0]
                    
                    #check if surrouding sites require relaxing
                    #if they do, append their index to list 2
                    if z[i] == z_thresh[i] + 1:
                        list_2.append(i)
                        count += 1
                        
                    if z[i - 1] == z_thresh[i - 1] + 1:
                        list_2.append(i - 1)
                        count += 1
                    if z[i + 1] == z_thresh[i + 1] + 1:
                        list_2.append(i + 1) 
                        count += 1
                           
                elif i == (L - 1):
                    #store the number of grains which have beem added before
                    #first instance of a grain leaving the system.
                    crossover.append(j + 1)
                    av_z.append(stat.mean(z))
                    
                    z[L - 2] += 1
                    z[L - 1] -= 1
                    
                    #site i definitely relaxed, so update threshold value
                    z_thresh[i] = random.choices([1, 2], [p_a, p_b])[0]
                    
                    #check if surrouding sites require relaxing
                    #if they do, append their index to list 2
                    if z[i] == z_thresh[i] + 1:
                        list_2.append(i)
                        count += 1
                        
                    if z[i - 1] == z_thresh[i - 1] + 1:
                        list_2.append(i - 1)
                        count += 1
                        
            list_1 = list_2 #update list, reqady for next loop iteration
            list_2 = []
        
        h_l.append(h) 
            
    #calculate steady state value using assumption that steady state is reached 
    #once the number of grains added equals the  square of system size
    sec_half = h_l[L**2:] 
    sted_state = stat.mean(sec_half)
    
    cross = crossover[0] #only interested in first instance of a grain leaving
    av_z = av_z[:-20]
    av_z_first = av_z[0] #collect the av slope at first grain leaving
    av_z_later = av_z[-1000:] #collect av slope once definitely in rec. configs.
    return h_l, sted_state, counts, cross, av_z_first, av_z_later
    