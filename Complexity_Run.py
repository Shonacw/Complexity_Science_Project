#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:07:05 2020

@author: ShonaCW
"""
#Import all neccessary modules
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
from heapq import nsmallest

import Complexity as C
import logbin230119 as logbin

#%%
"""TESTING STAGE"""
#Collect data using written Oslo model for L=100 and 200,000 grain additions
h_l, sted_state, counts, cross, av_z_first, av_z_later = C.Oslo(100, 200000, 0.5, 0.5)

"""HEIGHTS VS. GRAIN NUMBER FOR PROBABILITY p=1 """
plt.figure()
plt.xlabel('Grains Added')
plt.ylabel('Height of first pile')
plt.title('Preliminary testing for L=100, p=1')
plt.grid()
plt.plot(h_l)
ss = round(sted_state, 3)
plt.plot([0,200000], [ss, ss], '--', label=f'steady state height (L=100) ={ss}')

#collect values for Table.1
print('steady state height', ss) 
print('crossover time', cross-1)

"""AVALANCHE SIZES OVER TIME"""
plt.figure()
plt.plot(counts) #'counts' contains list of observed avalanche sizes
plt.xlabel('Grain addition')
plt.ylabel('Avalanche size')
plt.title('System Size L=100 with p=1')
plt.show()
#%%
"""STABLE CONFIGURATION MODEL"""
x = np.linspace(1,100,100)
y = np.linspace(100,1,100)
plt.bar(x, y)
plt.xlabel('i')
plt.ylabel(r'$h_i$')
plt.title('Stable Configuration for L=100, p=1')
plt.show()

#%%
"""DATA COLLECTION"""
"""Averaging, with p=0.5"""
#list of system sizes L, and the colours I will use to plot their data
sizes = [[4, 'b'], [8, 'm'], [16, 'c'], [32, 'k'], [64, 'g'], [128, 'y'], [256, 'r']]

#set up all lists that will be appended to in the data collection process
heights = []
counts_list = []
crossover_list = []
h_ll_list = []
av_z_list_mid = []
av_z_list = []
ss_av_list = []
av_z_first_list = []
av_z_later_list = []

#specify total number of grains to be added for each length of system
#written so during testing I can quickly change between max size of system
if sizes[-1][0] == 32:
    leng = 2000
if sizes[-1][0] == 64:
    leng = 5000
if sizes[-1][0] == 128:
    leng = 20000
if sizes[-1][0] == 256:
    leng = 80000

#Run simulation
for i in sizes:
    counts_list_mid = []
    crossover_list_mid = []
    ss_mid = []
    heights_mid = []
    av_z_first_mid = []
    av_z_later_mid = []
        
    for k in range(100): #repeat 100 times so can then take averages
        h_l, ss, counts, crossov, av_z_first, av_z_later = C.Oslo(i[0], leng, 0.5, 0.5)
        heights_mid.append(h_l)
        round(ss, 3) #round steady state value to 3 decimal places
        #add all the information found in each loop to lists
        ss_mid.append(ss)
        counts_list_mid.append(counts)
        crossover_list_mid.append(crossov)
        av_z_first_mid.append(av_z_first)
        av_z_later_mid.append(av_z_later)
    
    #calculate the mean of the stored information, and store in new list
    #before repeating process for next system size
    counts_list.append(np.mean(counts_list_mid, axis=0))
    crossover_list.append(stat.mean(crossover_list_mid))

    print('DONE') #to keep track of progress
    ss_av = stat.mean(ss_mid)
    ss_av_list.append(ss_av)
    h_ll = np.mean(heights_mid, axis = 0) #taking average of lists
    h_ll_list.append(h_ll)
    av_z_first_list.append(np.mean(av_z_first_mid))
    av_zz_later = np.mean(av_z_later_mid, axis = 0) #taking average of lists
    av_z_later_list.append(av_zz_later)


#store information in files for further graph plotting etc.
h_ll_list = np.array(h_ll_list).T
np.savetxt('heights.csv', h_ll_list, delimiter=',')
np.savetxt('crossover.csv', crossover_list, delimiter=',')
np.savetxt('avalanches.csv', counts_list, delimiter=',')

av_z_later_list_1 = np.array(av_z_later_list)
np.savetxt('av_z_later_12.csv', av_z_later_list_1, delimiter=',')
np.savetxt('av_z_first_12.csv', av_z_first_list, delimiter = ',')
np.savetxt('ss_av_12.csv', ss_av_list, delimiter=',')

#gather L values into a list 'p' for ease in rest of code
p=[] 
for k in sizes:
    p.append(k[0])

#%%
"""SYSTEM HEIGHT OVER TIME PLOT"""
#load required data into an array
heights = np.loadtxt('heights_12.csv', delimiter=',').T
ss_av_list = np.loadtxt('ss_av_12.csv', delimiter=',')
plt.figure(0)
plt.xlabel('t')
plt.ylabel('h(t;L)')
for i in range(len(heights)): #loop through data collected for each system size
    ss_av = ss_av_list[i]
    h_ll = heights[i]
    plt.plot(h_ll, color=sizes[i][1], label= f'L={sizes[i][0]}')
    plt.plot([0,leng], [ss_av, ss_av], '--', color=sizes[i][1], linewidth=1.1)
plt.yscale('log')
plt.title(r'System Height Over Time For System Size $L$')
plt.legend(prop={"size":8}, loc="lower right")
plt.show()

#%%
"""AVALANCHE SIZE HISTOGRAMS"""
#plotting the avalanche sizes over time, i.e. the size of avalanche induced 
#for each new grain added.

#collect required data
counts_list = np.loadtxt('avalanches_12.csv', delimiter=',')

for j in range(len(sizes)-1): #loop through data collected for each system size
    plt.figure(j+ 30)
    plt.plot(counts_list[j])
    plt.xlabel('Grain addition, t')
    plt.ylabel('Avalanche size')
    plt.title(f'Avalalche Sizes: System Size L={sizes[j][0]}')
    plt.show()
#%%
"""CROSSOVER TIME ANALYSIS: PROOF OF QUADRATIC DEPENDENCE ON L"""
#collect required data, place in a list
crossover_list = np.loadtxt('crossover_12.csv', delimiter=',').tolist()
first_av_z = np.loadtxt('av_z_first_12.csv', delimiter=',').tolist()
    
plt.figure(1)
ppp = np.poly1d(np.polyfit(p, crossover_list, 3))
t = np.linspace(0, 256, 50)
plt.plot(p, crossover_list, 'o', label='Measured Data')
plt.plot( t, ppp(t), '-', label='Polyfit')
plt.xlabel('L')
plt.ylabel(r'$\langle{t_{c}(L)}\rangle$')
plt.title('Cross-Over Time ')
plt.legend()
plt.show()

#%%
"""PLOT OF REAL VS. THEORETICAL CROSSOVER TIMES"""
#collect required data
later_avz = np.loadtxt('av_z_later_12.csv', delimiter=',')

#calculate values of "\langle z \langle _ext"
av_zz = []
for i in range(7):
    av_zz.append(np.mean(later_avz[i]))

    
#firstly, use average slope at moment first grain leaves system
t_av_1 = [] #expected crossover time for each system size
t_1 = []

#then, use a mean of averages z's taken from further on during the steady state
#at this point, the system will definitely be in a set of recurrent configs.
t_av_2 = [] #expected crossover time for each system size
t_2 = []

for i in range(7): #loop through data collected for each system size
    #'crossover_list' is my measured set of crossover times
    t_av_1.append(first_av_z[i] * 0.5 * p[i]**2 *(1 + 1/p[i])) #Using Eq. 11
    t_1.append(crossover_list[i]/ t_av_1[i])
    
    t_av_2.append(av_zz[i] * 0.5 * p[i]**2 *(1 + 1/p[i])) #Using Eq. 11
    t_2.append(crossover_list[i]/ t_av_2[i])
    
#plot ratio of the measured crossover time to the theoretical crossover time
plt.figure(2)
plt.plot(p, t_1, 'x-', label = r'Using $\langle z \rangle_{0}$')
plt.plot(p, t_2, 'x-', label = r'Using $\langle z \rangle_{ext}$')
plt.plot(np.linspace(4,258,2), [1, 1], '--', color='k')
plt.xlim(4,258)
plt.xlabel('L')
plt.ylabel(r'$\langle{t_{c}}\rangle / \langle{t_{c}^{theory}}\rangle$')
plt.title(r'Ratio of $\langle{t_{c}}\rangle$ with $\langle{t_{c}^{theory}}\rangle$')
plt.legend()
plt.show()

#%% 
""" HEIGHT DATA COLLAPSE"""
#collect stored data
heights = np.loadtxt('heights_12.csv', delimiter=',').T
crossover = np.loadtxt('crossover_12.csv', delimiter=',')

plt.figure(3, figsize=(10,5))
x = np.linspace(1, 80001, 80000) 
L = [4, 8, 16, 32, 64, 128, 256] 
for i in range(7): #loop through data collected for each system size
    xx = x/crossover[i] 
    yy = heights[i] / L[i]
    plt.plot(xx, yy, color=sizes[i][1], label=f'L= {L[i]}') 
    coef_coll, cov_coll = np.polyfit(np.log(xx[:100]), np.log(yy[:100]), 1, cov=True) 
grad_coll = coef_coll[0]
print('Gradient Linear Region Using L=256:', grad_coll)
print('Error on Gradient:', cov_coll[1][1])
plt.xlim(1.63e-05, 5589)
plt.ylim(0.0032, 1.9)
plt.xlabel(r'$t / L^2$')
plt.ylabel('h(t;L) / L')
plt.xscale('log')
plt.yscale('log')
plt.title('Height Data Collapse')
plt.legend(loc='lower right')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""DATA COLLECTION"""
""" No Averaging, p=0.5"""
#set up all lists that will be appended to in the data collection process
heights = []
counts_list = []
crossover_list = []
h_ll_list = []
av_z_list = []
sted_state_val = []
av_z_first_list = []
av_z_later_list = []

#specify total number of grains to be added for each length of system
#written so during testing I can quickly change between max size of system
if sizes[-1][0] == 32:
    leng = 2000
if sizes[-1][0] == 64:
    leng = 5000
if sizes[-1][0] == 128:
    leng = 20000
if sizes[-1][0] == 256:
    leng = 5000000

#collect data and add the prepared lists
for i in sizes:
    h_l, ss, counts, crossov, av_z_first, av_z_later = C.Oslo(i[0], leng, 0.5, 0.5)
    heights.append(h_l)
    round(ss, 3) #round steady state value to 3 decimal places
    sted_state_val.append(ss)
    counts_list.append(counts)
    crossover_list.append(crossov)
    av_z_first_list.append(av_z_first)
    av_z_later_list.append(av_z_later)
    print('DONE') #to keep track of progress

#store collected data
np.savetxt('heights_nonav_256.csv', heights, delimiter=',')
np.savetxt('crossover_nonav_256.csv', crossover_list, delimiter=',')
np.savetxt('stedstateval_256.csv', sted_state_val, delimiter=',')
np.savetxt('avalanche_sizes_256.csv', counts_list, delimiter=',')
np.savetxt('av_z_256_first.csv', av_z_first_list, delimiter=',')
np.save_txt('av_z_256_later.csv', av_z_later_list, delimiter=',')

#%%
"""PROBABILITIES GRAPH"""   
#collect data
heights_nonav = np.loadtxt('heights_nonav_256.csv', delimiter=',')
sted_state_val = np.loadtxt('stedstateval_256.csv', delimiter=',')

#set up plot of measured height probability P(h; L) vs. h
standard = [] #will have gaussian shape, so will need to collect the sigma
fig4 = plt.figure(4, figsize=(10,5))
ax4 = fig4.add_subplot(1,1,1)

#set up plot for probability data collapse
fig5 = plt.figure(5, figsize=(10,5))
ax5 = fig5.add_subplot(1,1,1)

#function/ data for plotting expected gaussian in the data collapse
def gaussian(X, Mu, Sigma):
    return np.exp(-np.power(X - Mu, 2.) / (2 * np.power(Sigma, 2.)))
vals = np.linspace(-4, 4, 100)
Gauss = gaussian(vals, 0, 1)

#collect and manipulate information from stored data set "heights"
for i in range(len(sizes)):
    #only look at last 4000,000 entries
    steady_state_heights = list(heights_nonav[i][-4000000:])
    standard.append(np.std(steady_state_heights))
    possible_heights = np.arange(min(steady_state_heights), max(steady_state_heights)+1)
    Prob = [steady_state_heights.count(height)/4000000 for height in possible_heights]
    
    #plot probabilities
    ax4.plot(possible_heights, Prob, color=sizes[i][1], label= f'L={sizes[i][0]}')
    
    #plot data collapse
    pos_height = [(HHH - sted_state_val[i])/standard[i] for HHH in possible_heights]
    prob = [PPP * np.sqrt(2 * np.pi) * standard[i] for PPP in Prob]
    ax5.plot(pos_height, prob, color=sizes[i][1], label= f'L={sizes[i][0]}')

#plot measured height probability P(h; L) vs. h for various system sizes
ax4.set_xlabel('h')
ax4.set_ylabel('P(h; L)')
ax4.set_title('Height Probability Distribution')
ax4.legend()

#plot the data collapse for various system sizes of probability P(h; L)
ax5.plot(vals, Gauss, '--', color='k', label = 'Normalised')
ax5.set_xlabel(r'(h-$\mu$)/$\sigma_{h}$')
ax5.set_ylabel(r'P(h;L) $\sigma_{h}$')
ax5.set_title('Collapsed Height Probability Distribution')
ax5.legend()

#%%
"""STANDARD DEVIATION_h"""
fig6 = plt.figure(6, figsize=(10,5))
ax6 = fig6.add_subplot(1,1,1)
coef_6, cov_6 = np.polyfit(np.log(p[1:]), np.log(standard[1:]), 1, cov=True)
info = np.polyfit(np.log(p), np.log(standard), 1, full=True)
print('SSR of slope:', info[1][0])
print('Error of slope:', cov_6[0])
poly1d_6 = np.poly1d(coef_6)
ax6.plot(p, standard, 'x', label = r'$\sigma_h(L)$ = $(\langle h^2(t;L)\rangle$-$\langle h(t;L)\rangle^2)^\frac{1}{2}$')
ax6.plot(p, np.exp(poly1d_6(np.log(p))), label=f'log($\sigma_h(L)$) = {round(coef_6[0], 3)}ln(L) {round(coef_6[1], 3)}')
ax6.set_xlabel('L')
ax6.set_ylabel(r'$\sigma_{h}$(L)')
ax6.set_title('Standard Deviation')
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_title(r'Standard Deviation $\sigma_h(L)$')
ax6.legend()


"""ERROR ON SIGMA_H"""
#Slope calculation using data for all L values
GRAD_ALL = np.polyfit(np.log(p), np.log(standard), 1)[0]
#Slope calculation using data for only greatest L values
GRAD_4 = np.polyfit(np.log(p[5:]), np.log(standard[5:]), 1)[0]
#Errir is half the difference between these two slopes
Error_sigma_H = (GRAD_ALL - GRAD_4)/2
print('Error on Sigma H', Error_sigma_H)
#%%
"""STANDARD DEVIATION_z OVER L"""
#collect standard deviation for each system size and divide by respective L
standard_L = []
for i in range(len(sizes)):
    standard_L.append( standard[i] / p[i])

fig8 = plt.figure(8, figsize=(10,5))
ax8 = fig8.add_subplot(1,1,1)
coef_8 = np.polyfit(np.log(p), np.log(standard_L), 1)
grad = coef_8[0]
poly1d_fn_1 = np.poly1d(coef_8) 
ax8.plot(p, standard_L, 'x', label = r'$\sigma_z(L)/L$ = $\frac{1}{2}(\langle h^2(t;L)\rangle$-$\langle h(t;L)\rangle^2)^\frac{1}{2}$')
ax8.plot(p, np.exp(poly1d_fn_1(np.log(p))), '-', label=f'log($\sigma_z(L)$) = {round(grad, 3)}ln(L) {round(coef_8[1], 3)}')
ax8.set_xscale('log')
ax8.set_yscale('log')
ax8.set_xlabel('L')
ax8.set_ylabel(r'$\sigma_{z}(L) / L$')
ax8.set_title(r'Standard Deviation $\sigma_z(L)$')
ax8.legend()

 #%%
#collect required data
sted_state_val = np.loadtxt('stedstateval_256.csv', delimiter=',').T
sted_state_val = sted_state_val.tolist()  

"""GRAPH TO FIND a_0: AVERAGE HEIGHT VS. SYSTEM SIZE"""
plt.figure(9, figsize=(10,5))

#only calculate gradient between L=64 and L=256 (i.e. largest system sizes)
coef_ss = np.polyfit(p[4:], sted_state_val[4:], 1)
plt.plot(p, sted_state_val, 'x-', label=f'Polyfit Slope $a_{0}$= 1.727')
plt.plot([64,64], [40,170], '--', color='r', linewidth=3)
plt.plot([256,256], [370, 500], '--', color='r', linewidth=3, label=r'Region used for a_0 calculation')
plt.xlabel('L')
plt.ylabel(r'$ \langle h(L)\rangle_t $')
plt.title(r'Graph to find $a_{0}$ gradient')
plt.legend()
plt.show()


"""MEASURED SCALED HEIGHT USING INITIAL ESTIMATE FOR a_0"""
manipulated = []
for i in range(len(p)):
    manipulated.append(1 - (sted_state_val[i]/(coef_ss[0] * p[i])))

log_l = [np.log(i) for i in p]
log_manipulated= [np.log(i) for i in manipulated]

plt.figure(10, figsize=(10,5))
POL1 = np.polyfit(log_l, log_manipulated, 1, full=True)
POL_11 = np.polyfit(log_l, log_manipulated, 1)
ppp = np.poly1d(POL_11)
t = np.linspace(min(log_l), max(log_l), 50)
plt.plot(log_l, log_manipulated, 'x-', label='Measured Scaled Height')
plt.plot(t, ppp(t), '-', label = f'Linear Fit of Data: {round(ppp[1], 3)}ln(L) + {round(ppp[0], 4)}')
plt.xlabel('Log(L)')
plt.ylabel(r'$Log\left( 1 - \frac{\langle h(L)\rangle_{t}}{a_0L}\right)$')
plt.title(r'Measured Scaled Height, $a_{0}$= 1.727')
plt.legend()
plt.show()

#%%
"""MINIMISING THE RESIDUAL"""
#function to check values surrouding inital estimate for a_0
#the Sum of Squared Residuals (SSR) is minimised to find a better a_0 estimate
def Min_Res():
    guesses = np.linspace(1.725,1.770,36) #check values either side of 1.727
    residuals = []
    log_l = [np.log(i) for i in p] 
    
    for guess in guesses: #run loop/ collect data associated with each guess
        man_1 = [] #collect 'manipulated' values, so can define a polyfit
        for i in range(len(p)): #where p is still the list of system sizes
            man_1.append(1 - (sted_state_val[i]/(guess * p[i])))
        
        log_manipulated_1 = [np.log(i) for i in man_1]
        res = np.polyfit(log_l, log_manipulated_1, 1, full=True)[1][0]
        
        #collect the guesses with their associated SSR value in a list
        residuals.append(guess)
        residuals.append(res)
    
    
    best_residual = min(residuals[1::2]) #only look at even indices: SSR values
    best_estimate_ind = residuals.index(best_residual) #find index of min SSR
    best_estimate = residuals[best_estimate_ind-1] #find associated guess value
    
    #Error Calculation:
    #find the a0 value which gives the next-smallest residual
    second_smallest = nsmallest(2, residuals[1::2])[1]
    #take the difference in a0 values (with the smallest) as the eroor
    error = residuals[residuals.index(second_smallest)-1] - best_estimate
    return best_estimate, best_residual, error

estimate = coef_ss[0]
best_est = Min_Res()
print('best estimate:', best_est[0], 'best residual:', best_est[1], 
      'error on best estimate:', round(best_est[2], 4))


#%%
"""MEASURED SCALED HEIGHT USING UPDATED (SSR-MINIMISED) ESTIMATE FOR a_0"""
#EXPLANATION OF ERROR CALCULATION:
#Add/ Subtract the error on a_0 from a_0 and evaluate Omega_Est from both
#Will then take the differences between Omega_Est's as the error on Omega
A_0_1 = best_est[0]-round(best_est[2], 4)
A_0_2 = best_est[0]+round(best_est[2], 4)

manipulated_1 = []
Man_Error_1 = []
Man_Error_2= []
for i in range(len(p)):
    manipulated_1.append(1 - (sted_state_val[i]/(best_est[0] * p[i])))
    Man_Error_1.append(1 - (sted_state_val[i]/(A_0_1 * p[i])))
    Man_Error_2.append(1 - (sted_state_val[i]/(A_0_2 * p[i])))

log_manipulated_1 = [np.log(i) for i in manipulated_1]
plt.figure(11, figsize=(10,5))
POL, cov = np.polyfit(log_l, log_manipulated_1, 1, cov=True)
ppp_1 = np.poly1d(POL)
t_1 = np.linspace(min(log_l), max(log_l), 50)
plt.plot(log_l, log_manipulated_1, 'x-', label='Measured Scaled Height')
plt.plot(t_1, ppp_1(t), '-', label = f'Linear Fit of Data: -0.530ln(L){round(ppp_1[0], 3)}') 
plt.xlabel('Log(L)')
plt.ylabel(r'$Log\left( 1 - \frac{\langle h(L)\rangle_{t}}{a_0L}\right)$')
plt.title(r'Measured Scaled Height, $a_{0}$ = 1.739')
plt.legend()
plt.show()
print('Error on gradient from fit', cov[0][0])
print('Error on intercept from fit', cov[1][1])

"""Error on Omega Calculation"""
log_Man_1 = [np.log(i) for i in Man_Error_1]
log_Man_2 = [np.log(i) for i in Man_Error_2]
M1_poly = np.poly1d(np.polyfit(log_l, log_Man_1, 1))
M2_poly = np.poly1d(np.polyfit(log_l, log_Man_2, 1))
Omega_1 = M1_poly[0]
Omega_2 = M2_poly[0]
Error_Om = Omega_2*-1 - Omega_1*-1

print(f'Omega Final = {round(ppp_1[1], 3)} +- {round(Error_Om, 3)}')

"""CALCULATION OF a_1 WITH ERRORS"""
a_1 = np.exp(ppp_1[0])

#calculation of errors
a_1_upper = np.exp(ppp_1[0] + Error_Om)
a_1_lower = np.exp(ppp_1[0] - Error_Om)
Error_a_1 = a_1_upper - a_1_lower
print(f'a_1 Final = {round(a_1, 3)} +- {round(Error_a_1, 3)}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""QUESTION 3"""
"""SCATTER NORMALISED AVALANCHE PLOT"""
#collect required data
counts_list = np.loadtxt('avalanche_sizes_256.csv', delimiter=',').tolist()
av_128 = counts_list[2][1000000:] #select data for system size L=128
maxx = max(av_128)
av_norm = [i/maxx for i in av_128] #normalise values
plt.figure(90)
x = np.arange(1,4000001)
av_norm = np.array(av_norm)
av_norm = av_norm.T
av_norm = list(av_norm)
plt.scatter(x, av_norm, s=0.2, color='mediumblue')
plt.xlabel('t')
plt.ylabel(r's/$s_{max}$')
plt.title('Normalised Avalanche Size')
plt.show()

#%%
"""DATA COLLAPSE"""
#The program will produce binned avalanche-size probability densities, that is, 
#normalised avalanche-size probabilities.
sizes = [[4, 'b'], [8, 'm'], [16, 'c'], [32, 'k'], [64, 'g'], [128, 'y'], [256, 'r']]
counts_list = np.loadtxt('avalanche_sizes_256.csv', delimiter=',').tolist()
s_M =[] #largest avalanche observed for each L will be stored here

"""P(s;L) vs s"""
fig13 = plt.figure(13, figsize=(10,5))
ax13 = fig13.add_subplot(1,1,1)

"""DATA COLLAPSE on P(s;L) vs s"""
fig14 = plt.figure(14, figsize=(10,5))
ax14 = fig14.add_subplot(1,1,1)

"""DATA COLLAPSE on P(s;L)s vs s"""
fig15 = plt.figure(15, figsize=(10,5))
ax15 = fig15.add_subplot(1,1,1)

for i in range(len(p)-1, -1, -1):  #use if plotting with scale-1
#for i in range(len(p)):
    P_s_t = []
    s_D = []
    av_sizes = counts_list[i]
    logbin_data =  logbin.logbin(av_sizes, scale=1)
    ax13.plot(logbin_data[0], logbin_data[1], color=sizes[i][1], label= f'L={sizes[i][0]}')
    for j in range(len(logbin_data[0])):
        P_s_t.append(logbin_data[1][j] * logbin_data[0][j]**(1.53))
        s_D.append(logbin_data[0][j] / p[i]**(2.14))
    ax14.plot(s_D, P_s_t, color=sizes[i][1], label= f'L={sizes[i][0]}')
    s_M.append(max(logbin_data[0])) #collect the largest avalanche sizes
    ax15.plot(logbin_data[0], P_s_t, color=sizes[i][1], label= f'L={sizes[i][0]}')

ax13.set_xlabel('s')
ax13.set_ylabel('P(s; L)')
ax13.set_xscale('log')
ax13.set_yscale('log')
ax13.set_title('P(s;L) vs. s, scale=1')
ax13.legend()

ax14.set_xlabel(r's/$L^{D}$')
ax14.set_ylabel(r'$s^{\tau_{s}}$ P(s;L)')
ax14.set_xscale('log')
ax14.set_yscale('log')
ax14.set_title(r'$s^{\tau_{s}}$ P(s;L)  vs.  s/$L^{D}$')
ax14.legend()

ax15.set_xlabel('s')
ax15.set_ylabel(r'$s^{\tau_{s}} $P(s;L)')
ax15.set_xscale('log')
ax15.set_yscale('log')
ax15.set_title(r'$s^{\tau_{s}}$ P(s;L)  vs.  s')
ax15.legend()
#%%
"""GRADIENT OF LINEAR REGION OF FIG. 15, L=256"""
#want to only consider linear region, so where logbin_data[0] is below L^2
logbin_data =  logbin.logbin(counts_list[6], scale=1.25)
x = np.log(logbin_data[0])
y = np.log(logbin_data[1])
plt.plot(np.exp(x[x<10]), np.exp(y[x<10]), label='256')
plt.xscale('log')
plt.yscale('log')
Poly_Ps, cov = np.polyfit(x[x<10].tolist(), y[x<10].tolist(), 1, cov=True)
POLY_PS = np.poly1d(Poly_Ps)
plt.legend()

print(f'L=256: {POLY_PS[1]}k{POLY_PS[0]}')
print('Error on gradient (tau) (for L=256):', cov[0][0])

"""EEROR ON 256"""
#L^2 corresponds to x=11.1 on this scale, so find the difference between
#the gradients when the data x[x<11], and x[x<10] is used
err = np.poly1d(np.polyfit(x[x<11].tolist(), y[x<11].tolist(), 1))
Error = err[1] - POLY_PS[1]
print('Error', Error)

#%%
"""MAX AVALANCHE SIZE AGAINST L"""
plt.plot(p, s_M, 'x')
pyplot, cov = np.polyfit(np.log(p[4:]), np.log(s_M[4:]), 1, cov=True)
PYPLOT = np.poly1d(pyplot)
plt.plot(p, np.exp(PYPLOT(np.log(p))), '-', label=f'$s_M$ = L^${round(PYPLOT[1],3)}$')
print('Gradient', PYPLOT[1])
print('Error on Gradient', cov[0][0])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel(r'$s_M$')
plt.title(r'Maximum Avalanche Size $s_M$')
plt.legend()
#%%
"""MOMENT ANALYSIS"""
#specify colours for future plotting
colors = ['b','m','c', 'goldenrod', 'g','y','r','darkorchid', 'k']
grads = []
SSR = []
counts_list = np.loadtxt('avalanche_sizes_256.csv', delimiter=',').tolist()

plt.figure(16)
for k in np.linspace(1, 9, 9):
    mom = []
    for i in range(len(p)): #calc all moments^k for one L, before exploring next L
        #considering last 4 million pieces of data, i.e. only steady state
        av_sizes_to_k = counts_list[i][1000000:]**k
        mom.append(sum(av_sizes_to_k)/ 4000000)
    
    plt.plot(p, mom, 'x', label = f'k = {int(k)}', color= colors[int(k)-1])
    pplot_t = np.polyfit(np.log(p[4:7]), np.log(mom[4:7]), 1, full=True)
    Polyplot = np.poly1d(pplot_t[0])
    plt.plot(p, np.exp(Polyplot(np.log(p))), '-', color=colors[int(k)-1])
    
    grads.append(np.exp(Polyplot[1]))
    SSR.append(pplot_t[1][0])
    
plt.xlabel('L')
plt.ylabel(r'$\langle s^{k} \rangle$')
plt.title('Kth Moment')
plt.xscale('log')
plt.yscale('log')
plt.legend()
print('SSR', SSR)
#%%
"""FINAL GRAPH OF D(1-frac) AGAINST K"""
#consider k values of 1 to 9
k_list = np.linspace(1, 9, 9)
fig17 = plt.figure(17, figsize=(10,5))
ax17 = fig17.add_subplot(1,1,1)
ax17.plot(k_list, np.log(grads), 'x')

pplot_1, cov = np.polyfit(k_list, np.log(grads), 1, cov=True)
pplot_2 = np.polyfit(k_list[:4], np.log(grads)[:4], 1)
Polyplot_1 = np.poly1d(pplot_1)
Polyplot_2 = np.poly1d(pplot_2)
ax17.plot(k_list, Polyplot_1(k_list), '-', color='b', label=fr'$D(1+k-\tau)$ =2.220k -1.222') #{round(Polyplot_2[1],4)}k{round(Polyplot_2[0],4)}
ax17.legend()
ax17.set_xlabel('k')
ax17.set_ylabel(r'$D(1+k-\tau)$')
ax17.set_title(r'$D(1+k-\tau)$ vs. k')

error_1 = cov[0][0] #slope
print(error_1)
error_2 = cov[1][1] #intercept 
print(error_2)

print('D (first 5)', Polyplot_2[1])
print('intercept (first 5)', Polyplot_2[0])

"""D ERROR CALCULATION"""
#Find the gradient (D) when all k values are used. Take the difference with the
#true gradient and take as error.
err = np.poly1d(np.polyfit(k_list, np.log(grads), 1))
Error_intercept = Polyplot_2[0] - err[0]
Error_D = Polyplot_2[1] - err[1]
Error_tau = np.sqrt(Error_D**2 + Error_intercept**2)
print('Error on D:', Error_D)
print('Error on tau', Error_tau)

