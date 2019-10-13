#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('case3.txt', delimiter=',', unpack=True)
time_data, pitch_angle = data
# STEP 1
# let's plot scatter for two variables from our dataset
plt.scatter(time_data, pitch_angle,s = 1, label = 'Real function')
plt.xlabel('time')
plt.ylabel('pitch angle')
#plt.legend()
#plt.show()


#STEP2: Preparations
var_pitch = np.var(pitch_angle) #variance of pitch angle
var_model = 50 #variance of model

N = np.shape(pitch_angle)[0]
print('Number of pitch observations=',N)
#STEP3
#Kalman Filter
pitch_opt = np.zeros(N) #creating a matrix with N elements
e_opt = np.zeros(N)
K = np.zeros(N)
pitch_opt_s = np.zeros(N)

pitch_opt[0] = pitch_angle[0] #initial conditions
pitch_opt_s[0] = pitch_angle[0]
e_opt[0] = var_pitch**(1/2) #base of the iteration (mean of the error's square)

for i in range(1, N):
    e_opt[i] = np.sqrt(var_pitch*(e_opt[i-1]**2+var_model)
                       /(var_model+e_opt[i-1]**2+var_pitch))
    K[i] = (e_opt[i]**2/var_pitch)
    pitch_opt[i] = (pitch_opt[i-1])*(1-K[i])+K[i]*pitch_angle[i]
print('Optimal filtered pitch angles=',pitch_opt)
plt.scatter(time_data,pitch_opt,s = 20,label = 'Kalman filtering',c ='black')

#Step4 Simple Kalman Filter
#plt.scatter(time_data,K,s = 50) #plot scatter for estimating Kstab
#From plot We can see that optimal value is 0.06568166
print('Kstab=',K)
K_opt = 0.065
for i in range(1, N):
     pitch_opt_s[i] = (pitch_opt_s[i-1])*(1-K_opt)+K_opt*pitch_angle[i]
plt.scatter(time_data,pitch_opt,s = 5,label = 'Simple Kalman filtering')
plt.legend()
plt.title('Kalman Filter')
plt.show()


# In[ ]:




