from pyrare import *
import numpy as np
from scipy.stats import norm
from time import time
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def S_test(X):
    #score function which is a black box
    return np.abs(X)

def get_params_test(q_test = 8, p_0 = 0.75,status_tracking = True):

    real_p = (1-norm.cdf(q_test))*2

    ###idealized situation

    n_0 = int(np.floor(np.log(real_p)/np.log(p_0)))
    r = real_p/(p_0**n_0)
    sigma_theoretical = np.sqrt(n_0*(1-p_0)/p_0 + (1-r)/r)
    L = [-np.Inf]
    for k in range(1,n_0+1,1):
        L = np.append(L, norm.ppf(1 - p_0**k/2))
    L_ideal = np.append(L, q_test)

    if status_tracking == True:
        print ("p_0 = " + str(p_0) + '\t n_0 =' + str(n_0) + "\t r = " + str(r))
        #print ("sequence of levels: "+ str(L_ideal))
        print ("real value of p: " + str(real_p))
        print ("theoretical relative deviation: " + str(sigma_theoretical))
    return real_p,  L_ideal, sigma_theoretical, n_0

def mu_0_test(N):
    '''
    param N: the size of particles
    '''
    return np.random.normal(0,1,N)



#tuning parameter sigma_1 for shaker


def shaker_gaussian(X,sigma_1):
    c = np.sqrt(1+sigma_1**2)
    return np.random.normal(X/c,sigma_1/c,1)

def shaker_metropolis(X,sigma_1):
    iter = np.random.uniform(X-sigma_1,X+sigma_1)
    if np.exp(-0.5*(iter**2-X**2))>np.random.rand(1):
    	return iter
    else:
    	return X


print ('\n============================================================')
p_0_test = 0.1 
N_test = 500
#shaker = shaker_metropolis
shaker = shaker_gaussian
shake_times = 10

print ('shaker: ' + str(shaker))
print ('shake_times: ' + str(shake_times))
print ('number of particles: '+ str(N_test))
params = get_params_test(q_test = 8, p_0 = p_0_test,status_tracking = True)

rare_test = RareEvents(mu_0 = mu_0_test, score_function = S_test, 
	level = 8,shaker = shaker, p_0 = p_0_test)
#rare_test.adaptive_levels(N = 1000, shake_times = 10, 
#	reject_rate = 0.5, sigma_default = 0.5, descent_step = 0.2,status_tracking=True)
#
##############################
# Simualtion & Plotting #
##############################
print ("Calculating...")
list_p = []
list_n_0 = []
list_r = []
for i in range(100):
    iter_simualtion = rare_test.adaptive_levels\
            (N = N_test,  shake_times = shake_times,reject_rate = 0.5, \
            sigma_default = 0.5, descent_step = 0.2,status_tracking=False)
    list_p = np.append(list_p,iter_simualtion[0])           
    list_n_0 += [iter_simualtion[1]]
    list_r = np.append(list_r,iter_simualtion[2])

r_hat = np.mean(list_r)
n_0_real = params[3]
p_real = params[0]
n_0_hat = list_n_0[np.argmax(np.bincount(list_n_0))] 
#n_0_hat = np.sort(list_n_0)[len(list_n_0)/2] 
sigma_hat = np.sqrt(n_0_hat*((1-p_0_test)/p_0_test) + (1-r_hat)/r_hat) 
alpha = norm.ppf(0.975)
p_hat = np.mean(list_p)
IC = [p_hat/(1+alpha*sigma_hat/np.sqrt(N_test)), p_hat/(1-alpha*sigma_hat/np.sqrt(N_test))]
print ('____________________________________________________________\n')
print ('estimation of p: %s' % p_hat)
print ('sigma_hat: %s' % sigma_hat)
print ('n_0_hat: %s' % n_0_hat)
print ('95% confidence interval(idealized): \n' + str(IC))
relative_var = np.std((list_p/p_real - 1) * np.sqrt(N_test))
print ('empirical relative variance: '+ str(relative_var))
print ('============================================================\n')
############### PLOT ###############    
plt.figure(figsize = [15,10])
plt.subplot(211)
sns.distplot(list_p,color = 'grey', label = 'estimation')
plt.plot(p_real,0,color = 'darkred',marker = 'o',label = 'real_p')
plt.xlim(p_real-5*p_real,p_real+5*p_real)
plt.legend()

plt.subplot(212)
sns.distplot(list_n_0,color = 'grey', label = 'estimation')
plt.plot(n_0_real,0,color = 'darkblue',marker = 'o',label = 'real_k')
plt.legend()
plt.show()
