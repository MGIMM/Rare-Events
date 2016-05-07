from pyrare import *
import numpy as np
from scipy.stats import norm
from time import time

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set_style("whitegrid")



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
    num_lev = len(L_ideal)

    if status_tracking == True:
        print ("p_0 = " + str(p_0) + '\t n_0 =' + str(n_0) + "\t r = " + str(r))
        #print ("sequence of levels: "+ str(L_ideal))
        print ("num_lev: "+ str(num_lev))
        #print ("level interested, L = "+ str(q_test))
        # real value of p
        print ("real value of p:" + str(real_p))
        print ("theoretical relative deviation: " + str(sigma_theoretical))
    return real_p,  L_ideal, sigma_theoretical

def mu_0_test(N):
    '''
    param N: the size of particles
    '''
    return np.random.normal(0,1,N)



#tuning parameter for shaker
####Attention !!!!!!!!   we could not choose a very large sigma_1 !!!

def shaker_test(X,sigma_1):
    c = np.sqrt(1+sigma_1**2)
    return np.random.normal(X/c,sigma_1/c,1)

def shaker_metropolis(X,sigma_1):
    iter = np.random.uniform(X-sigma_1,X+sigma_1)
    if np.exp(-0.5*(iter**2-X**2))>np.random.rand(1):
    	return iter
    else:
    	return X

p_0_test = 0.2

get_params_test(q_test = 8, p_0 = p_0_test,status_tracking = True)

#print ("\n_________________________\n")

rare_test = RareEvents(mu_0 = mu_0_test, score_function = S_test, level = 8,shaker = shaker_metropolis, p_0 = p_0_test)
rare_test.adaptive_levels(N = 100, shake_times = 5, reject_rate = 0.5, sigma_default = 0.5, 
											descent_step = 0.2,status_tracking=True)

