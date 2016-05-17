from pyrare import *
import sys
import numpy as np
from scipy.stats import norm
from time import time
from ProgressBar import *
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

def S_test(X):
    #score function which is a black box
    return np.abs(X)

def get_paramS_test(q_test = 8, p_0 = 0.75,status_tracking = True):

    real_p = (1-norm.cdf(q_test))*2
    n_0 = int(np.floor(np.log(real_p)/np.log(p_0)))
    r = real_p/(p_0**n_0)
    sigma_theoretical = np.sqrt(n_0*(1-p_0)/p_0 + (1-r)/r)

    if status_tracking == True:
        print ("p_0 = " + str(p_0) + '\t n_0 =' + str(n_0) + "\t r = " + str(r))
        print ("real value of p: " + str(real_p))
    return real_p, sigma_theoretical, n_0

def mu_0_test(n):
    '''
    param n: the size of particles
    '''
    return np.random.normal(0,1,n)



#tuning parameter sigma_1 for shaker


def shaker_gaussian(x,sigma_1):
    c = np.sqrt(1+sigma_1**2)
    return np.random.normal(x/c,sigma_1/c,1)

def shaker_metropolis(x,sigma_1):
    iter = np.random.uniform(x-sigma_1,x+sigma_1)
    if np.exp(-0.5*(iter**2-x**2))>np.random.rand(1):
    	return iter
    else:
    	return x


print ('\n============================================================')
####### parameters ######
p_0_test = 0.1 
N_test = 1000
#shaker = shaker_metropolis
shaker = shaker_gaussian
shake_times = 3 
num_simulation = 100
#def input_parameters():
#    print ('please input the parameters:\n')
#    p_0_test = input('p_0_test: ')
#    N_test = input('number of particles: ')
#    shake_times = input('shake_times: ')
#    shaker_choice = 'empty'
#    while shaker_choice not in ['metropolis', 'gaussian']:
#        shaker_choice = raw_input("shaker('metropolis' or 'gaussian'): ")
#    
#    if shaker_choice == 'metropolis':
#        shaker = shaker_metropolis
#    elif shaker_choice == 'gaussian':
#        shaker = shaker_gaussian
#    num_simulation = 100
#    return p_0_test, N_test, shaker, shake_times, num_simulation
#
#p_0_test, N_test, shaker, shake_times, num_simulation = input_parameters()
#########################

test_info = '|num_particles_' + str(N_test) + '|' + \
        str(shaker).split(' ')[1] + '|shake_times_' + str(shake_times) 
        

print ('Info: ' + test_info)
params = get_paramS_test(q_test = 8, p_0 = p_0_test,status_tracking = True)

rare_test = RareEvents(mu_0 = mu_0_test, score_function = S_test, 
	level = 8,shaker = shaker, p_0 = p_0_test)
#rare_test.adaptive_levels(n = 1000, shake_times = 10, 
#	reject_rate = 0.5, sigma_default = 0.5, descent_step = 0.2,status_tracking=True)
#
##############################
### simualtion & plotting ####
##############################
list_p = []
list_n_0 = []
list_r = []
list_s_called_times = []
bar = ProgressBar(total = num_simulation)
for i in range(num_simulation):
###ProgressBar###
    bar.move()
    bar.log()
#################
    iter_output = rare_test.adaptive_levels\
            (N = N_test,  shake_times = shake_times,reject_rate = 0.5, \
            sigma_default = 0.5, descent_step = 0.2,status_tracking=False)
    list_p = np.append(list_p, iter_output['p_hat'])           
    list_n_0 += [iter_output['n_0_hat']]
    list_r = np.append(list_r,iter_output['r_hat'])

r_hat = np.mean(list_r)
n_0_real = params[2]
p_real = params[0]
n_0_hat = list_n_0[np.argmax(np.bincount(list_n_0))] 
sigma_idealized = np.sqrt(n_0_hat*((1-p_0_test)/p_0_test) + (1-r_hat)/r_hat) 
p_hat = np.mean(list_p)
relative_var = (list_p/p_real - 1) * np.sqrt(N_test)
sigma_empirical = np.std(relative_var)
print ('estimation of p: ' + str(np.mean(list_p)))
print ('sigma (theoretical): ' + str(params[1]))
print ('sigma (idealized): ' + str(sigma_idealized))
print ('sigma_tilde (empirical): ' + str(sigma_empirical))
print ('============================================================\n')
