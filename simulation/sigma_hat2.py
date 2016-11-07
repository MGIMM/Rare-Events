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

def get_paramS_test(level_test = 8, p_0 = 0.75,status_tracking = True):

    real_p = (1-norm.cdf(level_test))*2

    ###idealized situation

    n_0 = int(np.floor(np.log(real_p)/np.log(p_0)))
    r = real_p/(p_0**n_0)
    sigma_theoretical = np.sqrt(n_0*(1-p_0)/p_0 + (1-r)/r)
    l = [-np.inf]
    for k in range(1,n_0+1,1):
        l = np.append(l, norm.ppf(1 - p_0**k/2))
    l_ideal = np.append(l, level_test)

    if status_tracking == True:
        print ("p_0 = " + str(p_0) + '\t n_0 =' + str(n_0) + "\t r = " + str(r))
        #print ("sequence of levels: "+ str(l_ideal))
        print ("real value of p: " + str(real_p))
        print ("relative deviation(ideal): " + str(sigma_theoretical))
    return real_p, sigma_theoretical, n_0, l_ideal

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
p_0_test = 0.8 
N_test = 2000
#shaker = shaker_metropolis
shaker = shaker_gaussian
shake_times = 2 
reject_rate = 0.3
num_simulation = 200
level_test = 6 
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
params = get_paramS_test(level_test = level_test, p_0 = p_0_test,status_tracking = True)
L_ideal = params[3]

rare_test = RareEvents(mu_0 = mu_0_test, score_function = S_test,\
	level = level_test,shaker = shaker, p_0 = p_0_test)

#test
rare_test.adaptive_levels(N = N_test, shake_times = 1, reject_rate = 0.5,
        sigma_default = 0.5, descent_step = 0.05,status_tracking=True)


##############################
### simualtion & plotting ####
##############################
list_p = []
list_n_0 = []
list_r = []
list_V = []
list_s_called_times = []
bar = ProgressBar(total = num_simulation)
for i in range(num_simulation):
###ProgressBar###
    bar.move()
    bar.log()
#################
##fixed_levels
#    iter_output = rare_test.fixed_levels\
#            (N = N_test, L = L_ideal,  shake_times = shake_times,reject_rate = reject_rate, \
#            sigma_default = 0.5, descent_step = 0.1,status_tracking=False)


#adaptive_levels
    iter_output = rare_test.adaptive_levels\
            (N = N_test, shake_times = shake_times,reject_rate = reject_rate, \
            sigma_default = 0.5, descent_step = 0.05,status_tracking=False)
    list_p = np.append(list_p, iter_output['p_hat'])           
    list_V = np.append(list_V, iter_output['V'])
#r_hat = np.mean(list_r)
n_0_real = params[2]
p_real = params[0]
#n_0_hat = list_n_0[np.argmax(np.bincount(list_n_0))] 
#sigma_idealized = np.sqrt(n_0_hat*((1-p_0_test)/p_0_test) + (1-r_hat)/r_hat) 
p_hat = np.mean(list_p)
relative_var = (list_p/p_real - 1) * np.sqrt(N_test)
sigma_empirical = np.var(relative_var)
#sigma_empirical = np.sum(relative_var**2)/float(num_simulation)
print ('estimation of p: ' + str(np.mean(list_p)))
#print ('sigma (theoretical): ' + str(params[1]))
#print ('sigma (idealized): ' + str(sigma_idealized))
print ('sigma_tilde2 (empirical): ' + str(sigma_empirical))
#print ('reject rate: ' + str(reject_rate))
print ('============================================================\n')
#print 'relative variation:'
#print relative_var
#print 'variance corrected:'
#print np.var((np.abs(relative_var)<12)*relative_var)
print ('list_prediction:')
print (list_V)
print ('mean of prediction:')
print (np.mean(list_V))
plt.hist(list_V, normed = 1, color = 'grey', alpha = 0.5)
plt.title("Histogram of Prediction")
plt.show()
