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

    ###idealized situation

    n_0 = int(np.floor(np.log(real_p)/np.log(p_0)))
    r = real_p/(p_0**n_0)
    sigma_theoretical = np.sqrt(n_0*(1-p_0)/p_0 + (1-r)/r)
    l = [-np.inf]
    for k in range(1,n_0+1,1):
        l = np.append(l, norm.ppf(1 - p_0**k/2))
    l_ideal = np.append(l, q_test)

    if status_tracking == True:
        print ("p_0 = " + str(p_0) + '\t n_0 =' + str(n_0) + "\t r = " + str(r))
        #print ("sequence of levels: "+ str(l_ideal))
        print ("real value of p: " + str(real_p))
        print ("theoretical relative deviation: " + str(sigma_theoretical))
    return real_p,  l_ideal, sigma_theoretical, n_0

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
#p_0_test = 0.1 
#N_test = 500
#shaker = shaker_metropolis
##shaker = shaker_gaussian
#shake_times = 5 
def input_parameters():
    print ('please input the parameters:\n')
    p_0_test = input('p_0_test: ')
    N_test = input('number of particles: ')
    shake_times = input('shake_times: ')
    shaker_choice = 'empty'
    while shaker_choice not in ['metropolis', 'gaussian']:
        shaker_choice = raw_input("shaker('metropolis' or 'gaussian'): ")
    
    if shaker_choice == 'metropolis':
        shaker = shaker_metropolis
    elif shaker_choice == 'gaussian':
        shaker = shaker_gaussian
    num_simulation = 100
    return p_0_test, N_test, shaker, shake_times, num_simulation

p_0_test, N_test, shaker, shake_times, num_simulation = input_parameters()
#########################

test_info = '|num_particles_' + str(N_test) + '|' + \
        str(shaker).split(' ')[1] + '|shake_times_' + str(shake_times) 
        

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
    list_s_called_times = np.append(list_s_called_times, iter_output['S_called_times']) 

r_hat = np.mean(list_r)

n_0_real = params[3]
p_real = params[0]
n_0_hat = list_n_0[np.argmax(np.bincount(list_n_0))] 
#n_0_hat = np.sort(list_n_0)[len(list_n_0)/2] 
sigma_hat = np.sqrt(n_0_hat*((1-p_0_test)/p_0_test) + (1-r_hat)/r_hat) 
alpha = norm.ppf(0.975)
p_hat = np.mean(list_p)
ic = [p_hat/(1+alpha*sigma_hat/np.sqrt(N_test)), p_hat/(1-alpha*sigma_hat/np.sqrt(N_test))]

### save the records ###
test_tracker = open('logs/INFO/INFO' + test_info + '.txt', 'w')
sys.stdout = test_tracker
print ('============================================================')
print ('status:\n')
get_paramS_test(q_test = 8, p_0 = p_0_test,status_tracking = True)
print ('shaker: ' + str(shaker))
print ('shake_times: ' + str(shake_times))
print ('number of particles: '+ str(N_test))
print ('____________________________________________________________')
print ('results:\n') 
print ('number of simulation: %s' % num_simulation)
print ('estimation of p: %s' % p_hat)
print ('sigma_hat: %s' % sigma_hat)
print ('n_0_hat: %s' % n_0_hat)
print ('95% confidence interval(idealized): \n' + str(ic))
relative_var = np.std((list_p/p_real - 1) * np.sqrt(N_test))
print ('empirical relative variance: '+ str(relative_var))
print ('score function called (average) %s times' % np.mean(list_s_called_times))
print ('============================================================\n')
test_tracker.close()
############### plot ###############    
plt.figure(figsize = [15,10], facecolor = 'white')
plt.subplot(211)
plt.hist((list_p/p_real - 1) * np.sqrt(N_test),color = 'grey', bins = num_simulation/10, label = 'estimation', normed = 1.)
plt.plot(np.arange(-100,100,0.1), norm.pdf(np.arange(-100,100,0.1), 0, params[2]),color = 'darkred',label = 'theoretical')
plt.xlim([-100,100])
plt.title('Relative Variance $\sqrt{N} (\hat p - p)/p$')
plt.grid(b=True, which='both', color='grey',linestyle='-')
plt.legend()

plt.subplot(212)
plt.hist(list_n_0,bins = len(list_n_0), color = 'grey', label = 'estimation' )
real_n_0, = plt.plot(n_0_real,0,color = 'darkred',marker = 'o',label = 'real $n_0$')
plt.title('Histogram of $\hat n_0$')
plt.legend(handler_map={ real_n_0: HandlerLine2D(numpoints=1)})
plt.grid(b=True, which='both', color='grey',linestyle='-')
plt.savefig('logs/FIGS/FIG' + test_info + '.png')
plt.show()
plt.close()
