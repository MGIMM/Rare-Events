import numpy as np
from time import time

class RareEvents: def __init__(self, mu_0, score_function, shaker,level, p_0 = 0.75):
        """
        :param level: level to estimate
        :param p_0: successful rate, the classical way is to choose p_0 large,
                    but for AMS, in order to fix the correct levels,
                    we would like to choose
                    p_0 relatively small. This will also improve the quality of the variance 
                    estimator.
        :param mu_0: distribution of X_0
        :param score_function: black box score_function
        :param shaker: metro-polis/Gibbs/Gaussian(for the toy example) kernel
        
        Remark:
        
        1. For the estimation of variance, we cannot change the times of
        shaking by the empirical reject_rate. The shaking procedure won't
        change the unbiasedness of the approximation of terminal path
        measures \gamma_n^N and \eta_n^N, but it will change the intergral
        operator \phi_p and the variance itself, and that's why the more we
        shake, the more close we get to the theoretical lower bound of the
        asymptotic variance. But, as we did before, if we really want a
        little variance which is close to the theoretical lower bound, this
        trick is still useful.

        2. We firstly provide a strongly consistent estimator for non
        asymptotic variance,
        whose complexity is O(N^2). We remark that it is also an weakly
        consistent estimator for the asymptotic variance. The stronly
        consistent estimator of asymptotic is O(N^3). Which is not that
        interest in practice due to the heavy calculation.
        
        3. We cannot use permutation to replace the multinomial procedure
        anymore. This trick will not effect the estimation of the
        \gamma_n and the crude estimator of the variance, but our
        estimator is not compatible with this trick. Further study will be
        done in a form of reconstruction of genealogical structure of the
        complete ancestral trees.  

        4. In practice, the difficulties of choosing p_0 close to 1 is that
        we need to shake many times to deduce de dependance of the parents
        and the children, while as we cannot use the accept-reject ratio to
        control this procedure anymore, it is difficult to find a reference
        to know if the particle system is well shaked.

        5. The function we defined will return the complete ancestral tree
        (xi, A) of the particle system. 

        """
        self.mu_0 = mu_0
        self.score_function = score_function
        self.shaker = shaker
        self.level = level
        self.p_0 = p_0

    def adaptive_levels(self,N, shake_times = 1, status_tracking = False):
        ###### Initiation
        t_0 = time()
        xi = []
        X = self.mu_0(N)
        xi += [X]
        A = [] 
        
        p_0 = self.p_0
        L = np.array([-np.Inf,np.sort(self.score_function(X))[np.int((1-p_0)*N)]])
        k = 1
        #E = []
        #E += [range(N)]

        while(L[k]<self.level):
            I = []
            survive_index = []
            for i in range(N):
                if self.score_function(X[i])>L[k]:
                    #I += [X[i]]
                    #S_called_times += 1
                    survive_index += [i]
            ell = len(survive_index) 
           # to ensure that I_k would not be empty
            if ell == 0:        
                break

           # we remark that the permutation trick below doesn't work for the estimator of var

            # I = np.random.permutation(I)
            # q = 0
            # for i in clone_index:
            #     X[i] = I[q%ell]
            #     q += 1
            A_k = np.zeros(N, dtype = np.int) 
            X_cloned = np.zeros(N)
            for i in range(N):
                A_k[i] = np.random.choice(survive_index)
                X_cloned[i] = X[A_k[i]]
            
            A += [A_k]   
            X = X_cloned

            
            for sigma_range in np.arange(0.35,0.05,-0.05):
                for j in range(N):
                    for index_shaker in range(shake_times):
                        X_iter = self.shaker(X[j],sigma_1 = sigma_range)
                    if self.score_function(X_iter)>L[k]:
                        X[j] = X_iter
            L = np.append(L, np.sort(self.score_function(X))[np.int((1-p_0)*N)])
            xi += [X]
            k += 1
        n = k
        N_L = np.sum((self.score_function(X)>self.level))
        r_hat = N_L/float(N)
        p_hat = p_0**(n-1)*r_hat 

        if status_tracking ==True:
            print ("estimation of p: " + str(p_hat))
            print ('____________________________________________________________\n')
            print ("Time spent: %s s" %(time() - t_0) )

        output = {'p_hat':p_hat,  \
                  'A':A,\
                  'xi':xi,\
                 }    
        return output 

##########################################################################################

from scipy.stats import norm
import sys

# some functions for testing the toy example

def S_test(X):
    '''
    score function which is a black box
    '''
    return np.abs(X)

def get_paramS_test(level_test = 8, p_0 = 0.75,status_tracking = True):
    '''
    This function returns the real values for the toy example 
    '''
    real_p = (1-norm.cdf(level_test))*2
    n_0 = int(np.floor(np.log(real_p)/np.log(p_0)))
    r = real_p/(p_0**n_0)
    sigma2_ideal = n_0*(1-p_0)/p_0 + (1-r)/r
    sigma2_ideal *= r**2

    # l = [-np.inf]
    # for k in range(1,n_0+1,1):
    #     l = np.append(l, norm.ppf(1 - p_0**k/2))
    # l_ideal = np.append(l, level_test)

    if status_tracking == True:
        print ("p_0 = " + str(p_0) + '\t n_0 =' + str(n_0) + "\t r = " + str(r))
        # print ("sequence of levels: "+ str(l_ideal))
        print ("real value of p: " + str(real_p))
        print ("relative variance(ideal): " + str(sigma2_ideal))

    output = {	'p':real_p, \
		'sigma2_ideal':sigma2_ideal, \
		'n_0':n_0, \
		'r':r}
    return output 

def mu_0_test(N):
    '''
    param n: the size of particles
    '''
    return np.random.normal(0,1,N)

def shaker_gaussian(x,sigma_1=0.2):
    '''
    a reversible transition kernel for mu_0_test
    '''
    c = np.sqrt(1+sigma_1**2)
    return np.random.normal(x/c,sigma_1/c,1)

##########################################################################################
# test by a toy example

if __name__ == '__main__':

    from ProgressBar import *
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')

    print ('\n============================================================')
    # parameters
    N_test = 1000 
    p_0_test = 0.5 
    shaker = shaker_gaussian
    shake_times = 1 
    num_simulation = 200
    level_test = 3 
    test_info = '|num_particles_' + str(N_test) + '|' + \
            str(shaker).split(' ')[1] + '|shake_times_' + str(shake_times) 
            
    ################################################################################
    
    print ('Info: ' + test_info)
    params = get_paramS_test(level_test = level_test, p_0 = p_0_test,status_tracking = True)
    
    # definition of the RareEvents class
    rare_test = RareEvents(mu_0 = mu_0_test, score_function = S_test,\
    	level = level_test,shaker = shaker, p_0 = p_0_test)
    
    # test
    test_result = rare_test.adaptive_levels(\
            N = N_test, shake_times = shake_times, status_tracking=True)
    # tracing the genealogical information
    A = test_result['A']
    xi = test_result['xi']
    print ('============================================================')

    def var_estimator_non_asym(xi,A,N = N_test):
        '''
        this function is simplified version for estimating the asymptotic variance for 
        f = \mathds{1}_{\{S(x)>level_test\}}. 
        '''
        
        n = np.shape(xi)[0]-1 
        O = np.zeros((n+1,n+1,N))
        for p in range(n+1):
            for q in np.arange(p,n+1):
                for i in range(N):
                    k = q
                    anc = i
                    while k>p:
                        anc = A[k-1][anc]
                        k -= 1
                    O[p][q][i] = anc 
            
        
        I_n = []
        for i in range(N):
            if xi[n][i]> level_test:
                I_n += [i]
                
        set_0 = np.zeros(N) 
        for ind_anc in range(N):
            for i in I_n:
                if O[0][n][i] == ind_anc:
                    set_0[ind_anc] += 1
                    
        V = np.sum(set_0)**2 - np.sum(set_0**2)
        V *= N**(n-1)*1.0/(N-1)**(n+1) 
        V = (float(len(I_n))/N)**2 -V
        
        V *= N
        return V
    
print ('Non asymptotic variance estimator: '+ str(var_estimator_non_asym(xi,A)))
