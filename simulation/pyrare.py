import numpy as np
from time import time


class RareEvents:
    def __init__(self, mu_0, score_function,
                 shaker,level, p_0 = 0.75):
        """
        :param level: level to estimate
        :param p_0: successful rate
        :param mu_0: distribution of X_0
        :param score_function: black box score_function
        :param shaker: metro-polis/Gibbs kernel
        """
        self.mu_0 = mu_0
        self.score_function = score_function
        self.shaker = shaker
        self.level = level
        self.p_0 = p_0

    def adaptive_levels(self,N, shake_times = 10,\
                                  reject_rate = 0.5, sigma_default = 0.5,\
                                descent_step = 0.02, status_tracking = False):
        ###### Initiation
        t_0 = time()
        X = self.mu_0(N)
        
        p_0 = self.p_0
        L = np.array([-np.Inf,np.sort(self.score_function(X))[np.int((1-p_0)*N)]])
        k = 1
        #S_called_times = N
        E = []
        E += [range(N)]
        ######

        while(L[k]<self.level):
            I = []
            survive_index = []
            for i in range(N):
                if self.score_function(X[i])>L[k]:
                    I += [X[i]]
                    #S_called_times += 1
                    survive_index += [i]
                    

            ell = len(I) 
       ####### to ensure that I_k would not be empty
            if ell == 0:        
                break

       ###### permutation trick to replace multinominal distribution
       # we remark that the permutation doesn't work for the estimator of var

            #I = np.random.permutation(I)
            #q = 0
            #for i in clone_index:
            #    X[i] = I[q%ell]
            #    q += 1
            A = [] 
            X_cloned = np.zeros(N)
            for i in range(N):
                A += [int(np.random.choice(survive_index))]
                X_cloned[i] = X[A[i]]
            ## permutation trick ##
            #A_permutation = np.random.permutation(survive_index)
            #for i in range(N):
            #    A += [A_permutation[i%ell]]
            #    X_cloned[i] = X[A[i]]
                
            X = X_cloned
            E_iter = []
            for i in range(N):
                E_iter += [E[k-1][A[i]]]

            E += [E_iter]
        ###### shaker
        ###### we will use an adaptive method to set the parameter of the shaker by
        ###### controlling the rejection rate.

            
            for index_shaker in range(shake_times):
                rate = 1.
                sigma_1 = sigma_default

                while(rate>reject_rate):
                    reject = N  
                    for j in range(N):
                        X_iter = self.shaker(X[j],sigma_1 = sigma_1)
                        if self.score_function(X_iter) > L[k]:
                            #S_called_times += 1
                            X[j] = X_iter
                            reject -= 1.
                    rate = reject/np.float(N)
                    sigma_1 -= descent_step
                    if sigma_1 <= 0:
                        break
        ######
            L = np.append(L, np.sort(self.score_function(X))[np.int((1-p_0)*N)])
            k += 1

        N_L = np.sum((self.score_function(X)>self.level))
        r_hat = N_L/float(N)
        p_hat = p_0**(k-1)*r_hat 
        n = k - 1
        phi = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if E[n][j] == i:
                    phi[i] += np.float(self.score_function(X[j])>self.level)
            phi[i] = phi[i]/float(N)

        phi = np.array(phi)
        m = (N/float(N-1))**k*(r_hat**2 - np.sum(phi**2))
        #m = (r_hat**2 - np.sum(phi**2))
        V = r_hat**2 - m
        V = V/r_hat**2
        V *= N

        if status_tracking ==True:
            print ("estimation of p: " + str(p_hat))
            print ('____________________________________________________________\n')
            print ("Time spent: %s s" %(time() - t_0) )
            print ("Estimation of variance: " + str(V))
            #print ("score_function called: %s times" % S_called_times)
        output = {'p_hat':p_hat,  \
                 
                  'V': V
                 }    
        return output 
########################################################################
########################################################################
########################################################################
##############################fixed levels##############################                
########################################################################
########################################################################
########################################################################
    def fixed_levels(self,N,L, shake_times = 10,\
                                  reject_rate = 0.5, sigma_default = 0.5,\
                                descent_step = 0.02, status_tracking = False):
        ###### Initiation
        t_0 = time()
        X = self.mu_0(N)
        
        p_0 = self.p_0
        k = 1
        #S_called_times = N
        list_hat_p =  []
        E = []
        E += [range(N)]
        ######

        while(k <= len(L) -1 ):

            I = []
            survive_index = []
            for i in range(N):
                if self.score_function(X[i])>L[k]:
                    I += [X[i]]
                    #S_called_times += 1
                    survive_index += [i]
                    

            ell = len(I) 

            if ell == 0:        ###### to ensure that I_k would not be empty
                break
            list_hat_p = np.append(list_hat_p, ell/float(N))

       ###### permutation trick to replace multinominal distribution
            #I = np.random.permutation(I)
            #q = 0
            #for i in clone_index:
            #    X[i] = I[q%ell]
            #    q += 1
            ### multinominal cloning ###
            A = [] 
            X_cloned = np.zeros(N)
            for i in range(N):
                A += [int(np.random.choice(survive_index))]
                X_cloned[i] = X[A[i]]
                
            ## permutation trick ##
            #A_permutation = np.random.permutation(survive_index)
            #for i in range(N):
            #    A += [A_permutation[i%ell]]
            #    X_cloned[i] = X[A[i]]
                
            X = X_cloned
            E_iter = []
            for i in range(N):
                E_iter += [E[k-1][A[i]]]

            E += [E_iter]
        ###### shaker
        ###### we will use an adaptive method to set the parameter of the shaker by
        ###### controlling the rejection rate.

            
            for index_shaker in range(shake_times):
                rate = 1.
                sigma_1 = sigma_default

                while(rate>reject_rate):
                    reject = N  
                    for j in range(N):
                        X_iter = self.shaker(X[j],sigma_1 = sigma_1)
                        if self.score_function(X_iter) > L[k]:
                            #S_called_times += 1
                            X[j] = X_iter
                            reject -= 1.
                    rate = reject/np.float(N)
                    sigma_1 -= descent_step
                    if sigma_1 <= 0:
                        break
        ######


            k += 1
            #X_trace += [X]

        p_hat = np.prod(list_hat_p) 
        N_L = np.sum((self.score_function(X)>L[-1]))
        r_hat = N_L/float(N)
        #p_hat = r_hat*p_hat
        n = k - 1
        phi = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if E[n][j] == i:
                    #phi[i] += np.float(self.score_function(X[j])>L[-1])
                    phi[i] += 1
            phi[i] = phi[i]/float(N)

        phi = np.array(phi)
        print phi
        #m = (N/float(N-1))**(n+1)*(r_hat**2 - np.sum(phi**2))
        m = (N/float(N-1))**(n+1)*(1 - np.sum(phi**2))
        V = 1 - m
        #V = r_hat**2 - m
        #V = V/r_hat**2
        V *= N

        if status_tracking ==True:
            print ("estimation of p: " + str(p_hat))
            print ('____________________________________________________________\n')
            print ("Time spent: %s s" %(time() - t_0) )
            #print ("score_function called: %s times" % S_called_times)
        output = {'p_hat':p_hat,  \
                 
                  'V': V
                 }    
        return output 
