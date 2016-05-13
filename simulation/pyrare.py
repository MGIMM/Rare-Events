import numpy as np
from time import time


class RareEvents:
    def __init__(self, mu_0, score_function,
                 shaker,level, p_0 = 0.75):
        """
        :param level: the level to estimate
        :param p_0: the successful rate
        :param mu_0: the distribution of X_0
        :param score_function: the black box score_function
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
        S_called_times = 0



        ######

        while(L[k]<self.level):
    #         if status_track == True:
    #             print ("\t")
    #             print ("k = " + str(k))
    #             print ('current level: ' + str(L[k]))
            I = []
            for i in range(N):
                if self.score_function(X[i])>L[k]:
                    S_called_times += 1
                    I = np.append(I, X[i])
            ell = len(I)
            if ell == 0:        ###### to ensure that I_k would not be empty
                break
            X[0:ell] = I


        ###### permutation trick to replace multinominal distribution
            I = np.random.permutation(I)
            for i in range(ell,N,1):
                X[i] = I[i%ell]

        ###### shaker
        ###### we will use an adaptive method to set the parameter of the shaker by
        ###### controlling the rejection rate.

            for index_shaker in range(shake_times):
                rate = 1.
                sigma_1 = sigma_default
                while(rate>reject_rate):

                    reject = N - ell          #if we don't change the elements of I_k
                    for j in range(ell,N,1):
                    #reject = N               #if we change the elements of I_k to have less dependence
                    #for j in range(N):

                        X_iter = self.shaker(X[j],sigma_1 = sigma_1)
                        if self.score_function(X_iter) > L[k]:
                            S_called_times += 1
                            X[j] = X_iter
                            reject -= 1.
                    rate = reject/np.float(N - ell)
                    #rate = reject/np.float(N) #if we change the elements of I_k to have less dependence

                    sigma_1 -= descent_step
                    if sigma_1 <= 0:
                        break
        ######


            L = np.append(L, np.sort(self.score_function(X))[np.int((1-p_0)*N)])
            k += 1

        N_L = np.sum((self.score_function(X)>self.level))
        S_called_times += N
        p_hat = N_L/float(N)*p_0**(k-1)
    #     L_adapted = L[0:-1]
    #     L_adapted = np.append(L_adapted, q_test)

        if status_tracking ==True:
            print ("final k = " + str(k))
            print ("estimation of p: " + str(p_hat))
            print ('______________________________\n')
            print ("Time spent: %s s" %(time() - t_0) )
            print ("score_function called: %s times" % S_called_times)
        return p_hat, np.int(k-1), N_L/float(N)
