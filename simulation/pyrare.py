import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from scipy.stats import norm
from time import time


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
        print ("sequence of levels: "+ str(L_ideal))
        print ("num_lev: "+ str(num_lev))
        print ("level interested, L = "+ str(q_test))
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



class RareEvents_test:
    def __init__(self, mu_0 = mu_0_test, score_function = S_test,
                 shaker = shaker_test, param = None, level = 8, p_0 = 0.75):
        """
        :param param: real_p, theoretical relative standard deviation,
        sequence of levels for fixed levels method
        :param level: the level to estimate
        :param p_0: the successful rate
        :param mu_0: the distribution of X_0
        :param score_function: the black box score_function, default = np.abs
        :param shaker: metro-polis kernel
        """
        if param == None:
            self.param = get_params_test(q_test = level, p_0 = p_0 ,status_tracking = False)
        self.mu_0 = mu_0_test
        self.score_function = S_test
        self.shaker = shaker_test
        self.level = level
        self.p_0 = p_0



    def fixed_levels(self, N, reject_rate = 0.5, sigma_default = 0.5,
                                descent_step = 0.08, shake_times = 5, status_tracking = False):

        t_0 = time()
        L = self.param[1]
        real_p = self.param[0]
        num_lev = len(L)

        list_p_hat = []
        index_finish = False ###### to ensure that I_k won't be empty
        while(index_finish == False):
            X= self.mu_0(N)
            for k in range(num_lev - 1):


            ###### construction of I_k
                I = [X[j] for j in range(N) if self.score_function(X[j])>L[k+1]]
                ell = len(I)
                if ell == 0:
                    break



            ###### estimation of p_k
                list_p_hat = np.append(list_p_hat, ell/np.float(N))

            ###### start of transition
                X[0:ell] = I
            ###### permutation trick
                if ell != N:

                    I = np.random.permutation(I)
                    for j in range(ell,N,1):
                        X[j] = I[j%ell]
                ###### we only accept the transition in A_{k+1} / shaker

                ###### we conduct an adaptive method to choose the sigma_1 for shaker
                ###### by controling the reject_rate of the transition in A_{k+1}

                    for index_shaker in range(shake_times):
                        rate = 1.
                        sigma_1 = sigma_default
                        while(rate>reject_rate):

                            reject = N - ell
                            for j in range(ell,N,1):  #if we don't change the elements of I_k
                            #for j in range(N):       #if we change the elements of I_k to have less dependence

                                X_iter = self.shaker(X[j],sigma_1 = sigma_1)
                                if self.score_function(X_iter) > L[k+1]:
                                    X[j] = X_iter
                                    reject -= 1.
                            rate = reject/np.float(N - ell)
                            #rate = reject/np.float(N)

                            sigma_1 -= descent_step
                            if sigma_1 <= 0:
                                break



        ######
            index_finish = True

        ###### end of transition

        ###### estimation of p
        p_hat = np.prod(list_p_hat)
        ###### relative variation
        rel_var = (p_hat - real_p)/real_p
        ###### tracking status
        if status_tracking == True:
            print ("levels: " + str(L))
            print ("real value of p:" + str(real_p))
            print ("estimation of p: " + str(p_hat))
            print ("sqrt(N) * relative variation: " + str(rel_var*np.sqrt(N)))
            print ("N: " + str(N))
            print ("Time spent :"+ str(time() - t_0)+"s")

        return p_hat, rel_var


    def adaptive_levels(self,N, shake_times = 5,
                                  reject_rate = 0.3, sigma_default = 0.5,
                                descent_step = 0.02, status_tracking = False):
        ###### Initiation
        t_0 = time()
        X = self.mu_0(N)
        real_p = self.param[0]
        p_0 = self.p_0
        L = np.array([-np.Inf,np.sort(self.score_function(X))[np.int((1-p_0)*N)]])
        k = 1



        ######

        while(L[k]<self.level):
    #         if status_track == True:
    #             print ("\t")
    #             print ("k = " + str(k))
    #             print ('current level: ' + str(L[k]))
            I = []
            for i in range(N):
                if self.score_function(X[i])>L[k]:
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
        p_hat = N_L/float(N)*p_0**(k-1)
        rel_var = (p_hat-real_p)/real_p
    #     L_adapted = L[0:-1]
    #     L_adapted = np.append(L_adapted, q_test)

        if status_tracking ==True:
            print ("final k = " + str(k))
            print ("real value of p:" + str(real_p))
            print ("estimation of p: " + str(p_hat))
            print ("sqrt(N) * relative variation: " + str(rel_var * np.sqrt(N)))
            print ("Time spent :"+ str(time() - t_0)+"s")
        return p_hat, rel_var


    def test(self,list_N = [100,500,1000,3000], n_sim = 100, shake_times = 5,
                                  reject_rate = 0.5, sigma_default = 0.5,
                                descent_step = 0.02, method ="adaptive",hist_plot = True):
        n_choice_N = len(list_N)


        sim_N =[[ [] for j in range(n_sim)] for i in range(n_choice_N)]
        print ("Total number of simulation: " + str(n_choice_N))

        if method == "adaptive":
            t_0 = time()
            for index_sim in range(n_choice_N):
                for i in range(n_sim):
                    sim_N[index_sim][i] = self.adaptive_levels( N = list_N[index_sim], shake_times = 5,
                                                               reject_rate = 0.5, sigma_default = 0.5,
                                                               descent_step = 0.1, status_tracking = False)
                print ("simulation completed: "+ str(index_sim+1))
            print ("Time spent :"+ str(time() - t_0)+"s")

        elif method == 'fixed':
            t_0 = time()
            for index_sim in range(n_choice_N):
                for i in range(n_sim):
                        sim_N[index_sim][i] = self.fixed_levels( N = list_N[index_sim], shake_times = 5,
                                                                reject_rate = 0.5, sigma_default = 0.5,
                                                                descent_step = 0.1, status_tracking = False)
                print ("simulation completed: "+ str(index_sim+1))
            print ("Time spent :"+ str(time() - t_0)+"s")

        elif method == 'both':
            t_0 = time()
            sim_N_adaptive =[[ [] for j in range(n_sim)] for i in range(n_choice_N)]
            sim_N_fixed =[[ [] for j in range(n_sim)] for i in range(n_choice_N)]

            for index_sim in range(n_choice_N):
                for i in range(n_sim):
                    sim_N_adaptive[index_sim][i] = self.adaptive_levels( N = list_N[index_sim], shake_times = 5,
                                                                        reject_rate = 0.5, sigma_default = 0.5,
                                                                        descent_step = 0.1, status_tracking = False)
                    sim_N_fixed[index_sim][i] = self.fixed_levels( N = list_N[index_sim], shake_times = 5,
                                                                  reject_rate = 0.5, sigma_default = 0.5,
                                                                  descent_step = 0.1, status_tracking = False)
                print ("simulation completed: "+ str(index_sim+1))
            print ("Time spent :"+ str(time() - t_0)+"s")

            sim_N_adaptive = np.array(sim_N_adaptive)
            sim_N_fixed = np.array(sim_N_fixed)

        if method != 'both':
            sim_N = np.array(sim_N)

            print ('\t.............')
            sigma_theoretical = self.param[2]
            estimation_sim = np.array([sim_N[i][:,0] for i in range(n_choice_N)])
            sigma_relative_variation_sim = np.array([sim_N[i][:,1] * np.sqrt(list_N[i]) for i in range(n_choice_N)])
            print ("theoretical std of relative variation: " + str(sigma_theoretical))
            std_sqrtN = np.array([np.std(sigma_relative_variation_sim[i]) for i in range(n_choice_N)])
            print ("sqrt N * std of relative variation : " + str(std_sqrtN ))

            if hist_plot == True:
                plt.figure(figsize = [10,int(n_choice_N*4)])

                for i in range(n_choice_N):
                    plt.subplot(n_choice_N,1,i+1)

                    sns.distplot(estimation_sim[i] , label = "estimation",color = "grey")
                    plt.title('Histogram of estimation (N ='+str(list_N[i])+', q_test = ' + str(self.level) +')')
                    x = np.arange(-1,1,0.1)
                    plt.plot(self.param[0],0,marker = "o",color = 'darkred',label = "real value")
                    plt.legend()
                plt.show()

                print('\t.............')
                plt.figure(figsize = [10,int(n_choice_N*4)])

                for i in range(n_choice_N):
                    plt.subplot(n_choice_N,1,i+1)

                    sns.distplot(sigma_relative_variation_sim[i] , label = "empirical", color = "grey")
                    plt.title('Histogram of relative variation(sigma) (N ='+str(list_N[i])+', q_test = ' + str(self.level) +')')
                    x = np.arange(-50,50,0.5)
                    plt.plot(x,norm.pdf(x,0,sigma_theoretical), label = "idealized", color = "darkred")
                    plt.legend()
                plt.show()

            print('\t.............')
            plt.figure(figsize = [10,5])
            plt.plot(np.log(list_N), np.log(std_sqrtN/ np.sqrt(list_N)), label = "simulated", marker = 'o',color = "grey")
            plt.plot(np.log(list_N),  np.log(sigma_theoretical /np.sqrt(list_N)), label = "theoretical", marker = 'o' ,color = "darkred" )
            plt.legend()
            plt.title("Relative standard variation (log-log) for (N = "+str(list_N)+', q_test = ' + str(self.level) +')')
            plt.show()


        elif method == 'both':
            sigma_theoretical = self.param[2]
            estimation_sim_fixed = np.array([sim_N_fixed[i][:,0] for i in range(n_choice_N)])
            sigma_relative_variation_sim_fixed = np.array([sim_N_fixed[i][:,1] * np.sqrt(list_N[i]) for i in range(n_choice_N)])

            estimation_sim_adaptive = np.array([sim_N_adaptive[i][:,0] for i in range(n_choice_N)])
            sigma_relative_variation_sim_adaptive = np.array([sim_N_adaptive[i][:,1] * np.sqrt(list_N[i]) for i in range(n_choice_N)])

            print ("theoretical std of relative variation: " + str(sigma_theoretical))
            std_sqrtN_fixed = np.array([np.std(sigma_relative_variation_sim_fixed[i]) for i in range(n_choice_N)])
            std_sqrtN_adaptive = np.array([np.std(sigma_relative_variation_sim_adaptive[i]) for i in range(n_choice_N)])
            print ('\t')
            print ("sqrt N * std of relative variation (fixed) : " + str(std_sqrtN_fixed ))
            print ("sqrt N * std of relative variation (adaptive) : " + str(std_sqrtN_adaptive ))
            print('\t')
            plt.figure(figsize = [10,5])
            plt.plot(np.log(list_N), np.log(std_sqrtN_fixed/ np.sqrt(list_N)), label = "fixed_levels", marker = '.')
            plt.plot(np.log(list_N), np.log(std_sqrtN_adaptive/ np.sqrt(list_N)), label = "adaptive_levels", marker = '.')
            plt.plot(np.log(list_N),  np.log(sigma_theoretical /np.sqrt(list_N)), label = "theoretical", marker = '.' ,color = "darkred" )
            plt.legend()
            plt.title("Relative standard variation (log-log) for (N = "+str(list_N)+', level_test = ' + str(self.level) +')')
            plt.show()

            if hist_plot == True:
                plt.figure(figsize = [10,int(n_choice_N*4)])

                for i in range(n_choice_N):
                    plt.subplot(n_choice_N,1,i+1)

                    sns.distplot(estimation_sim_fixed[i] , label = "fixed_levels")
                    sns.distplot(estimation_sim_adaptive[i] , label = "adaptive_levels")
                    plt.title('Histogram of estimation (N ='+str(list_N[i])+', q_test = ' + str(self.level) +')')
                    x = np.arange(-1,1,0.1)
                    plt.plot(self.param[0],0,marker = "o",color = 'darkred',label = "real value")
                    plt.legend()
                plt.show()

                print('\t.............')
                plt.figure(figsize = [10,int(n_choice_N*4)])

                for i in range(n_choice_N):
                    plt.subplot(n_choice_N,1,i+1)

                    sns.distplot(sigma_relative_variation_sim_fixed[i] , label = "fixed_levels")
                    sns.distplot(sigma_relative_variation_sim_adaptive[i] , label = "adaptive_levels")
                    plt.title('Histogram of relative variation(sigma) (N ='+str(list_N[i])+', level_test = ' + str(self.level) +')')
                    x = np.arange(-50,50,0.5)
                    plt.plot(x,norm.pdf(x,0,sigma_theoretical), label = "idealized", color = "darkred")
                    plt.legend()
                plt.show()
