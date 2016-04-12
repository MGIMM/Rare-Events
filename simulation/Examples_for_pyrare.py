
# coding: utf-8

#

from pyrare import *



# # fixed_levels & adaptive_levels algorithm
#
#
#
#     fixed_levels(self, N, reject_rate = 0.5, sigma_default = 0.5,
#                                 descent_step = 0.08, shake_times = 5, status_tracking = False)
#
#
#     adaptive_levels(self,N, shake_times = 5,
#                                   reject_rate = 0.3, sigma_default = 0.5,
#                                 descent_step = 0.02, status_tracking = False)

#
print ("eg.1 We want to calculate level = 3 with fixed_leve1 method: ")
rare_test = RareEvents_test(level = 3)
rare_test.fixed_levels(N = 100,status_tracking=True)


#
# # Test Fuction
#     RareEvents_test.test(self,list_N = [100,500,1000,3000], n_sim = 100, shake_times = 5,
#                                   reject_rate = 0.5, sigma_default = 0.5,
#                                 descent_step = 0.02, method ="adaptive",hist_plot = True)

#
print ("eg.2 We want to calculate level = 3 with adaptive_leve1 method: ")
rare_test.adaptive_levels(N = 2000,status_tracking=True)


#
print ("eg.3 We want to draw a simulation(n_sim = 100, level = 3) with fixed_leve1 method, we would like to see the Histograms. ")
rare_test.test(list_N =[5,10,30], method = 'fixed',hist_plot = True)


#
print ("eg.4 We want to draw a simulation(n_sim = 100, level = 3) with fixed_leve1 method, we don't want to see the Histograms. ")
rare_test.test(list_N =[5,10,30], method = 'adaptive',hist_plot=False)


#

print ("eg.5 We want to draw a simulation(n_sim = 100, level = 3) with both two algoritms and we want to compare them. ")
rare_test.test(list_N =[5,10,30], method = 'both')


#
