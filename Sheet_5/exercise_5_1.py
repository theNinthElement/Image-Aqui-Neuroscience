import numpy as np
import scipy.stats
import pandas as pd

study = pd.read_excel('bipolar.xls')
study = study.to_numpy()

medicine = study[np.equal(np.rint(study[:,3]).astype(int),1)]
placebo = study[np.equal(study[:,3],0)]
delta_medicine = medicine[:,2]-medicine[:,1]
delta_placebo = placebo[:,2]-placebo[:,1]

########################################################################
# a)
########################################################################
# Model: 
#		both groups (medicine, placebo) Gaussian distributed with 
# 		assumption of equal standard deviation (-> or not: to do: Add Welch's modification and argparse flag for mode)
#
# null hypothesis:
#		means equal for both groups
#
# method:
#		two sided (assuming no prior knowledge about sign of deviation)
#		two-sample (since both group means have to be estimated) t-test 
print('\n------------a------------\n')

########################################################################
# b)
########################################################################
print('\n------------b------------\n')

t_test=scipy.stats.ttest_ind(delta_medicine, delta_placebo, nan_policy='raise')
print(t_test)
print('Null hypothesis rejected: {}\n'.format(t_test[1]<0.05))
# is rejeceted if alpha = 5% (qu.: what is the correct name of the parameter alpha?)
# p-value = 0.96%
print('Average increase in amygdala size: {}\n'.format(np.mean(delta_medicine)-np.mean(delta_placebo)))
# 0.064 increase

########################################################################
# c)
########################################################################
print('\n------------c------------\n')
# use one-way analysis of variance (one way ANOVA), since no factorial design
anova=scipy.stats.f_oneway(delta_medicine, delta_placebo)
print(anova)
print('Null hypothesis rejected: {}\n'.format(t_test[1]<0.05))
# is rejeceted if alpha = 5%, p-value = 0.96%
# since we have exactly two groups and were using a two-sided two-sample 
# test in a), the oneway ANOVA should yield the same p value:

print('\nMore info and answers: see code\n')

