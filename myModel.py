import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from GPmodels import GPclassifier, MFGPclassifier, SMFGPclassifier, SGPclassifier
from scipy.cluster.vq import kmeans
from myPlot import plot_multiclass
import sys

plt.rcParams['image.cmap'] = 'coolwarm'
np.random.seed(2)


# Define the functions used in the first example of the publication

# In[11]:


def low_fidelity(X):
     return (0.5 + np.sin(2.2 * X[:, 0] * np.pi) / 5 - X[:, 1]) > 0
    #return (0.5 + np.sin(2.2 * X[:, 0] * np.pi) / 2.5 - X[:, 1]) > 0


def high_fidelity(X):
     return (0.48 + np.sin(2.5 * X[:, 0] * np.pi) / 6 - X[:, 1]) > 0
    #return (0.48 + np.sin(2.5 * X[:, 0] * np.pi) / 3 - X[:, 1]) > 0

def low_fidelity_bis(X):
    return (0.55 - np.sin(2.3 * X[:, 1] * np.pi) / 10 - X[:, 0]) > 0
    #return (0.65 - np.sin(2.3 * X[:, 1] * np.pi) / 10 - X[:, 0]) > 0


def high_fidelity_bis(X):
    return (0.5 - np.sin(2.2 * X[:, 1] * np.pi) / 12 - X[:, 0]) > 0
    #return (0.7 - np.sin(2 * X[:, 1] * np.pi) / 12 - X[:, 0]) > 0



def low_fidelity_boundary(X):
    return (0.5 + np.sin(2.2 * X[:, 0] * np.pi) / 5)
    #return (0.5 + np.sin(2.2 * X[:, 0] * np.pi) / 2.5)


def high_fidelity_boundary(X):
    return (0.48 + np.sin(2.5 * X[:, 0] * np.pi) / 6)
    #return (0.48 + np.sin(2.5 * X[:, 0] * np.pi) / 3)

def low_fidelity_boundary_bis(X):
    return (0.55 - np.sin(2.3 * X[:, 0] * np.pi) / 10)
    #return (0.65 - np.sin(2.3 * X[:, 0] * np.pi) / 10)


def high_fidelity_boundary_bis(X):
    return (0.5 - np.sin(2.2 * X[:, 0] * np.pi) / 12)
    #return (0.7 - np.sin(2 * X[:, 0] * np.pi) / 12)


# Generate some data

# In[3]:
def invlogit(x, eps=sys.float_info.epsilon):
    return (1.0 + 2.0 * eps) / (1.0 + np.exp(-x)) + eps


# upper and lower bounds of the data. Used for normalization
# and to set the parameter space in the active learning case
lb = np.array([0, 0])
ub = np.array([1, 1])

N_L = 50  # 50  number of low fidelity points
N_H = 12  # 12  number of high fidelity points


X_L = lhs(2, N_L)  # generate data with a Latin hypercube desing


labels_low = low_fidelity(X_L)
Y_L = 1.0 * labels_low


# separation w.r.t. function 1

# to generate the high fidelity data we choose some points
# from both of classes in the low fidelity data
ind1 = np.where(Y_L > 0)[0]
ind0 = np.where(Y_L == 0)[0]

X_H1 = X_L[np.random.choice(ind1, N_H / 2, replace=False)]
X_H0 = X_L[np.random.choice(ind0, N_H / 2, replace=False)]

X_H = np.concatenate((X_H1, X_H0))

labels_high = high_fidelity(X_H)
Y_H = 1.0 * labels_high


X_test = lhs(2, 1000)  # to test the accuracy

'''
# ... and plot it

# In[4]:

fig = plt.figure()
fig.set_size_inches((8, 8))

plt.scatter(X_L[:, 0], X_L[:, 1], c=Y_L, marker='x', label='low-fidelity data')
plt.scatter(X_H[:, 0], X_H[:, 1], c=Y_H, label='high-fidelity data')

x = np.linspace(0, 1, 100)[:, None]

plt.plot(x, low_fidelity_boundary(x), 'k--', label='low-fidelity boundary')
plt.plot(x, high_fidelity_boundary(x), 'k', label='high-fidelity boundary')

#plt.plot(low_fidelity_boundary_bis(x),x, 'r--', label='low-fidelity boundary 2')
#plt.plot(high_fidelity_boundary_bis(x),x, 'r', label='high-fidelity boundary 2')

plt.legend(frameon=False)

plt.show()
'''


#separation w.r.t. function 2


labels_low_bis = low_fidelity_bis(X_L)
Y_L_bis = 1.0 * labels_low_bis

# to generate the high fidelity data we choose some points
# from both of classes in the low fidelity data
ind1_bis = np.where(Y_L_bis > 0)[0]
ind0_bis = np.where(Y_L_bis == 0)[0]

X_H0_bis = X_L[np.random.choice(ind0_bis, N_H / 2, replace=False)]
X_H1_bis = X_L[np.random.choice(ind1_bis, N_H / 2, replace=False)]

X_H_bis = np.concatenate((X_H0_bis, X_H1_bis))

labels_high_bis = high_fidelity_bis(X_H_bis)
Y_H_bis = 1.0 * labels_high_bis

'''
# ... and plot it

# In[4]:


fig = plt.figure()
fig.set_size_inches((8, 8))

plt.scatter(X_L[:, 0], X_L[:, 1], c=Y_L_bis, marker='x', label='low-fidelity data')
plt.scatter(X_H_bis[:, 0], X_H_bis[:, 1], c=Y_H_bis, label='high-fidelity data')

x = np.linspace(0, 1, 100)[:, None]

plt.plot(low_fidelity_boundary_bis(x),x, 'r--', label='low-fidelity boundary 2')
plt.plot(high_fidelity_boundary_bis(x),x, 'r', label='high-fidelity boundary 2')

plt.legend(frameon=False)

plt.show()
'''


# classes w.r.t. two boundaries
labels_low_tot = (low_fidelity(X_L) + 2 * low_fidelity_bis(X_L))/4.
Y_L_tot = 1.0 * labels_low_tot


ind_tot_1 = np.intersect1d(ind0, ind0_bis)
ind_tot_2 = np.intersect1d(ind0, ind1_bis)
ind_tot_3 = np.intersect1d(ind1, ind0_bis)
ind_tot_4 = np.intersect1d(ind1, ind1_bis)

X_H1_tot = X_L[np.random.choice(ind_tot_1, N_H / 4, replace=False)]
X_H2_tot = X_L[np.random.choice(ind_tot_2, N_H / 4, replace=False)]
X_H3_tot = X_L[np.random.choice(ind_tot_3, N_H / 4, replace=False)]
X_H4_tot = X_L[np.random.choice(ind_tot_4, N_H / 4, replace=False)]

X_H_tot = np.concatenate((X_H1_tot, X_H2_tot, X_H3_tot, X_H4_tot))

labels_high_tot = (high_fidelity(X_H_tot) + 2 * high_fidelity_bis(X_H_tot))/4.
Y_H_tot = 1.0 * labels_high_tot



# ... and plot it

# In[4]:


fig = plt.figure()
fig.set_size_inches((8, 8))

plt.scatter(X_L[:, 0], X_L[:, 1], c=Y_L_tot, marker='x', label='low-fidelity data')
plt.scatter(X_H_tot[:, 0], X_H_tot[:, 1], c=Y_H_tot, label='high-fidelity data')


x = np.linspace(0, 1, 100)[:, None]

plt.plot(x, low_fidelity_boundary(x), 'k--', label='low-fidelity boundary')
plt.plot(x, high_fidelity_boundary(x), 'k', label='high-fidelity boundary')

plt.plot(low_fidelity_boundary_bis(x),x, 'r--', label='low-fidelity boundary 2')
plt.plot(high_fidelity_boundary_bis(x),x, 'r', label='high-fidelity boundary 2')

plt.legend(frameon=False)

plt.show()
########################################################################################

#GPC

GPc = GPclassifier(X_L, Y_L, lb, ub, low_fidelity, X_test = X_test)
GPc.create_model()
GPc.sample_model()
GPc.plot()

h = GPc.h

pred = GPc.pred_samples_grid

GPc_bis = GPclassifier(X_L, Y_L_bis, lb, ub, low_fidelity_bis, X_test = X_test)
GPc_bis.create_model()
GPc_bis.sample_model()
GPc_bis.plot()

pred_bis = GPc_bis.pred_samples_grid

plot_multiclass (pred, pred_bis, h, X_L, Y_L_tot ,high_fidelity_boundary, high_fidelity_boundary_bis, filename='GP_multiclass.png')

##############################################################################
'''
#SGPC sparse

X_Lu = kmeans(X_L,30)[0] # k-means clustering to obtain the position of the inducing points
SGPc = SGPclassifier(X_L, Y_L, X_Lu, lb, ub, low_fidelity, X_test = X_test)
SGPc.create_model()
SGPc.sample_model()
SGPc.plot() # only works for 2D

h = SGPc.h

pred_s = SGPc.pred_samples_grid

SGPc_bis = SGPclassifier(X_L, Y_L_bis, X_Lu, lb, ub, low_fidelity_bis, X_test = X_test)
SGPc_bis.create_model()
SGPc_bis.sample_model()
SGPc_bis.plot()

pred_s_bis = SGPc_bis.pred_samples_grid

plot_multiclass (pred_s, pred_s_bis, h, X_L, Y_L_tot,high_fidelity_boundary, high_fidelity_boundary_bis,filename = 'SGP_multiclass.png' )
'''
##################################################################################
'''
#MFGP multifidelity

MFGPc = MFGPclassifier(X_L, Y_L, X_H, Y_H, lb, ub, high_fidelity, X_test = X_test)
MFGPc.create_model()
MFGPc.sample_model()
MFGPc.plot()

h = MFGPc.h

pred_mf = MFGPc.pred_samples_grid

MFGPc_bis = MFGPclassifier(X_L, Y_L_bis, X_H_bis, Y_H_bis, lb, ub, high_fidelity_bis, X_test = X_test)
MFGPc_bis.create_model()
MFGPc_bis.sample_model()
MFGPc_bis.plot()

pred_mf_bis = MFGPc_bis.pred_samples_grid

plot_multiclass (pred_mf, pred_mf_bis, h, X_H_tot, Y_H_tot,high_fidelity_boundary, high_fidelity_boundary_bis, filename = 'MFGP_multiclass.png' )
'''
#####################################################################
'''
#SMFGP sparse multifidelity

X_Lu = kmeans(X_L,30)[0] # k-means clustering to obtain the position of the inducing points
X_Hu = X_H # we use the high fidelity points as inducing points as well because there only 10.

SMFGPc = SMFGPclassifier(X_L, Y_L, X_H, Y_H, X_Lu, X_Hu, lb, ub, high_fidelity, X_test = X_test)
SMFGPc.create_model()
SMFGPc.sample_model()
SMFGPc.plot()

h = SMFGPc.h

pred_smf = SMFGPc.pred_samples_grid

X_Hu_bis = X_H_bis

SMFGPc_bis = SMFGPclassifier(X_L, Y_L_bis, X_H_bis, Y_H_bis, X_Lu, X_Hu_bis, lb, ub, high_fidelity_bis, X_test = X_test)
SMFGPc_bis.create_model()
SMFGPc_bis.sample_model()
SMFGPc_bis.plot()

pred_smf_bis = SMFGPc_bis.pred_samples_grid

plot_multiclass (pred_smf, pred_smf_bis, h, X_H_tot, Y_H_tot,high_fidelity_boundary, high_fidelity_boundary_bis,filename = 'SMFGP_multiclass.png' )

'''





