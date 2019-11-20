  #!/usr/bin/env python
from __future__ import division
import argparse 
import numpy as np
import os
import GPy
import matplotlib.pyplot as plt
from fipy import *
from scipy.interpolate import griddata
from pdb import set_trace as keyboard
import time

#parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-N', dest = 'N', type = int, 
                    default = 10000, help  = 'Number of samples of the random inputs.')
# x=[x1, x2]
parser.add_argument('-nx1', dest = 'nx1', type =  int, 
                    default = 32, help = 'Number of FV cells in the x1 direction.')
parser.add_argument('-nx2', dest = 'nx2', type = int, 
                    default = 32, help = 'Number of FV cells in the x2 direction.')
parser.add_argument('-lx1', dest = 'lx1', type = float, 
                    default = 0.02, help = 'Lengthscale of the random field along the x1 direction.')
parser.add_argument('-lx2', dest = 'lx2', type = float, 
                    default = 0.02, help = 'Lengthscale of the random field along the x2 direction.')
parser.add_argument('-var', dest = 'var', type = float, 
                    default = 1., help = 'Signal strength (variance) of the random field.')
parser.add_argument('-k', dest = 'k', type = str, 
                    default = 'exp', help = 'Type of covariance kernel (rbf, exp, mat32 or mat52).')
# used seed 0 for training data, seed 23 for testing data
parser.add_argument('-seed', dest = 'seed', type = int, 
                    default = 0, help  = 'Random seed number.')
args = parser.parse_args()
kernels = {'rbf':GPy.kern.RBF, 'exp':GPy.kern.Exponential, 
           'mat32':GPy.kern.Matern32, 'mat52':GPy.kern.Matern52}

num_samples = args.N
nx1 = args.nx1
nx2 = args.nx2
ellx1 = args.lx1
ellx2 = args.lx2
variance = args.var 
k_ = args.k
assert k_ in kernels.keys()
kern = kernels[k_]
seed = args.seed

np.random.seed(seed=seed)

#define a mean function
def mean(x):
    """
    Mean of the conductivity field. 

    m(x) = 0. 
    """
    n = x.shape[0]
    return np.zeros((n, 1))

#data directory
cwd = os.getcwd()
data='data'
datadir = os.path.abspath(os.path.join(cwd, data))
if not os.path.exists(datadir):
    os.makedirs(datadir)

#GPy kernel
k=kern(input_dim = 2,
       lengthscale = [ellx1, ellx2],
       variance = variance,
       ARD = True)

#defining mesh to get cellcenters
Lx1 = 1.  # always put . after 1 
Lx2 = 1.  # always put . after 1 
mesh = Grid2D(nx=nx1, ny=nx2, dx=Lx1/nx1, dy=Lx2/nx2) # with nx1*nx2 number of cells/cellcenters/pixels/pixelcenters
cellcenters = mesh.cellCenters.value.T # (nx1*nx2,2) matrix
np.save(os.path.join(datadir, 'cellcenters_nx1='+str(nx1)+'_nx2='+str(nx2)+'.npy'), cellcenters)


#get covariance matrix and compute its Cholesky decomposition
m = mean(cellcenters)
nugget = 1e-6 # This is a small number required for stability
Cov = k.K(cellcenters) + nugget * np.eye(cellcenters.shape[0])
L = np.linalg.cholesky(Cov)

#define matrices to save results 
inputs = np.zeros((num_samples, nx1*nx2))

start = time.time()
#generate samples
for i in xrange(num_samples):
    #display
    if (i+1)%100 == 0:
        print "Generating sample "+str(i+1)
    
    #generate a sample of the random field input
    z = np.random.randn(cellcenters.shape[0], 1)
    f = m + np.dot(L, z)
    sample = np.exp(f)
    #save data 
    inputs[i] = sample.ravel()

#end timer
finish = time.time() - start
print "Time (sec) to generate "+str(num_samples)+" samples : " +str(finish)
print inputs

#save data
datafile = k_+"_nx1="+str(nx1)+\
            "_nx2="+str(nx2)+\
            "_lx1="+str(ellx1)+\
            "_lx2="+str(ellx2)+\
            "_v="+str(variance)+\
            "_num_samples="+str(num_samples)+".npy"
np.save(os.path.join(datadir,datafile), inputs)
