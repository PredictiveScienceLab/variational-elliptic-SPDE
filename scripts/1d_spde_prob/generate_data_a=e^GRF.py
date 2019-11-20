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
parser.add_argument('-nx', dest = 'nx', type =  int, 
                    default = 100, help = 'Number of FV cells in the x direction.')
parser.add_argument('-lx', dest = 'lx', type = float, 
                    default = 0.02, help = 'Lengthscale of the random field along the x direction.')
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
nx = args.nx
ellx = args.lx
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
k = kern(1, lengthscale = ellx, variance = variance)

#defining mesh to get cellcenters
Lx = 1.  # always put . after 1 
mesh = Grid1D(nx=nx, dx=Lx/nx) # with nx number of cells/cellcenters/pixels/pixelcenters
cellcenters = mesh.cellCenters.value.T  #(nx,1) matrix
np.save(os.path.join(datadir, 'cellcenters_nx='+str(nx)+'.npy'), cellcenters)


#get covariance matrix and compute its Cholesky decomposition
m = mean(cellcenters)
nugget = 1e-6 # This is a small number required for stability
Cov = k.K(cellcenters) + nugget * np.eye(cellcenters.shape[0])
L = np.linalg.cholesky(Cov)

#define matrices to save results 
inputs = np.zeros((num_samples, nx))

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
datafile = k_+"_nx="+str(nx)+\
            "_lx="+str(ellx)+\
            "_v="+str(variance)+\
            "_num_samples="+str(num_samples)+".npy"
np.save(os.path.join(datadir,datafile), inputs)
