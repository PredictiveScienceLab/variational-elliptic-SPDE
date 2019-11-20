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
import random 
from scipy.stats import truncnorm 

#parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-N', dest = 'N', type = int, 
                    default = 10000, help  = 'Number of MC samples.')
# x=[x1, x2]
parser.add_argument('-nx1', dest = 'nx1', type =  int, 
                    default = 32, help = 'Number of FV cells in the x1 direction.')
parser.add_argument('-nx2', dest = 'nx2', type = int, 
                    default = 32, help = 'Number of FV cells in the x2 direction.')
parser.add_argument('-var', dest = 'var', type = float, 
                    default = 1., help = 'Signal strength (variance) of the random field.')
parser.add_argument('-k', dest = 'k', type = str, 
                    default = 'exp', help = 'Type of covariance kernel (rbf, exp, mat32 or mat52).')
parser.add_argument('-seed', dest = 'seed', type = int, 
                    default = 19, help  = 'Random seed number.')
args = parser.parse_args()
kernels = {'rbf':GPy.kern.RBF, 'exp':GPy.kern.Exponential, 
           'mat32':GPy.kern.Matern32, 'mat52':GPy.kern.Matern52}

num_samples = args.N
nx1 = args.nx1
nx2 = args.nx2
variance = args.var 
k_ = args.k
assert k_ in kernels.keys()
kern = kernels[k_]

os.environ['PYTHONHASHSEED'] = '0'

seed = args.seed
# Setting the seed for numpy-generated random numbers
np.random.seed(seed=seed)

# Setting the seed for python random numbers
random.seed(seed)

#define a mean function
def mean(x):
    """
    Mean of the conductivity field. 

    m(x) = 0. 
    """
    n = x.shape[0]
    return np.zeros((n, 1))


#defining mesh to get cellcenters
Lx1 = 1.  # always put . after 1 
Lx2 = 1.  # always put . after 1 
mesh = Grid2D(nx=nx1, ny=nx2, dx=Lx1/nx1, dy=Lx2/nx2) # with nx1*nx2 number of cells/cellcenters/pixels/pixelcenters
cellcenters = mesh.cellCenters.value.T # (nx1*nx2,2) matrix
np.save('cellcenters_nx1='+str(nx1)+'_nx2='+str(nx2)+'.npy', cellcenters)

# https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
# X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
#define the distribution of truncated normals for lengthscales 
l1rv = truncnorm((0.07-0.1)/0.03, (0.13-0.1)/0.03, 0.1, 0.03)
l2rv = truncnorm((0.47-0.5)/0.03, (0.53-0.5)/0.03, 0.5, 0.03)

lx1s = l1rv.rvs(num_samples)
lx2s = l2rv.rvs(num_samples)
ls = np.array(zip(lx1s, lx2s))

#define matrices to save results 
inputs = np.zeros((num_samples, nx1*nx2))
outputs = np.zeros((num_samples, nx1*nx2))

start = time.time()
#generate samples
for i in xrange(num_samples):
    #display
    if (i+1)%100 == 0:
        print "Generating sample "+str(i+1)

    l1sample = ls[i][0]
    l2sample = ls[i][1]

    #GPy kernel
    k=kern(input_dim = 2,
           lengthscale = [l1sample, l2sample],
           variance = variance,
           ARD = True)

    #get covariance matrix and compute its Cholesky decomposition
    m = mean(cellcenters)
    nugget = 1e-6 # This is a small number required for stability
    Cov = k.K(cellcenters) + nugget * np.eye(cellcenters.shape[0])
    L = np.linalg.cholesky(Cov)
    
    #generate a sample of the random field input
    z = np.random.randn(cellcenters.shape[0], 1)
    f = m + np.dot(L, z)
    sample = np.exp(f) # 'sample' is one image of input field: conductivity image 

    # bounding input fields from below and above
    lower_bound =  np.exp(-5.298317366548036) # 0.005000000000000002
    upper_bound =  np.exp(3.5) # 33.11545195869231

    sample = np.where(sample < lower_bound, lower_bound, sample) 
    sample  = np.where(sample > upper_bound, upper_bound, sample) 

    # FIPY solution
    value_left=1.
    value_right=0.
    value_top=0.
    value_bottom=0.
 
    # define cell and face variables
    phi = CellVariable(name='$T(x)$', mesh=mesh, value=0.)
    D = CellVariable(name='$D(x)$', mesh=mesh, value=1.0) ## coefficient in diffusion equation
    # D = FaceVariable(name='$D(x)$', mesh=mesh, value=1.0) ## coefficient in diffusion equation
    source = CellVariable(name='$f(x)$', mesh=mesh, value=1.0)
    C = CellVariable(name='$C(x)$', mesh=mesh, value=1.0)

    # apply boundary conditions
    # dirichet
    phi.constrain(value_left, mesh.facesLeft)
    phi.constrain(value_right, mesh.facesRight)
    
    # homogeneous Neumann
    phi.faceGrad.constrain(value_top, mesh.facesTop)
    phi.faceGrad.constrain(value_bottom, mesh.facesBottom)
    
    # setup the diffusion problem
    eq = -DiffusionTerm(coeff=D)+ImplicitSourceTerm(coeff=C) == source

    c = 0.
    f = 0. # source

    source.setValue(f)
    C.setValue(c)

    D.setValue(sample.ravel())

    eq.solve(var=phi)
    x_fipy = mesh.cellCenters.value.T ## fipy solution (nx1*nx2,2) matrix # same as cellcenters defined above
    u_fipy = phi.value[:][:, None] ## fipy solution  (nx1*nx2,1) matrix

    #save data 
    inputs[i] = sample.ravel()
    outputs[i] = u_fipy.flatten()   

#end timer
finish = time.time() - start
print "Time (sec) to generate "+str(num_samples)+" MC samples : " +str(finish)

print np.shape(inputs)
print np.shape(outputs)
print inputs
print outputs

#save data
np.save("Uncertain_truncated_normal_sampled_ls.npy",ls)

np.save("MC_samples_inputfield_"+\
            k_+"_nx1="+str(nx1)+\
            "_nx2="+str(nx2)+\
            "_uncertain_ls"+\
            "_v="+str(variance)+\
            "_num_samples="+str(num_samples)+".npy", inputs)
np.save("MC_samples_u_fipy_"+\
            k_+"_nx1="+str(nx1)+\
            "_nx2="+str(nx2)+\
            "_uncertain_ls"+\
            "_v="+str(variance)+\
            "_num_samples="+str(num_samples)+".npy", outputs)

# END