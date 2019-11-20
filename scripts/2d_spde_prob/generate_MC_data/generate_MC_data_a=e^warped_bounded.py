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

seed=19
os.environ['PYTHONHASHSEED'] = '0'
# Setting the seed for numpy-generated random numbers
np.random.seed(seed=seed)
# Setting the seed for python random numbers
random.seed(seed)

num_samples=100000    # Number of MC samples.

nx1=32
nx2=32

kern_1=GPy.kern.RBF
ellx1_1=2
ellx2_1=2
variance_1=0.25

kern_2=GPy.kern.RBF
ellx1_2=0.1
ellx2_2=0.1
variance_2=0.75

#define a mean function
def mean_1(x):
    return x

#define a mean function
def mean_2(x):
    n = x.shape[0]
    return np.zeros((n, 1))

#GPy kernel
k_1=kern_1(input_dim = 2,
       lengthscale = [ellx1_1, ellx2_1],
       variance = variance_1,
       ARD = True)

# GPy kernel
k_2=kern_2(input_dim = 2,
       lengthscale = [ellx1_2, ellx2_2],
       variance = variance_2,
       ARD = True)

#defining mesh to get cellcenters
Lx1 = 1.  # always put . after 1 
Lx2 = 1.  # always put . after 1 
mesh = Grid2D(nx=nx1, ny=nx2, dx=Lx1/nx1, dy=Lx2/nx2) # with nx1*nx2 number of cells/cellcenters/pixels/pixelcenters
cellcenters = mesh.cellCenters.value.T # (nx1*nx2,2) matrix

np.save('cellcenters_nx1='+str(nx1)+'_nx2='+str(nx2)+'.npy', cellcenters)

print cellcenters

#define matrices to save results 
inputs = np.zeros((num_samples, nx1*nx2))
outputs = np.zeros((num_samples, nx1*nx2))

start = time.time()

#generate samples
for i in xrange(num_samples):
    #display
    if (i+1)%10000 == 0:
        print "Generating sample "+str(i+1)

    #get covariance matrix and compute its Cholesky decomposition
    m_1 = mean_1(cellcenters)
    nugget = 1e-6 # This is a small number required for stability
    Cov_1 = k_1.K(cellcenters) + nugget * np.eye(cellcenters.shape[0])
    L_1 = np.linalg.cholesky(Cov_1)

    #generate a sample 
    z_1 = np.random.randn(cellcenters.shape[0], 1)
    f_1 = m_1 + np.dot(L_1, z_1)

#     print f_1
#     print np.shape(f_1)
    
    #get covariance matrix and compute its Cholesky decomposition
    m_2 = mean_2(f_1)
    nugget = 1e-6 # This is a small number required for stability
    Cov_2 = k_2.K(f_1) + nugget * np.eye(f_1.shape[0])
    L_2 = np.linalg.cholesky(Cov_2)

    #generate a sample 
    z_2 = np.random.randn(f_1.shape[0], 1)
    f_2 = m_2 + np.dot(L_2, z_2)

#     print f_2
#     print np.shape(f_2)

    sample = np.exp(f_2)# 'sample' is one image of input field: conductivity image

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
np.save("MC_samples_inputfield_warped_double_rbf"+\
            "_nx1="+str(nx1)+\
            "_nx2="+str(nx2)+\
            "_num_samples="+str(num_samples)+".npy", inputs)
np.save("MC_samples_u_fipy_warped_double_rbf"+\
            "_nx1="+str(nx1)+\
            "_nx2="+str(nx2)+\
            "_num_samples="+str(num_samples)+".npy", outputs)

# END