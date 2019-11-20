#!/usr/bin/env python
import argparse 
from heateqsolver import SteadyStateHeat2DSolver
import numpy as np
import os
import GPy
import matplotlib.pyplot as plt
from fipy import *
from scipy.interpolate import griddata
from pdb import set_trace as keyboard
import time
# from kle import KarhunenLoeveExpansion
import subprocess 
from multiprocessing import Pool
np.random.seed(1)

def func(case):
    cmd = ['python', 'generate_data.py', '-k', case[0], \
           '-nx', str(case[1]),\
           '-ny', str(case[2]),\
           '-lx', str(case[3]),\
           '-ly', str(case[4]),\
           '-N', str(case[5])]
    assert subprocess.call(cmd) == 0, 'Failed to run case: '+str(case)


if __name__ == '__main__':
    #covfuncs = ['mat52', 'rbf', 'mat32', 'exp']
    covfuncs = ['exp']
    # # generate data at design lengthscales
    # Lgrid = []
    # c = 0
    # n_lx = 60 #number of lengthscale pairs 
    # while True:
    #    u = np.random.rand(3)
    #    if u[2] > np.exp(-1.5*np.sum(u[:2])):
    #        continue
    #    else:
    #        Lgrid.append(u[:2])
    #        c += 1
    #        if c == n_lx:
    #            break 
    # Lgrid = np.array(Lgrid)
    # Lgrid = 0.035 + Lgrid*(1. - 0.035)
    # Neach = 150   # Neach = 500 for training dataset # Neach = 150 for testing dataset 
    # N = Lgrid.shape[0] * Neach

    # generate test data from arbitrary lengthscales 
    Neach = 100
    lx = np.linspace(0.035, 1., 10)
    Lx, Ly = np.meshgrid(lx, lx)
    Lgrid = np.hstack([Lx.flatten()[:, None], Ly.flatten()[:, None]])
    
    #grid size 
    nx = 32
    ny = 32
    cases = [(k, nx, ny, l[0], l[1], Neach) for k in covfuncs for l in Lgrid]

    #generate some data
    pool = Pool(4)
    pool.map(func, cases)

    for k in covfuncs:
        i = 0
        datafiles = [x for x in os.listdir(os.path.join(os.getcwd(), 'data')) if k in x and 'lx' in x]
        for datafile in datafiles:
            data = np.load(os.path.join(os.getcwd(), 'data', datafile))
            if i == 0:
                inputs = data['inputs']
                outputs = data['outputs']
                Ellx = [float(data['lx'])]
                Elly = [float(data['ly'])]
            else:
                inputs = np.vstack([inputs, data['inputs']])
                outputs = np.vstack([outputs, data['outputs']])
                Ellx.append(float(data['lx']))
                Elly.append(float(data['ly']))
            i += 1
        # np.savez(os.path.join(os.getcwd(), 
        #          'data', 'train_data_var0.75_'+k+'.npz'), inputs = inputs, 
        #                                           outputs = outputs,
        #                                           lx = np.array(Ellx),
        #                                           ly = np.array(Elly),
        #                                           Neach = Neach)
        # np.savez(os.path.join(os.getcwd(), 
        #          'data', 'test_data_var0.75_'+k+'.npz'), inputs = inputs, 
        #                                           outputs = outputs,
        #                                           lx = np.array(Ellx),
        #                                           ly = np.array(Elly),
        #                                           Neach = Neach)
        np.savez(os.path.join(os.getcwd(), 
                 'data', 'test_arbitrary_data_var0.75_'+k+'.npz'), inputs = inputs, 
                                                  outputs = outputs,
                                                  lx = np.array(Ellx),
                                                  ly = np.array(Elly),
                                                  Neach = Neach)







