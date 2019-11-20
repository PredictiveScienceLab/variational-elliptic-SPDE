#!/usr/bin/env python
"""
We define a class ``SteadyStateHeat2DSolver'' which
solves the steady state heat equation
in a two-dimensional square grid.

For now we don't add any forcing function and the
boundary conditions are Dirichlet.

"""


__all__ = ['SteadyStateHeat2DSolver']

import fipy
import numpy as np
from pdb import set_trace as keyboard
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 


class SteadyStateHeat2DSolver(object):
    
    """
    Solves the 2D steady state heat equation with dirichlet boundary conditions.
    It uses the stochastic model we developed above to define the random conductivity.
    _
    Arguments:
    nx           -    Number of grid points along x direction.
    ny           -    Number of grid points along y direction.
    value_left   -    The value at the left side of the boundary.
    value_right  -    The value at the right side of the boundary.
    value_top    -    The value at the top of the boundary.
    value_bottom -    The value at the bottom of the boundary.
    """

    def __init__(self, nx=100, ny=100, value_left=1.,
                 value_right=0., value_top=0., value_bottom=0.,
                 q=None):
        """
        ::param nx:: Number of cells in the x direction.
        ::param ny:: Number of cells in the y direction.
        ::param value_left:: Boundary condition on the left face.
        ::param value_right:: Boundary condition on the right face.
        ::param value_top:: Boundary condition on the top face.
        ::param value_bottom:: Boundary condition on the bottom face.
        ::param q:: Source function. 
        """
        #set domain dimensions
        self.nx = nx
        self.ny = ny
        self.dx = 1. / nx
        self.dy = 1. / ny
        
        #define mesh
        self.mesh = fipy.Grid2D(nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy)

        #get the location of the middle of the domain
        #cellcenters=np.array(self.mesh.cellCenters).T
        #x=cellcenters[:, 0]
        #y=cellcenters[:, 1]
        x, y = self.mesh.cellCenters
        x_all=x[:self.nx]
        y_all=y[0:-1:self.ny]
        loc1=x_all[(self.nx-1)/2]
        loc2=y_all[(self.ny-1)/2]
        self.loc=np.intersect1d(np.where(x==loc1)[0], np.where(y==loc2)[0])[0]

        #get facecenters 
        X, Y = self.mesh.faceCenters

        #define cell and face variables 
        self.phi = fipy.CellVariable(name='$T(x)$', mesh=self.mesh, value=1.)
        self.C = fipy.CellVariable(name='$C(x)$', mesh=self.mesh, value=1.)
        self.source=fipy.CellVariable(name='$f(x)$', mesh=self.mesh, value=0.)
        
        #apply boundary conditions
        #dirichet
        self.phi.constrain(value_left, self.mesh.facesLeft)
        self.phi.constrain(value_right, self.mesh.facesRight)
        
        #homogeneous Neumann
        self.phi.faceGrad.constrain(value_top, self.mesh.facesTop)
        self.phi.faceGrad.constrain(value_bottom, self.mesh.facesBottom)
        
        #setup the diffusion problem
        self.eq = -fipy.DiffusionTerm(coeff=self.C) == self.source
    
    def set_source(self, source):
        """
        Initialize the source field.
        """
        self.source.setValue(source)
        
    def set_coeff(self, C):
        """
        Initialize the random conductivity field.
        """
        self.C.setValue(C)
        
    def solve(self):
        self.eq.solve(var=self.phi)

    def ObjectiveFunction(self):
        """
        We look at the temperature in the middle of the domain.
        """
        return self.phi.value[self.loc]

    def NeumannSpatialAverage(self):
        """
        Spatial average of the independent variable on the right side 
        Neumann boundary. 
        """
        loc = np.where(np.int32(self.mesh.facesRight.value) == 1)[0]
        val = self.phi.faceValue.value[loc]
        return np.mean(val)

    
    def RandomField(self):
        facecenters=np.array(self.mesh.faceCenters).T
        xf=facecenters[:, 0]
        yf=facecenters[:, 1]
        zf=self.C.value
        xif=yif=np.linspace(0.01, 0.99, 32)
        zif=griddata((xf, yf), zf, (xif[None,:], yif[:,None]), method='cubic')
        return zif
        