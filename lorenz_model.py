# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 22:12:31 2022

@author: isido
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# =============================================================================
# Sistema de lorenz (solucion iterativa)
# =============================================================================

def lorenz(rho, sigma, beta, dt, x0, y0, z0):
    """
    https://en.wikipedia.org/wiki/Lorenz_system
    """
    dx = (rho * (y0 - x0)) * dt
    dy = (x0 * (sigma - z0) - y0) * dt
    dz = (x0 * y0 - beta * z0) * dt
    
    x1 = x0 + dx
    y1 = y0 + dy
    z1 = z0 + dz
    
    return x1, y1, z1

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
dt = 0.01
final_time = 40

x0, y0, z0 = 1, 1, 1
out = [[x0, y0, z0]]
time = 0
while (time<final_time):
    x1, y1, z1 = lorenz(rho, sigma, beta, dt, x0, y0, z0)
    out.append([x1, y1, z1])    
    x0, y0, z0 = x1, y1, z1
    time = time + dt
    
out=np.array(out)

# =============================================================================
# Solucion mediante integracion scipy
# =============================================================================

def Lorenz(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, final_time, dt)
states = odeint(Lorenz, state0, t)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
p1 = ax.scatter3D(out[:,0], out[:,1], out[:,2], c="k", s=3, alpha=0.5)
p1 = ax.plot3D(out[:,0], out[:,1], out[:,2], 'gray')
ax = fig.add_subplot(1, 2, 2, projection='3d')
p2=ax.scatter3D(out[:,0], out[:,1], out[:,2], c="k", s=3, alpha=0.5)
p2=ax.plot3D(out[:,0], out[:,1], out[:,2], 'gray')


