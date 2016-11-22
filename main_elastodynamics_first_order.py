# This is a configuration step for the exercise. 
#matplotlib notebook                                                                                                                                   

import numpy as np
import matplotlib.pyplot as plt
import RK4

import RK4

plt.ion()


# ---------------------------------------------------------                                                                                            
# Simple finite difference solver
#
# elastic wave equation  rho*v_t = sigma_x +  src
#                        1/mu*sigma_t = v_x
# 1-D regular grid                         
                                                                                                              
# ---------------------------------------------------------
L = 10.0      # length of the domain [km]
tend = 8.0       # final time
nx = 200      # grid points in x                                                                                                                       
dx = L/nx     # grid increment in x
dt = 0.5*dx    # Time step
c0 = 3.464      # velocity (km/s) (can be an array)                                                                                                             
isx = nx / 2  # source index x                                                                                                                          
ist = 100     # shifting of source time function                                                                                                       
f0 = 100.0    # dominant frequency of source (Hz)
isnap = 20    # snapshot frequency
T = 1.0 / f0  # dominant period
order = 2          # order of accuracy

rho = 2.6702       # density [g/cm^3]
mu = rho*c0**2     # shear modulus [GPa]

dt = 0.5/c0*dx     # Time step
nt = int(round(tend/dt))      # number of time steps


# reflection coefficients: -1<= r <= 1
# (clamped: r = -1; free-surface: r = 1, non-reflecting: r = 0)
r0 = 1.0          # left boundary
r1 = -1.0         # right boundary

# Initialize: particle velocity (v); and shear stress (s)
v = np.zeros((nx, 1))
s = np.zeros((nx, 1))

#Initialize the domain
x = np.zeros((nx, 1))

# Initial particle velocity perturbation and discretize the domain
sigma = 0.3
for j in range(0, nx):
    v[j, :] = np.exp(-np.log(2)*((j-isx)*dx)**2/(2*sigma**2))  # particle velocity
    x[j, :] = j*dx                                     # discrete domain

# Inialize the plot
image, = plt.plot(x, v, animated=True)

# Evolve the solution for nt time-steps using 4th order Runge-Kutta method
for it in range(nt):
    
    RK4.elastic_RK4(v, s, v, s, rho, mu, nx, dx, order, dt, r0, r1)
    
    t = it*dt
    # print('time = ', t)
    if it % isnap == 0: 
       
        plt.title("time = %.2fs" % t)
        #plt.plot(x, v)
        image.set_data(x,v)
        #plt.gcf().canvas.draw()
        #plt.colorbar()
        plt.ylabel('v [m/s]')
        plt.xlabel('x [km]')
        plt.show() 

plt.ioff()
plt.show()

