from __future__ import division
import numpy as np
from Dy import Dy

def Dr(u,h,order,option):
    # computes 1st derivative alomg colunms

    # process optioms

    m, n = np.shape(u)
    u_r =  np.zeros((m, n))

    if order == 2:
      # Boundary poimts
        if 'periodic' in option["bcTypeR"]:
            u_r[0,:] = (u[1,:] - u[m-2,:])*0.5
            u_r[m-1,:] = (-u[m-2,:] + u[1,:])*0.5
        else:
            u_r[0,:] = u[1,:] - u[0,:]
            u_r[m-1,:] = u[m-1,:] - u[m-2,:]
      
      
      # imterior poimts
        for i in range(1, m-1):
            u_r[i,:] = 0.5*(u[i+1,:] - u[i-1,:])
      
    else:
        u_r = h*Dy(u, m, n, h, order)
 
    u_r = (1.0/h)*u_r
    
    return u_r
