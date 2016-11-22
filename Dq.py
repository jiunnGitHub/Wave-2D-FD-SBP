from __future__ import division
import numpy as np
from Dx import Dx

def Dq(u,h,order,option):
    # computes 1st derivative along columns

    # process options

    m, n = np.shape(u)
    u_q = np.zeros((m, n))

    if order == 2:
      # Boundary points
        if 'periodic' in option["bcTypeQ"]:
            u_q[:,0] = (u[:,1] - u[:,n-2])*0.5
            u_q[:,n-1] = (-u[:,n-2] + u[:,1])*0.5
        else:
            u_q[:,0] = u[:,1] - u[:,0]
            u_q[:,n-1] = u[:,n-1] - u[:,n-2]
      
      
      # interior points
        for i in range(1, n-1):
            u_q[:,i] = 0.5*(u[:,i+1] - u[:,i-1])
      
    else:
        u_q = h*Dx(u, m, n, h, order)
 
    u_q = (1.0/h)*u_q
    
    return u_q
