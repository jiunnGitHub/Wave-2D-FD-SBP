from __future__ import division
import numpy as np
from Dyy import D_yy

def Drr_mu(u,G,h,order,options=None):
    # 2nd spatial derivative for variable material properties
    # for the coordinate on rows

    # G = variable coefficient (such as shear modulus)

    # defaultOptions = struct('bcTypeR','traction')
    # if (nargin<4), options = [] end
    # bcType = optimget(options,'bcTypeR',defaultOptions,'fast')


    m, n = np.shape(u)
    u_rr = np.zeros((m, n))


    if order == 2:
      if 'periodic' in options["bcTypeR"]:
        # u_rr[0,:] = u[end-1, :] - 2*u[0,:] + u[1,:]
        # u_rr[-1,:] =u[-2,:] - 2*u[-1,:] + u[1,:]
        
        
        u_rr[0,:] = 0.5*np.multiply((G[0,:]+ G[m-2,:]) ,u[m-2,:]) \
          - 0.5*np.multiply((G[1,:]+2*G[0,:]+ G[m-2,:]) ,u[0,:]) \
          + 0.5* np.multiply((G[0,:]+G[1,:]) ,u[1,:])
        
        u_rr[m-1,:] = 0.5* np.multiply((G[m-1,:]+G[m-2,:]) ,u[m-2,:]) \
          - 0.5* np.multiply((G[1,:]+2*G[m-1,:]+G[m-2,:]) ,u[m-1,:]) \
          + 0.5*np.multiply((G[m-1,:]+G[1,:]) ,u[1,:])

      for ind in range(1, m-1):
        # u_yy(i,:) = u[i+1,:] - 2*u[i,:] + u[i-1,:]
        
        u_rr[ind,:] = 0.5*np.multiply((G[ind,:]+G[ind-1,:]) ,u[ind-1,:]) \
          - 0.5*np.multiply((G[ind+1,:]+2*G[ind,:]+G[ind-1,:]) ,u[ind,:]) \
          + 0.5*np.multiply((G[ind,:]+G[ind+1,:]) ,u[ind+1,:])

      u_rr = (1.0/h**2)*u_rr
      
    else:
      u_rr = D_yy(u, G, m, n, h, order)
      
    return u_rr
