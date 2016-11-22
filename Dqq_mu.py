from __future__ import division
import numpy as np
from Dxx import D_xx

def Dqq_mu(u,G,h,order,options=None):
    # 2nd spatial derivative for variable material properties for the
    # coordinate along the columnes


    # process options
    # defaultOptions = struct('bcTypeQ','traction')
    # if (nargin<4), options = [] 
    # bcType = optimget(options,'bcTypeQ',defaultOptions,'fast')

    m, n = np.shape(u)
    u_qq = np.zeros((m, n))
    

    if order == 2:
      if 'periodic' in options["bcTypeQ"]:
    #     u_qq[:,0] = u[:,n-2] - 2*u[:,0] + u[:,1]
    #     u_qq[:,-1] =u[:,n-2] - 2*u[:,-1] + u[:,1]
        
        u_qq[:,0] = 0.5*np.multiply((G[:,0]+G[:,n-2]), u[:,n-2]) \
          - 0.5*np.multiply((G[:,1]+2*G[:,0]+G[:,n-2]), u[:,0]) \
          + 0.5*p.multiply((G[:,0]+nG[:,1]), u[:,1])
        
        u_qq[:,n-1] = 0.5*np.multiply((G[:,n-1]+G[:,n-2]), u[:,n-2]) \
          - 0.5*np.multiply((G[:,1]+2*G[:,n-1]+G[:,n-2]), u[:,n-1]) \
          + 0.5*np.multiply((G[:,n-1]+G[:,1]), u[:,1])
      
      for ind in range(1, n-1):
         #u_qq[:,ind] = u[:,ind+1] - 2*u[:,ind] + u[:,ind-1]
         
        u_qq[:,ind] = 0.5*np.multiply((G[:,ind]+G[:,ind-1]), u[:,ind-1]) \
          - 0.5*np.multiply((G[:,ind+1]+2*G[:,ind]+G[:,ind-1]), u[:,ind]) \
          + 0.5*np.multiply((G[:,ind]+G[:,ind+1]), u[:,ind+1])
      
      u_qq = (1.0/h**2)*u_qq
      
    else:
        u_qq = D_xx(u, G, m, n, h, order)
    
    return u_qq

    
