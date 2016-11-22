from __future__ import division
import numpy as np
from Dq import Dq
from Dr import Dr

def Traction_2d(u,G,q_x,q_y,r_x,r_y,order,hq, hr, option):
    # Compute tractions in curvilinear coordinates
    # order = order of accuracy
    # h = grid spacing
    # G = shear modulus
    # u = displacement
    # q_x = dq/dx etc
    #
    # The (normalized) normals are:
    # nq = 1,sqrt(q_x**2 + q_y**2) , [q_x q_y]
    # nr = 1,sqrt(r_x**2 + r_y**2) , [r_x r_y]

    # Thus the tractions are
    # T = G * (ux uy) dot n
    # -> Tq = np.multiply(G,sqrt(q_x**2 + q_y**2) , (np.multiply(u_x,q_x) + np.multiply(u_y,q_y)))
    
    u_q = Dq(u,hq,order,option)
    u_r = Dr(u,hr,order,option)

    u_x = np.multiply(q_x , u_q) + np.multiply(r_x , u_r)
    u_y = np.multiply(q_y , u_q) + np.multiply(r_y , u_r)

    Tq = np.multiply((np.multiply(u_x,q_x) + np.multiply(u_y,q_y)) , np.divide(G,np.sqrt(np.square(q_x) + np.square(q_y))))
    Tr = np.multiply((np.multiply(u_x,r_x) + np.multiply(u_y,r_y)) , np.divide(G,np.sqrt(np.square(r_x) + np.square(r_y))))


    return Tq,Tr
