from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from Dx import Dx
from Dy import Dy
from Dq import Dq
from Dr import Dr
from Dxx import D_xx
from Dyy import D_yy
from Dqq_mu import Dqq_mu
from Drr_mu import Drr_mu
from Gen_Surf_Mesh import Gen_Surf_Mesh
from Traction_2d import Traction_2d
from copy import deepcopy
import sys

plt.switch_backend("TkAgg")
plt.ion()
 


# Inhomeneous 2D wave equation with curvilinear coordinates q,r
#
# rho u_tt =  (mu u_x)_x +  (mu u_y)_y    on  x,y in domain
# density: rho, shear modulus: mu
#
# -> J rho u_tt =  (\hat{G} u_q)_q + (\hat{G} u_r)_r
#
# where J is the Jacobian
#
# Boundary Conditions:
#  x= +-Lx: Z_s*u_t +- mu*u_x = 0
#  y= Ly: Z_s*u_t + mu*u_y = 0
#  free-surface topography: =  mu*u_y = 0

def Du(u, A, B, C, CT, G, hq, hr, order, option):

    #print "Debug"
    #print Dqq_mu(u,A,hq,order,option) 
    #print Drr_mu(u,B,hr,order,option)
    #print Dr(np.multiply(C ,Dq(u,hq,order,option)),hr,order,option)
    #print Dq(np.multiply(CT,Dr(u,hr,order,option)),hq,order,option)

    #print "Debug\n\n"
    return np.divide((Dqq_mu(u,A,hq,order,option) \
                    + Drr_mu(u,B,hr,order,option)\
                    + Dr(np.multiply(C ,Dq(u,hq,order,option)),hr,order,option)\
                    + Dq(np.multiply(CT,Dr(u,hr,order,option)),hq,order,option)), J)



# spatial order of accuraccy: 2, 4, 6
order = 4 if (len(sys.argv) == 1) else int(sys.argv[-1]) 

# simulation domain
mx = 101
my = 51        # set the number of grid points

nt = 800      # number of time steps
isx = mx / 2  # source index x
isz = my / 2  # source index z
ist = 100     # shifting of source time function
f0 = 100.0    # dominant frequency of source (Hz)
isnap = 10    # snapshot frequency
T = 1.0 / f0  # dominant period


# Receiver locations
irx = np.array([int(np.floor(0.2*mx)), int(np.floor(0.4*mx)), int(np.floor(0.6*mx)), int(np.floor(0.8*mx))])
irz = np.array([int(np.floor(0.05*my)), int(np.floor(0.05*my)), int(np.floor(0.05*my)),int(np.floor(0.05*my))])
seis = np.zeros((len(irx), nt+1))


# initialize the bounds of our rectangular block
xMin = 0
xMax = 10     
yMin = 0
yMax = 4


hx = (xMax-xMin)/(mx-1)
hy = (yMax-yMin)/(my-1)


topo = 0.0

XB = np.linspace(xMin, xMax, mx)
XT = np.linspace(xMin, xMax, mx)
YL = np.linspace(yMin, yMax, my)
YR = np.linspace(yMin, yMax, my)


XL = np.ones(my)
XR = np.ones(my)
YB = np.ones(mx)
YT = np.ones(mx)

XL = XB[0]*XL 
XR = XB[mx-1]*XR

YB = YL[0]*YB + 0.0*topo*np.sin(2*np.pi*XB)
YT = YL[my-1]*YT + topo*np.sin(np.pi*XT) # perturb the topography with a np.sin funtion
mid = np.ceil(mx/2)

#peak =  YT(my)-0.1
#YT(mid-10: mid) = linspace(YT(mid-10), peak, 11)
#YT(mid: mid+10) = linspace(peak, YT(mid+10), 11)
#---------------------------------------------------

# code for creation of mesh (X,Y) 
X, Y = Gen_Surf_Mesh(mx,my,XL,XR,XB,XT,YL,YR,YB,YT)


# boundary conditions
# bcTypeR = 'periodic'
bcTypeR = 'traction'
bcTypeQ = 'traction'
option = {
    'bcTypeR' : bcTypeR,
    'bcTypeQ' : bcTypeQ
}

# dicretize the tranformed domain: the unite square (q,r) = [0,1]**2
hq = 1.0/(mx-1)
hr = 1.0/(my-1)


q = np.linspace(0, 1, mx)
r = np.linspace(0, 1, my)

print(q[mx-1], r[my-1])

Q, R = np.meshgrid(q, r)

# compute metric derivatives and Jacobian
y_r = Dr(Y,hr,order,option)
y_q = Dq(Y,hq,order,option)
x_r = Dr(X,hr,order,option)
x_q = Dq(X,hq,order,option)

J = np.multiply(x_q, y_r) - np.multiply(y_q, x_r)                         # Jacobian (determinant of metric)
#ad = J[:,1]
#print(ad)



#exit(-1)

q_x =   np.divide(y_r, J)
r_x = - np.divide(y_q, J)
q_y = - np.divide(x_r, J)
r_y =   np.divide(x_q, J)

# Time-stepping parameters
tmax = 20
cmax = 3.464
cfl = 0.2
dt = cfl/cmax*hx


t, dummy = np.linspace(0, tmax, retstep=dt)


# material properties (can be heterogenous)
rho = 2.5001 + 0.0*X
#rho(:, ceil(mx/2)-15:ceil(mx/2)+15) = 2.5001
G = rho*cmax**2 + 0.0*X

# SBP penalty weights
if (order==2):
    hq11 = 0.5*hq
    hr11 = 0.5*hr

elif (order == 4):
    hq11 = (17/48)*hq 
    hr11 = (17/48)*hr

elif(order == 6):
    hq11 = 13649/43200*hq
    hr11 = 13649/43200*hr


# penalty coefficients
penq = np.sqrt(np.square(q_x) + np.square(q_y))/hq11
penr = np.sqrt(np.square(r_x) + np.square(r_y))/hr11


# 2nd spatial deriv
A  = np.multiply(np.multiply(J,(np.square(q_x) + np.square(q_y))), G)
B  = np.multiply(np.multiply(J,(np.square(r_x) + np.square(r_y))), G)
C  = np.multiply(np.multiply(J,(np.multiply(r_x,q_x) + np.multiply(r_y,q_y))), G)
CT = np.multiply(np.multiply(J,(np.multiply(r_x,q_x) + np.multiply(r_y,q_y))), G)  

#  wave speed
cs = np.sqrt(np.divide(G,rho))

# initial soultion
amp = 20.0
cx = 0.5*(xMin + xMax)
cy = 0.5*(yMin + yMax) # location of center of Gaussian
delta = 0.01

u0 = amp* np.multiply(np.exp(-np.square((X-cx))/delta), np.exp(-np.square((Y-cy))/delta))
u = deepcopy(u0)
u0_t = 0* np.multiply(np.exp(-np.square((X-cx))/delta),np.exp(-np.square((Y-cy))/delta))



# compute the spatial approximations in the transformed domain using SBP
# operators

Dmu = Du(u, A, B, C, CT, G, hq, hr, order, option) # without SAT

# compute the tranctions on the boundaries
Tq, Tr = Traction_2d(u,G,q_x,q_y,r_x,r_y, order,hq, hr,option)

# penalize the tranctions on the boundaries                     
Dmu[:,0] = Dmu[:,0] + np.multiply(penq[:,0],Tq[:,0])
Dmu[:,mx-1] = Dmu[:,mx-1] - np.multiply(penq[:,mx-1],Tq[:,mx-1])
Dmu[0,:] = Dmu[0,:] + np.multiply(penr[0,:],Tr[0,:])
Dmu[my-1,:] = Dmu[my-1,:] - np.multiply(penr[my-1,:],Tr[my-1,:])


# initial time-step compute the solutions
nn = 0
uNew = u + dt*u0_t + np.multiply(0.5*np.divide(dt**2,rho),Dmu)

# update the solutions
uPrev = deepcopy(u)
u = deepcopy(uNew)

print("Okay !")

v = 2.0
image = plt.imshow(u, interpolation='nearest', animated=True,
                   vmin=-v, vmax=+v, cmap=plt.cm.RdBu)

# Plot the receivers
for x, z in zip(irx, irz):
    plt.text(x, z, '+')

plt.text(isx, isz, 'o')
plt.title("Order {:d} Operator".format(order))
plt.colorbar()
plt.xlabel('ix')
plt.ylabel('iz')
plt.gca().invert_yaxis()

#exit(-1)

ir = np.arange(len(irx)-1)

# loop until the final time
for tInd in range(nt) :
    #np.linspace(2, np.ceil(len(t)/20-1)):
    nn += 1
    if nn == 4000:
        #print "\n"*5 + "End of program" + "\n"*5
        for i in range(my):
            print(i, Tq[i,0], Tq[i, mx-1], Tr[0, i], Tr[my-1, i])
                
        break

    Dmu = Du(u, A, B, C, CT, G, hq, hr, order, option) # without SATo

    print(nn)

    # compute the tranctions on the boundaries
    Tq, Tr = Traction_2d(u,G,q_x,q_y,r_x,r_y, order,hq, hr,option)

    # penalize the tranctions on the boundaries                     
    Dmu[:,0] = Dmu[:,0] + np.multiply(penq[:,0],Tq[:,0])
    Dmu[:,mx-1] = Dmu[:,mx-1] - np.multiply(penq[:,mx-1],Tq[:,mx-1])
    Dmu[0,:] = Dmu[0,:] + np.multiply(penr[0,:],Tr[0,:])
    Dmu[my-1,:] = Dmu[my-1,:] - np.multiply(penr[my-1,:],Tr[my-1,:])


    # interior
    uNew[:,:] = np.divide(Dmu[:,:]*dt**2,rho[:,:]) + 2.0*u[:,:] - uPrev[:,:]

    #uNew[0,:] = 0.0
    #uNew[my-1,:] = 0.0
    #uNew[:,0] = 0.0
    #uNew[:,mx-1] = 0.0
    
  
    # Boundaries
    #  the left  and right boundaries 
    uNew[:,0] = (dt**2)*np.divide(Dmu[:,0],rho[:,0]) + 2.0*u[:,0] \
                + np.multiply((0.5*dt*np.multiply(cs[:,0],penq[:,0])-1),uPrev[:,0])
    uNew[:,0] = np.divide(uNew[:,0],(1.0+0.5*dt*np.multiply(cs[:,0],penq[:,0])))
  
    uNew[:,-1] = (dt**2)*np.divide(Dmu[:,-1],rho[:,-1]) + 2.0*u[:,-1] \
                   + np.multiply((-1.0 + 0.5*dt*np.multiply(cs[:,-1],penq[:,-1])),uPrev[:,-1])
    uNew[:,-1] =  np.divide(uNew[:,-1],(1.0+0.5*dt*np.multiply(cs[:,-1],penq[:,-1])))
  
  
    # bottom boundary
    #if "traction" in bcTypeR.lower():
    uNew[0,:] = dt**2*np.divide(Dmu[0,:],rho[0,:]) + 2.0*u[0,:] \
                + np.multiply((0.5*dt*np.multiply(cs[0,:],penr[0,:])-1.0),uPrev[0,:])
    uNew[0,:] = np.divide(uNew[0,:],(1.0+0.5*dt*np.multiply(cs[0,:],penr[0,:])))
    #uNew[-1,:] = dt**2*Dmu[-1,:],rho[-1,:]  + 2*u[-1,:] + np.multiply((cs[-1,:],0.5*dt,penr[-1,:]-1),uPrev[-1,:])
    #uNew[-1,:] = uNew[-1,:], np.multiply((1+cs[-1,:],0.5*dt,penr[-1, :])
    
    #plt.title("time: %.2f" % t)
    if nn % isnap == 0:                    # you can change the speed of the plot by increasing the plotting interval
        #plt.title("time: %.2f" % t)
        image.set_data(u)
        plt.gcf().canvas.draw()
        plt.show()
        print(nn)
    
  
    # update which is the n+1, n, and n-1 steps
    uPrev[:,:] = u[:,:]
    u[:,:] = uNew[:,:]
    #uPrev = deepcopy(u)
    #u = deepcopy(uNew)

    # Save seismograms
    seis[ir, nn] = u[irz[ir], irx[ir]]


plt.ioff()
plt.figure(figsize=(12, 12))

plt.subplot(221)
time = np.arange(nt) * dt
plt.plot(time, src)
plt.title('Source time function')
plt.xlabel('Time (s) ')
plt.ylabel('Source amplitude ')

plt.subplot(222)
ymax = seis.ravel().max()  
for ir in range(len(seis)):
    plt.plot(time, seis[ir, :] + ymax * ir)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.subplot(223)
ymax = seis.ravel().max()
for ir in range(len(seis)):
    plt.plot(time, seis[ir, :] + ymax * ir)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.subplot(224)
# The velocity model is influenced by the Earth model above
plt.title('Velocity Model')
v1 = c.max()
v0 = 0.0 #*c.min()
#plt.imshow(c)
plt.imshow(c, vmin=v0, vmax=v1, cmap=plt.cm.RdBu)
plt.xlabel('ix')
plt.ylabel('iz')
plt.gcf().canvas.draw()
plt.colorbar()

plt.show()
