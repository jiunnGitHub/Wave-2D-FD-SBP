def centered_fd_derivative(pxx, pzz, p, nx, nz, dx, dz, nop):
    if nop==3:
        # calculate partial derivatives, be careful around the boundaries
        for i in range(1, nx - 1):
            pzz[:, i] = (p[:, i + 1] - 2 * p[:, i] + p[:, i - 1]) / dx ** 2
        for j in range(1, nz - 1):
            pxx[j, :] = (p[j - 1, :] - 2 * p[j, :] + p[j + 1, :]) / dz ** 2

    if nop==5:
        #################################################
        # IMPLEMENT 5 POINT CENTERED DIFFERENCES HERE
        #################################################
        # calculate partial derivatives, be careful around the boundaries
        #for i in range(2, nx - 2):
        #    pzz[:, i] = (-1./12*p[:,i+2]+4./3*p[:,i+1]-5./2*p[:,i]+4./3*p[:,i-1]-1./12*p[:,i-2])/dz**2
        #for j in range(2, nz - 2):
        #    pxx[j, :] = (-1./12*p[j+2,:]+4./3*p[j+1,:]-5./2*p[j,:]+4./3*p[j-1,:]-1./12*p[j-2,:])/dx**2
        for i in range(2, nz - 2):
            pzz[:, i] = (-1.0/12.0*p[:, i-2] + 4.0/3.0*p[:, i-1] - 5.0/2.0*p[:, i] + 4.0/3.0*p[:, i + 1] - 1.0/12.0*p[:, i+2])/dx**2
        for j in range(2, nx - 2):
            pxx[j, :] = (-1.0/12.0*p[j-2, :] + 4.0/3.0*p[j-1, :] - 5.0/2.0*p[j, :] + 4.0/3.0*p[j + 1, :] - 1.0/12.0*p[j+2, :])/dz**2
