def elastic_rate2d(D, F, Mat, nf, nx, ny, dx, dy, order, r):
    # we compute rates that will be used for Runge-Kutta time-stepping
    #
    import first_derivative_sbp_operators
    import numpy as np
    import boundarycondition

    # initialize arrays for computing derivatives
    dxF = np.zeros((nf))
    dyF = np.zeros((nf))

    # compute the elastic rates
    for i in range(0, nx):
        for j in range(0, ny):
        
            first_derivative_sbp_operators.dx2d(dxF, F, nx, i, j, dx, order)
            first_derivative_sbp_operators.dy2d(dyF, F, ny, i, j, dy, order)
            
            D[i,j,0] = 1.0/Mat[i,j,0]*(dxF[2] + dyF[4])
            D[i,j,1] = 1.0/Mat[i,j,0]*(dxF[4] + dyF[3])
            D[i,j,2] = (2.0*Mat[i,j,2] + Mat[i,j,1])*dxF[0] + Mat[i,j,1]*dyF[1]
            D[i,j,3] = (2.0*Mat[i,j,2] + Mat[i,j,1])*dyF[1] + Mat[i,j,1]*dxF[0]
            D[i,j,4] = Mat[i,j,2]*(dyF[0] + dxF[1])
    

    # impose boundary conditions using penalty: SAT
    impose_bc(D, F, Mat, nx, ny, nf, dx, dy, order, r)

def impose_bc(D, F, Mat, nx, ny, nf, dx, dy, order, r):
    # impose boundary conditions
    import numpy as np
    import boundarycondition

    # penalty weights
    if order==2:
        hx = 0.5*dx
        hy = 0.5*dy
        
    if order== 4:
        hx = (17.0/48.0)*dx
        hy = (17.0/48.0)*dy

    if order==6:
        hx = 13649.0/43200.0*dx
        hy = 13649.0/43200.0*dy
    
    BF0x = np.zeros((ny, nf))
    BFnx = np.zeros((ny, nf))

    BF0y = np.zeros((nx, nf))
    BFny = np.zeros((nx, nf))
      
    # compute SAT terms
    boundarycondition.bcm2dx(BF0x, F, Mat, nx, ny, r)
    boundarycondition.bcp2dx(BFnx, F, Mat, nx, ny, r)
    
    boundarycondition.bcm2dy(BF0y, F, Mat, nx, ny, r)
    boundarycondition.bcp2dy(BFny, F, Mat, nx, ny, r)

    # penalize boundaries with the SAT terms

    D[0,:,:] =  D[0,:,:] -  1.0/hx*BF0x[:,:]
    D[nx-1,:,:] =  D[nx-1,:,:] -  1.0/hx*BFnx[:,:]
    

    D[:,0,:] =  D[:,0,:] -  1.0/hy*BF0y[:,:]
    D[:,ny-1,:] =  D[:,ny-1,:] -  1.0/hy*BFny[:,:]
    
    

    
    








    

    
    
