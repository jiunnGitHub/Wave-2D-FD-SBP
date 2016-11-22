def elastic_rate(hv, hs, v, s, rho, mu, nx, dx, order, r0, r1):
    # we compute rates that will be used for Runge-Kutta time-stepping
    #
    import first_derivative_sbp_operators
    import numpy as np
    import boundarycondition

    # initialize arrays for computing derivatives
    vx = np.zeros((nx, 1))
    sx = np.zeros((nx, 1))

    # compute first derivatives for velocity and stress fields
    first_derivative_sbp_operators.dx(vx, v, nx, dx, order)
    first_derivative_sbp_operators.dx(sx, s, nx, dx, order)

    # compute the elastic rates
    hv[:,:] = (1.0/rho)*sx
    hs[:,:] = mu*vx

    # impose boundary conditions using penalty: SAT
    impose_bc(hv, hs, v, s, rho, mu, nx, dx, order, r0, r1)

def impose_bc(hv, hs, v, s, rho, mu, nx, dx, order, r0, r1):
    # impose boundary conditions
    import numpy as np
    import boundarycondition

    # penalty weights
    if order==2:
        h11 = 0.5*dx
    if order== 4:
        h11 = (17.0/48.0)*dx

    if order==6:
        h11 = 13649.0/43200.0*dx
    
    mv = np.zeros((1,1))
    ms = np.zeros((1,1))

    pv = np.zeros((1,1))
    ps = np.zeros((1,1))

    v0 = v[0,:]
    s0 = s[0,:]

    vn = v[nx-1,:]
    sn = s[nx-1,:]

    # compute SAT terms
    boundarycondition.bcm(mv, ms, v0, s0, rho, mu, r0)
    boundarycondition.bcp(pv, ps, vn, sn, rho, mu, r1)

    # penalize boundaries with the SAT terms
    hv[0,:] =  hv[0,:] - 1.0/h11*mv
    hs[0,:] =  hs[0,:] - 1.0/h11*ms
    
    hv[nx-1,:] =  hv[nx-1,:] - 1.0/h11*pv
    hs[nx-1,:] =  hs[nx-1,:] + 1.0/h11*ps

    
    








    

    
    
