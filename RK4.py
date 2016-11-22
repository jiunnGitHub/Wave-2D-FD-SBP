def elastic_RK4(rv, rs, v, s, rho, mu, nx, dx, order, dt, r0, r1):

    # fourth order Runge-Kutta time-stepping
    import rate
    import numpy as np

    # intialize arrays for Runge-Kutta stages
    k1v = np.zeros((nx, 1))
    k1s = np.zeros((nx, 1))
    k2v = np.zeros((nx, 1))
    k2s = np.zeros((nx, 1))
    k3v = np.zeros((nx, 1))
    k3s = np.zeros((nx, 1))
    k4v = np.zeros((nx, 1))
    k4s = np.zeros((nx, 1))


    rate.elastic_rate(k1v, k1s, v, s, rho, mu, nx, dx, order, r0, r1)
    rate.elastic_rate(k2v, k2s, v+0.5*dt*k1v, s+0.5*dt*k1s, rho, mu, nx, dx, order, r0, r1)
    rate.elastic_rate(k3v, k3s, v+0.5*dt*k2v, s+0.5*dt*k2s, rho, mu, nx, dx, order, r0, r1)
    rate.elastic_rate(k4v, k4s, v+dt*k3v, s+dt*k3s, rho, mu, nx, dx, order, r0, r1)

    # update fields
    rv[:,:] = v + (dt/6.0)*(k1v + 2.0*k2v + 2.0*k3v + k4v)
    rs[:,:] = s + (dt/6.0)*(k1s + 2.0*k2s + 2.0*k3s + k4s)
