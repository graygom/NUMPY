#
# TITLE:
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: engineering mathematics (UW ME564), 2014, Steve Brunton
#            engineering mathematics (UW ME565), 2015, Steve Brunton
#


import time
import numpy as np
import scipy as sc
import sympy as sy
import matplotlib.pyplot as plt
import PIL


#
# ME564 Lec - 01
#

if False:
    # matrix > rainy, nice, cloudy probability
    A = [ [0.50, 0.50, 0.25],
          [0.25, 0.00, 0.25],
          [0.25, 0.50, 0.50] ]
    A = np.array(A, dtype=float)
    # vector > forecast
    x_forecast = []
    # vector > today
    x_forecast.append( np.array( [ [1.00],
                                   [0.00],
                                   [0.00] ], dtype=float ) )
    # vector > forecast
    for index in range(10):
        x_forecast.append( A @ x_forecast[index] )
    # vector > forecast
    x_forecast = np.array(x_forecast)
    # visualization
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(x_forecast[:,0], 'r.-', label='rainy')
    ax.plot(x_forecast[:,1], 'b.-', label='nice')
    ax.plot(x_forecast[:,2], 'g.-', label='cloudy')
    ax.grid(ls=':')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


#
# ME564 Lec - 02
#

if False:
    # euler number
    print(np.exp(0.05))


#
# ME564 Lec - 03
#

if False:
    # Taylor series, expansion
    x = np.linspace(-10.0, 10.0, 1001)
    y = np.sin(x)

    # Taylor series
    P1 = [+1.0, 0.0]
    yT1 = np.polyval(P1, x)
    P3 = [-1.0/np.prod(range(1,4)), 0.0] + P1
    yT3 = np.polyval(P3, x)
    P5 = [+1.0/np.prod(range(1,6)), 0.0] + P3
    yT5 = np.polyval(P5, x)
    P7 = [-1.0/np.prod(range(1,8)), 0.0] + P5
    yT7 = np.polyval(P7, x)
    
    # visualization
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.plot(x, y, 'k', linewidth=2.0, label='sin(x)')
    ax.plot(x, yT1, 'r:', linewidth=1.0, label='Taylor 1st')
    ax.plot(x, yT3, 'b:', linewidth=1.0, label='Taylor 3rd')
    ax.plot(x, yT5, 'g-', linewidth=1.0, label='Taylor 5th')
    ax.plot(x, yT7, 'b-', linewidth=1.0, label='Taylor 7th')
    ax.grid(ls=':')
    ax.set_ylim(-4.0, 4.0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


#
# ME564 Lec - 04
#

if False:
    # damped harmonic oscillator
    w = 2.0 * np.pi
    d = 0.25
    #
    A = [ [0.0, 1.0],
          [-w**2, -d] ]
    A = np.array(A)
    # timeline
    t = np.linspace(0.0, 10.0, 1001)
    # solutions
    x = np.zeros([2, 1001], dtype=float)
    # numerical integration
    for index in range(1001):
        if index == 0:
            x[:,index] = np.array( [0.1, 0.0] )
        else:
            dt = t[index] - t[index-1]
            x[:,index] = x[:,index-1] + A @ x[:,index-1] * dt
    # visualization
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x[0,:])
    ax.plot(x[1,:])
    ax.grid(ls=':')
    plt.tight_layout()
    plt.show()
    plt.close()


#
# ME564 Lec - 05
#

if False:
    # suspend variables
    A = [ [  0.0,  1.0,  0.0,  0.0 ],
          [  0.0,  0.0,  1.0,  0.0 ],
          [  0.0,  0.0,  0.0,  1.0 ],
          [ -7.0, -1.0, -2.0, -5.0] ]
    A = np.array(A, dtype=float)
    # solutions of characteristic equation
    sols = np.linalg.eig(A)
    print(sols)


#
# ME564 Lec - 06
#

if False:
    # eigenvalue equations
    A = [ [ 0.0,  1.0],
          [-2.0, -3.0] ]
    A = np.array(A)
    # Diagonal Mat., Eigenvectors Mat.
    D, T = np.linalg.eig(A)
    print(D)
    print(T)
    # eigenvalue * eigenvector
    print(A @ T[:,0])
    print(A @ T[:,1])


#
# ME564 Lec - 07, 08
#

if False:
    # suspend variables
    A = [ [ 3.0, -1.0],
          [-1.0,  3.0] ]
    A = np.array(A, dtype=float)
    # solve eigenvalues, eigenvectors 
    D_vec, T_mat = np.linalg.eig( A )
    # finding inverse
    T_inv_mat = np.linalg.inv( T_mat )
    T_inv_mat = T_mat.T.conj()
    # vector operation
    E_vec = np.exp( D_vec )
    # convert vector to matrix
    D_mat = np.diag( E_vec )
    print(D_mat)


#
# ME564 Lec - 09, 10, 11
#

if False:
    # system (degenerate system)
    A = [ [-0.009,  1.0 ],
          [   0.0, -0.01] ]
    A = np.array(A, dtype=float)
    # time
    t = np.linspace(0.0, 1000.0, 10001)
    # solutions
    sol = np.zeros((2,10001), dtype=float)
    # ode45
    for index in range(10001):
        if index == 0:
            sol[:,index] = [0.0, 1.0]
        else:
            dt = t[index] - t[index-1]
            sol[:,index] = sol[:,index-1] + A @ sol[:,index-1] * dt
    # visualization
    fig, ax = plt.subplots(2,1,figsize=(6,4))
    ax[0].plot(t, sol[0,:], label='pos.')
    ax[1].plot(t, sol[1,:], label='vel.')
    ax[0].grid(ls=':')
    ax[1].grid(ls=':')
    ax[0].legend(fontsize=9)
    ax[1].legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    
    # solve eigenvalues, eigenvectors
    D, T = np.linalg.eig(A)
    print(D)
    print(T)
    xi1 = T[:,0]
    xi2 = T[:,1]
    print(xi1)
    print(xi2)

    # system 
    A = [ [1.0, 1.0],
          [0.0, 1.0] ]
    A = np.array(A, dtype=float)
    D, T = np.linalg.eig(A)
    print(D)
    print(T)
    xi1 = T[:,0]
    xi2 = T[:,1]
    print(xi1)
    print(xi2)


#
# ME564 Lec - 12, 13
#

if False:
    # dynamic equations
    A = np.array( [ [  0.0,  1.0 ],
                    [ -1.0, -0.1 ] ], dtype=float)
    B = np.array( [ [ 0.0 ],
                    [ 1.0 ] ], dtype=float)
    # measurements
    C = np.eye(2, dtype=float)
    D = np.array( [ [ 0.0 ],
                    [ 0.0 ] ], dtype=float)
    # making system
    system = sc.signal.StateSpace(A, B, C, D)

    # impulse
    t, y = sc.signal.impulse(system)
    plt.plot(t, y, 'o:')
    plt.grid(ls=':')
    plt.show()
    plt.close()
    
    # triangular control input
    t = np.linspace(0.0, 50.0, 5001, dtype=float)
    u = np.zeros(5001)
    u[1001:2001] = t[:1000] / 1e5
    u[2001:3001] = u[2000] - t[:1000] / 1e5

    # simulate continuous time domain system
    t_out, y_out, x_out = sc.signal.lsim(system, u, t)

    # visualization
    plt.plot(t, u)
    plt.plot(t_out, y_out)
    plt.show()


#
# ME564 Lec - 14, 15
#

if False:
    # numerical differentiation 1
    dt = 0.4
    t = np.arange(-2.0, 4.0+dt, dt)
    f = np.sin(t)
    dfdt = np.cos(t)
    # forward difference
    dfdt_FD = ( np.sin(t+dt) - np.sin(t) )/ dt
    # backward difference
    dfdt_BD = ( np.sin(t) - np.sin(t-dt) )/ dt
    # central difference
    dfdt_CD = ( np.sin(t+dt) - np.sin(t-dt) )/ (2*dt)
    # visualization
    plt.plot(t, f, 'k--', label='function', linewidth=1.2)
    plt.plot(t, dfdt, 'k', label='exact derivative', linewidth=3.0)
    plt.plot(t, dfdt_FD, 'b', label='forward diff', linewidth=1.2)
    plt.plot(t, dfdt_BD, 'g', label='backward diff', linewidth=1.2)
    plt.plot(t, dfdt_CD, 'r', label='central diff', linewidth=1.2)
    plt.grid(ls=':')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.close()

    # numerical differentiation 2
    dx = 0.1
    x = np.arange(0.1, 3.0+dx, dx)
    f = np.sin(x)
    dfdx = np.cos(x)    # analytic derivative
    #
    dfdx_diff = np.zeros(len(x), dtype=float)
    dfdx_diff[0] = (f[1] - f[0]) / (x[1] - x[0])                # forward difference
    dfdx_diff[1:-1] = (f[2::] - f[:-2:]) / (x[2::] - x[:-2:])   # central difference
    dfdx_diff[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])           # backward difference
    #
    rng = np.random.default_rng(seed=0)                             # random number generator
    f_noise = f + rng.normal(loc=0.0, scale=0.01, size=x.size)      # normal distribution
    #
    dfdx_diff_noise = np.zeros(len(x), dtype=float)
    dfdx_diff_noise[0] = (f_noise[1] - f_noise[0]) / (x[1] - x[0])                # forward difference
    dfdx_diff_noise[1:-1] = (f_noise[2::] - f_noise[:-2:]) / (x[2::] - x[:-2:])   # central difference
    dfdx_diff_noise[-1] = (f_noise[-1] - f_noise[-2]) / (x[-1] - x[-2])           # backward difference

    # visualization
    fig, ax = plt.subplots(1,2,figsize=(9,4))
    ax[0].plot(x, f, 'k', linewidth=2, label='func.')
    ax[0].plot(x, f_noise, 'ro:', linewidth=1, label='func. w/ noise')
    ax[0].grid(ls=':')
    ax[0].legend(fontsize=9)
    ax[1].plot(x, dfdx, 'k', label='analytic derivative', linewidth=2)
    ax[1].plot(x, dfdx_diff, 'g:', label='numerical diff.', linewidth=1.0)
    ax[1].plot(x, dfdx_diff_noise, 'ro:', label='numerical diff. w/ noise', linewidth=1.0)
    ax[1].grid(ls=':')
    ax[1].legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.close()


#
# ME564 Lec - 16
#

if False:
    # numerical integration 1
    a, b, dx = 0.0, 10.0, 0.2
    x = np.arange(a, b+dx, dx, dtype=float)     # vector
    f = np.sin(x)                               # vector
    # left rectangle
    f_left = f[:-1]
    f_left_int = np.cumsum( f_left * (x[1:]-x[:-1]) )
    # right rectangle
    f_right = f[1:]
    f_right_int = np.cumsum( f_right * (x[1:]-x[:-1]) )
    # trapezoidal integration
    f_trapezoidal = f[:-1] + (f[1:]-f[:-1])/2.0
    f_trapezoidal_int = np.cumsum( f_trapezoidal * (x[1:]-x[:-1]) )
    # Simpson's rule
    f_simpson = (f[:-2:2] + 4.0*f[1:-1:2] + f[2::2])/3.0
    f_simpson_int = np.cumsum( f_simpson * (x[2::2]-x[:-2:2])/2.0 )
    # visualization
    fig, ax = plt.subplots(1, 2, figsize=(9,4))
    ax[0].plot(x, f, 'k', label='func', linewidth=2.0)
    ax[0].step(x, f, 'ro:', label='post', where='post')
    ax[0].grid(ls=':')
    ax[0].legend(fontsize=9)
    ax[1].plot(x[:-1], f_left_int, 'r', label='left rect', linewidth=1.0)
    ax[1].plot(x[:-1], f_right_int, 'g', label='right rect', linewidth=1.0)
    ax[1].plot(x[:-1], f_trapezoidal_int, 'b', label='trapezoidal', linewidth=1.0)
    ax[1].plot(x[1::2], f_simpson_int, 'k', label='Simpson', linewidth=1.0)
    ax[1].grid(ls=':')
    ax[1].legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.close()


#
# ME564 Lec - 17
#

if False:
    # damping oscillator conditions
    w = 2.0*np.pi
    d = 0.5
    # system
    A = np.array( [ [   0.0,      1.0 ],
                    [ -w**2, -2.0*d*w ] ], dtype=float )
    # check eignvalues
    eigen_val, eigen_vect = np.linalg.eig( A )
    print(eigen_val)
    # timeline
    dt = 0.01
    t = np.arange(0.0, 10.0+dt, dt, dtype=float)
    # solution
    sol_f = np.zeros([2, t.size], dtype=float)
    sol_b = np.zeros([2, t.size], dtype=float)
    # forward/backward Euler scheme
    for index in range(t.size):
        if index == 0:
            # initial condition
            sol_f[:,0] = [2.0, 0.0]
            sol_b[:,0] = [2.0, 0.0]
        else:
            # time evolution
            sol_f[:,index] = (np.eye(2) + dt * A) @ sol_f[:,index-1]
            sol_b[:,index] = np.linalg.inv( (np.eye(2) - dt * A) ) @ sol_b[:,index-1]
    # visualization
    fig, ax = plt.subplots(1, 3, figsize=(13,4))
    ax[0].plot(t, sol_f[0,:], label='x(t) fw')
    ax[0].plot(t, sol_b[0,:], label='x(t) bw')
    ax[1].plot(t, sol_f[1,:], label='v(t) fw')
    ax[1].plot(t, sol_b[1,:], label='v(t) bw')
    ax[2].plot(sol_f[0,:], sol_f[1,:], label='phase fw')
    ax[2].plot(sol_b[0,:], sol_b[1,:], label='phase bw')
    ax[0].grid(ls=':')
    ax[1].grid(ls=':')
    ax[2].grid(ls=':')
    ax[0].legend(fontsize=9)
    ax[1].legend(fontsize=9)
    ax[2].legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.close()


#
# ME564 Lec - 18, 19, 20
#

if False:
    # solving Lorentz equation w/ Runge-Kutta 4th order (aka ode45 in matlab)
    # Lorentz equation
    def lorentz_eq(x, sigma, beta, rho):
        dx_dt = np.zeros(3, dtype=float)
        dx_dt[0] = sigma * (x[1] - x[0])            # Vx
        dx_dt[1] = x[0] * ( rho - x[2] ) - x[1]     # Vy
        dx_dt[2] = x[0] * x[1] - beta * x[2]        # Vz
        return dx_dt
    # model parameters
    sigma, beta, rho = 10.0, 8.0/3.0, 28.0
    # timeline
    trange = np.arange(0.0, 10.0, 0.01)
    # solution
    sol = np.zeros([trange.size,3], dtype=float)
    # time evolution
    for index in range(trange.size):
        #
        if index == 0:
            # initial conditions
            sol[index,:] = np.array([-8.0, 8.0, 27.0])
        else:
            # dt
            dt = trange[index] - trange[index-1]
            # Runge-Kutta 4th order 1
            f1 = lorentz_eq(sol[index-1,:], sigma, beta, rho)
            f2 = lorentz_eq(sol[index-1,:]+dt/2*f1, sigma, beta, rho)
            f3 = lorentz_eq(sol[index-1,:]+dt/2*f2, sigma, beta, rho)
            f4 = lorentz_eq(sol[index-1,:]+dt*f3, sigma, beta, rho)
            # Runge-Kutta 4th order 2
            sol[index,:] = sol[index-1,:] + dt / 6.0 * (f1 + 2.0*f2 + 2.0*f3 + f4)
    # visualization
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(trange, sol[:,0], 'r.:', label='X coordinate')
    ax1.plot(trange, sol[:,1], 'g.:', label='Y coordinate')
    ax1.plot(trange, sol[:,2], 'b.:', label='Z coordinate')
    ax1.set_xlabel('time')
    ax1.set_ylabel('coordinate')
    ax1.set_title('Lorentz eq. w/ RK4 integ.')
    ax1.grid(ls=':')
    ax1.legend(fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(sol[:,0], sol[:,1], sol[:,2], label='init pos = (-8,8,27)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Lorentz eq. w/ RK4 integ.')
    ax2.grid(ls=':')
    ax1.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('lorentz_eq_using_RK4.png')
    plt.show()
    plt.close()
    

#
# ME564 Lec - 25
#

if False:
    # hypocycloid: x**(2/3) + y**(2/3) = a**(2/3)
    a = 1.0
    theta = np.linspace(0.0, 2.0*np.pi, 361)
    x = a * np.cos(theta)**3
    y = a * np.sin(theta)**3
    # visualization
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    ax.plot(x, y, 'b.:', linewidth=2.0, label='$x^{2/3}$+$y^{2/3}$=$a^{2/3}$')
    ax.grid(':')
    ax.legend(fontsize=9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('hypocycloid')
    plt.tight_layout()
    plt.savefig('hypocycloid.png')
    plt.show()
    plt.close()


#
# ME564 Lec - 28
#

if False:
    # double gyre flow
    # model parameters
    A, eps, om = 0.1, 0.25, 2.0*np.pi/10.0
    # spatial
    xrange = np.arange(0.0, 2.0, 0.025)
    yrange = np.arange(0.0, 1.0, 0.025)
    X, Y = np.meshgrid(xrange, yrange)    
    # temporal
    trange = np.arange(0.0, 15.0, 0.005)
    # solution
    sol = np.zeros([trange.size, 2, xrange.size, yrange.size], dtype=float)
    sol_vel = np.zeros([trange.size, 2, xrange.size, yrange.size], dtype=float)
    # time evolution
    for index in range(trange.size):
        # modeling parameters at t
        a = eps * np.sin( om * trange[index] )
        b = 1.0 - 2.0 * a
        # check time
        if index == 0:
            # modeling equation
            f = a * X.T**2 + b * X.T
            df = 2.0 * a * X.T + b
            # stream function at t = psi = np.sin(np.pi*f*X.T)*np.sin(np.pi*Y.T)
            # vector field at t
            u = -np.pi * A * np.sin(np.pi*f) * np.cos(np.pi*Y.T)          #  d psi / d Y
            v =  np.pi * A * np.cos(np.pi*f) * np.sin(np.pi*Y.T) * df     # -d psi / d X
            # initial conditions
            sol[index,0,:,:] = X.T
            sol[index,1,:,:] = Y.T
            sol_vel[index,0,:,:] = u
            sol_vel[index,1,:,:] = v
        else:
            # dt
            dt = trange[index] - trange[index-1]
            # modeling equation
            f = a * sol[index-1,0,:,:]**2 + b * sol[index-1,0,:,:]
            df = 2.0 * a * sol[index-1,0,:,:] + b
            # stream function at t = psi = np.sin(np.pi*f*sol[index-1,0,:,:])*np.sin(np.pi*sol[index-1,1,:,:])
            # vector field at t
            u = -np.pi * np.sin(np.pi*f) * np.cos(np.pi*sol[index-1,1,:,:])          #  d psi / d Y
            v =  np.pi * np.cos(np.pi*f) * np.sin(np.pi*sol[index-1,1,:,:]) * df     # -d psi / d X
            # after dt
            sol[index,0,:,:] = sol[index-1,0,:,:] + u * dt
            sol[index,1,:,:] = sol[index-1,1,:,:] + v * dt
            sol_vel[index,0,:,:] = u
            sol_vel[index,1,:,:] = v
    # visualization
    images = []
    step = 1
    for t_index in range(0,trange.size,10):
        fig, ax = plt.subplots(2, 2, figsize=(8,4))             # subplot_kw={'projection':'3d'}
        ax[0,0].quiver(sol[0,0,::step,::step], sol[0,1,::step,::step],
                       sol_vel[0,0,::step,::step], sol_vel[0,1,::step,::step], units='xy',
                       scale=3.0, zorder=3, color='blue', width=0.007, headwidth=5.0, headlength=5.0)
        ax[0,0].set_xlabel('x')
        ax[0,0].set_ylabel('y')
        ax[0,0].set_title('u, v at t_index 0')
        ax[0,0].grid(ls=':')
        ax[0,1].quiver(sol[t_index,0,::step,::step], sol[t_index,1,::step,::step],
                       sol_vel[t_index,0,::step,::step], sol_vel[t_index,1,::step,::step], units='xy',
                       scale=20.0, zorder=3, color='blue', width=0.007, headwidth=5.0, headlength=5.0)
        ax[0,1].set_xlabel('x')
        ax[0,1].set_ylabel('y')
        ax[0,1].set_title('u, v at t_index %i' % t_index)
        ax[0,1].grid(ls=':')
        ax[1,0].plot( sol[0,0,:,:].flatten(), sol[0,1,:,:].flatten(), 'r.', markersize=1.0)
        ax[1,0].set_xlabel('x')
        ax[1,0].set_ylabel('y')
        ax[1,0].set_title('particles at t_index 0')
        ax[1,0].grid(ls=':')
        ax[1,1].plot( sol[t_index,0,:,:].flatten(), sol[t_index,1,:,:].flatten(), 'r.', markersize=1.0)
        ax[1,1].set_xlabel('x')
        ax[1,1].set_ylabel('y')
        ax[1,1].set_title('particles at t_index %i' % t_index)
        ax[1,1].grid(ls=':')
        plt.tight_layout()
        output_filename = 'particle_trajectory_%i.png' % t_index
        plt.savefig(output_filename)
        plt.close()
        print(output_filename)
        # making animation in gif 1
        if t_index != 0:
            images.append( PIL.Image.open(output_filename) )
    # making animation in gif 2
    im = PIL.Image.open('particle_trajectory_0.png')
    im.save('particle_trajectory_animation.gif', save_all=True, append_images=images, duration=200, loop=0)


#
# ME565 Lec - 04
#

if False:
    # Cauchy integral formula (CIF)
    N = 1000
    R = 1.0
    theta = np.linspace( complex(0, 0), complex(0, 2.0*np.pi), N)
    dtheta = 2.0*np.pi / N
    z = R * np.exp(theta)[:-1]
    dz = complex(0, 1) * z * dtheta     # dL
    f1 = np.cos(z)
    f2 = np.sin(z)
    f3 = np.exp(z) / z
    integ1 = (f1 * dz).sum()            # integ f1(z) dL
    integ2 = (f2 * dz).sum()            # integ f2(z) dL
    integ3 = (f3 * dz).sum()            # integ f3(z) dL
    print('f1 CIF = Re=%.2e, Im=%.2e' % (integ1.real, integ1.imag))
    print('f2 CIF = Re=%.2e, Im=%.2e' % (integ2.real, integ2.imag))
    print('f3 CIF = Re=%.2e, Im=%.2e' % (integ3.real, integ3.imag))


#
# ME565 Lec - 11
#

if False:
    # numerical solution of laplace equation (on retangular grid)
    L, H = 100, 100
    u  = np.zeros([L, H], dtype=float)
    Lu = np.zeros([L, H], dtype=float)
    Au = np.zeros([L, H], dtype=float)
    
    # boundary conditions 1
    u[ 0, :] = 0.0
    u[-1, :] = 0.0
    u[ :, 0] = 1.0
    u[ :,-1] = 1.0
    
    # boundary conditions 2
    u[ 0, :] = 0.0
    u[-1, :] = 0.0
    u[ :, 0] = 0.0
    u[ :,-1] = np.sin( np.arange(L, dtype=float) / L * 2.0 * np.pi )

    #
    Au = u.copy()
    
    # method 1
    dt = 0.1
    loops = 1000
    # CPU time
    start_t = time.time()
    for itera in range(loops):
        # laplacian (5 points stencil)
        for index_x in range(1, L-1):
            for index_y in range(1, H-1):
                Lu[index_x, index_y] = ( -4.0 * u[index_x, index_y] +
                                         u[index_x+1, index_y] + u[index_x-1, index_y] +
                                         u[index_x, index_y+1] + u[index_x, index_y-1] )
        # time evolution (forward Euler)
        u += Lu * dt
    # CPU time
    end_t = time.time()
    print('method 1 > CPU time = %.4fsec' % (end_t - start_t))
    
    # method 2
    dt = 0.1
    loops = 1000
    # CPU time
    start_t = time.time()
    for itera in range(loops):
        # laplacian (5 points stencil)
        for index_x in range(1, L-1):
            for index_y in range(1, H-1):
                Au[index_x, index_y] = (Au[index_x-1, index_y] + Au[index_x+1, index_y] +
                                        Au[index_x, index_y-1] + Au[index_x, index_y+1]) / 4.0
    # CPU time
    end_t = time.time()
    print('method 2 > CPU time = %.4fsec' % (end_t - start_t))
    
    # visualization
    fig, ax = plt.subplots(1, 2, figsize=(9,4))
    ax[0].imshow(u)
    ax[1].imshow(Au)
    plt.tight_layout()
    plt.show()
    plt.close()

if False:
    # finding coefficient 
    L, H = 100.0, 100.0
    xrange, yrange = np.arange(L), np.arange(H)
    X, Y = np.meshgrid(xrange, yrange)
    # boundary conditions
    bc = np.sin(2.0*np.pi*yrange/H)
    # calation of coefficient
    a2 = 2.0 / (H * np.sinh(2.0*np.pi*L/H) ) * (np.sin(2.0*np.pi/H*yrange)**2).sum()
    print('A2 = %.3e' % a2)
    # analytic solution
    u = a2 * np.sin(2.0*np.pi/H*Y) * np.sinh(2.0*np.pi/H*X)
    # visualization
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(u)
    plt.tight_layout()
    plt.show()
    plt.close()


#
# ME565 Lec - 12
#

if False:
    # Fourier series
    dx = 0.005
    L = 1.0
    fs_order = 100
    x = np.arange(dx, L+dx, dx, dtype=float)
    f = np.ones(x.shape, dtype=float)
    f[:int(f.shape[0]/2)] = 0.0
    # finding coefficients
    f_fs, An, Bn = [], [], []
    A0 = 2.0 / L * ( f * np.cos( 2.0 * np.pi * 0 * x / L ) ).sum() * dx
    f_fs.append( A0/2.0 * np.ones( x.shape, dtype=float ) )
    for index_n in range(1, fs_order):
        An.append( 2.0 / L * ( f * np.cos( 2.0 * np.pi * index_n * x / L ) ).sum() * dx )
        Bn.append( 2.0 / L * ( f * np.sin( 2.0 * np.pi * index_n * x / L ) ).sum() * dx )
        f_fs.append( An[-1] * np.cos( 2.0 * np.pi * index_n * x / L ) +
                     Bn[-1] * np.sin( 2.0 * np.pi * index_n * x / L ) )
    f_fs = np.vstack( f_fs ).sum( axis=0 )
    # debugging
    print(x.shape, f.shape, f_fs.shape)
    # visualization
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.plot(x, f, label='f(x)')
    ax.plot(x, f_fs, label='f_fs(x) order=%i' % fs_order)
    ax.grid(ls=':')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect(1.0)
    plt.tight_layout()
    plt.show()
    plt.close()


#
# ME565 Lec - 16
#

if False:
    # data
    dt = 0.001
    t = np.arange(0.0, 1.0+dt, dt, dtype=float)
    x = np.sin(2.0*np.pi*50.0*t) + np.sin(2.0*np.pi*120.0*t)
    # fast Fourier transform
    N = t.shape[0]
    Y = np.fft.fft(x, N)                # FFT
    PSD = Y * Y.conj() / N              # power spectrum density
    freq = 1/(dt*N) * np.arange(N)      # frequency
    L = np.where( freq < freq[-1]/2.0 ) # half
    # visualization
    fig, ax = plt.subplots(1, 2, figsize=(9,5))
    ax[0].plot(t, x)
    ax[1].plot(freq[L], PSD[L])
    plt.tight_layout()
    plt.show()
    plt.close()
    
if False:
    # fast Fourier transform
    N = 500
    x = np.ones(N, dtype=float)
    y = np.fft.fft(x)
    # discrete Fourier transform > vandermonde matrix (slow)
    start_t = time.time()
    w = np.exp( complex(0.0, -2.0 * np.pi / N) )
    DFT1 = np.zeros([N, N], dtype=float)
    for index_x in range(N):
        for index_y in range(N):
            DFT1[index_x, index_y] = w**(index_x * index_y)
    end_t = time.time()
    print('DFT matrix 1 CPU time = %.3esec' % (end_t-start_t))
    print(DFT1.shape)
    # discrete Fourier transform > vandermonde matrix (fast)
    start_t = time.time()
    w = np.exp( complex(0.0, -2.0 * np.pi / N) )
    xrange, yrange = np.arange(N, dtype=float), np.arange(N, dtype=float)
    X, Y = np.meshgrid(xrange, yrange)
    DFT2 = w**(X.T * Y.T)
    end_t = time.time()
    print('DFT matrix 2 CPU time = %.3esec' % (end_t-start_t))
    print(DFT2.shape)
    # visualization
    fig, ax = plt.subplots(1, 2, figsize=(9,5))
    ax[0].imshow(DFT1.real)
    ax[0].set_aspect(1.0)
    ax[1].imshow(DFT2.real)
    ax[1].set_aspect(1.0)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    













    






