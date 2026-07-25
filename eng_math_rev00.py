#
# TITLE:
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: engineering mathematics (UW ME564/565), 2014, Steve Brunton
#


import numpy as np
import matplotlib.pyplot as plt


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

