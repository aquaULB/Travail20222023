import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

'''
Unless stated otherwise, all the code below is my own content.
'''

def d1_mat(nx,dx):
    '''
    Constructs the second-order centered first-order derivative.
    
    Parameters
    ----------
    nx : integer
        Number of grid points
    dx : float
        Grid spacing

    Returns
    ----------
    d1_mat : np.ndarray
        Matrix to compute the second-order centered first-order derivative
    '''
    diagonals = [[-1/2],[1/2]]
    offsets = [-1,1]

    return diags(diagonals, offsets, shape=(nx, nx)).A / dx

def d3_mat(nx,dx):
    '''
    Constructs the second-order centered third-order derivative.
    
    Parameters
    ----------
    nx : integer
        Number of grid points
    dx : float
        Grid spacing

    Returns
    ----------
    d3_mat : np.ndarray
        Matrix to compute the second-order centered third-order derivative
    '''
    diagonals = [[-1/2],[1],[-1],[1/2]]
    offsets = [-2,-1,1,2]

    return diags(diagonals, offsets, shape=(nx, nx)).A / dx**3

def rhs(u,d1m2nd,d3m2nd):
    '''
    Constructs the solution at each stencil at constant time.
    
    Parameters
    ----------
    u : np.ndarray
        Vector containing the solution at a certain grid point $j$.

    Returns
    ----------
    u : np.ndarray
        Vector containing the solution at the grid point $j+1$.
    '''
    return -6*u*(d1m2nd @ u) - d3m2nd @ u

def rk4(u,f,dt,d1m2nd,d3m2nd):
    '''
    Finds the solution at the next time at constant stencil.
    
    Parameters
    ----------
    u : np.ndarray
        Vector containing the solution at a certain grid point.
    f : python function
        Performs the time-independant computations
    dt : float
        Time step

    Returns
    ----------
    d3_mat : np.ndarray
        Matrix to compute the second-order centered third-order derivative
    '''
    k1 = f(u,d1m2nd,d3m2nd)
    k2 = f(u+dt*k1/2,d1m2nd,d3m2nd)
    k3 = f(u+dt*k2/2,d1m2nd,d3m2nd)
    k4 = f(u+k3*dt,d1m2nd,d3m2nd)
    return u + dt/6 *(k1+2*(k2+k3)+k4)

def soliton(x,t,c,a):
    return c/2 * np.cosh(np.sqrt(c)/2*(x-c*t-a))**(-2)

def solution_soliton(x,dx,nx,dt,nt,a,c,e,c2,a2):
    d1m2nd = d1_mat(nx,dx)
    d3m2nd = d3_mat(nx,dx)

    u = np.empty((nt+1,nx))
    u[0] = soliton(x,0,c,a) + e*soliton(x,0,c2,a2)

    for i in range(nt):
        u[i+1] = rk4(u[i],rhs,dt,d1m2nd,d3m2nd)
    return u

def animate(frame, dt, x, u,line):
    line.set_data(x, u[frame])
    
    line.set_label(f"t={frame * dt: .2f}")
    
    ax.legend(loc="upper right")
    
    return line,