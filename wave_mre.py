# -*- coding: utf-8 -*-
"""
@author: mre

u_tt = u_xx,  x in (0,1), t>0

inital conditions:
        
    u_0(x)   = ...
    u_t_0(x) = 0
    
boundary conditions:   
    
    homogeneous Dirichlet
"""


import matplotlib.pyplot as plt
import numpy as np
import qade

# analytic solution
def uex(x,t):
    u0p = lambda x : 1-2*abs((x+0.5)%2-1) # periodische Fortsetzung
    return 0.5*(u0p(x-t) + u0p(x+t))

def uex(x,t):
    return np.sin(2*np.pi * x) * np.cos(2*np.pi * t)

# def usol(xt):
#     x,t = xt.T
#     return uex(x,t)
    


# grid spacing in x and t
x = np.linspace(0, 1, 101)
t = np.linspace(0, 0.5, 51)

# define function for approximate solution
# 2 scalar parameters in (x, t), one scalar out
u = qade.function(n_in=2, n_out=1)

# collocation points (meshgrid flattened)
xt = np.array([[x_elem, t_elem] for x_elem in x for t_elem in t])

# PDE
pde = qade.equation(u[2, 0] - u[0, 2], xt)

# Initial conditions 
xt0 = np.array([[x_elem, 0] for x_elem in x])

u0 = qade.equation(u[0, 0] - uex(x, 0), xt0)
u1 = qade.equation(u[0, 1] - 0*x      , xt0)
  
# Boundary conditions for phi at x = 0 and x = 1
x0t = np.array([[0, t_elem] for t_elem in t])
bc0 = qade.equation(u[0, 0] - 0 * t, x0t)

x1t = np.array([[1, t_elem] for t_elem in t])
bc1 = qade.equation(u[0, 0] - 0 * t, x1t)

# solve, Fourier base
usol = qade.solve(
    [pde, u0, u1, bc0, bc1], qade.basis("fourier", 3), n_spins=2, verbose = True
)
print(f"loss = {usol.loss:.3}, weights = {np.around(usol.weights, 2)}")



# Show the results
for tk in np.linspace(t.min(), t.max(), 5):
    xt = np.array([[x_elem, tk] for x_elem in x])
    plt.plot(x, usol(xt), label=f"t = {tk}")
    plt.plot(x, uex(x, tk), "k:")

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$u(x, t)$")

