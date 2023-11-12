#%% Importing
import numpy as np
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
import ode_solver
from tqdm import tqdm
import scipy
import linearInitialValueProblem as LIVP

# Defining some help functions:
# defines the right hand side of the Lotta Volterra IVP i.e the f in dy/dt = f(t,y)
def Lotta_Volterra(a,b,c,d):
    def f(t,u):
        return np.array([a*u[0]-b*u[0]*u[1],c*u[0]*u[1]-d*u[1]])
    def H(x,y):
        return c*x + b*y - d*np.log(x) - a*np.log(y)
    return f,H



#%% IVP constants and time grid
# ivp: dy/dt = f()y
a,b,c,d = 3, 9, 15, 15 
f,H = Lotta_Volterra(a,b,c,d)
Y0 = np.array([1.5,0.5])
Tol = 10**(-10)     # the tolerance used in RK43
T,N =  20 , 1000    # Final time and number of steps
h = T/N             # step size
t_grid = np.arange(0,T+h,h)    # time grid
index = 1           # which coordinate of the solution to plot
#%%

# Numerical Euler, RK4 and RK3 solvers

solver4 = ode_solver.RungeKutta4(f)
solver3 = ode_solver.RungeKutta3(f)
solver43 = ode_solver.RungeKutta43(f)


# setting the initial conditions
solver4.set_initial_conditions(Y0)
solver3.set_initial_conditions(Y0)
solver43.set_initial_conditions(Y0)

# Getting numerical solutions
y_RK4,t = solver4.solve(t_grid)
y_RK3,t = solver3.solve(t_grid)
y_RK43,t43 = solver43.solve(t_grid[0],T,Tol)
#%%

# Plotting solutions
plt.plot(t_grid,y_RK3[:,index],label='RK3')
plt.plot(t_grid,y_RK4[:,index],label='RK4')
plt.plot(t43,y_RK43[:,index],label='RK43')
plt.xlabel('time grid')
plt.ylabel('y(t) Exact and numerical solutions')
plt.title(f'Lotta_Volterra a,b,c,d= {a},{b},{c},{c} and x0,y0 = {Y0}')
plt.legend()
plt.show()

plt.plot(y_RK4[:,0],y_RK4[:,1],label ='y(x)')
plt.title(f'Lotta_Volterra phase portrait x0,y0 = {Y0}')
plt.legend()
plt.xlabel('x (# of Rabbits)')
plt.ylabel('y (# of Foxes)')
plt.show()
H_t = [round(H(y_RK43[i,0],y_RK43[i,1]),4) for i in range(len(t43))]
H_0 = H(*Y0)
H_t = np.abs((H_t / H_0) - 1)
plt.plot(t43,H_t,label = 'H(x,y)')
plt.legend()
plt.show()
# %%
