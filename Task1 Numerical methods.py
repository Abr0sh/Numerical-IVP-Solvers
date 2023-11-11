#%% Importing
import numpy as np
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
import ode_solver
from tqdm import tqdm
import scipy
import linearInitialValueProblem as LIVP

# Defining some help functions:

# This function is the stability function for the Implicit Euler (Use it to advance by one step y_n+1 = R()y_n)
def advance_Im_Euler(t,h,yi):
    E = np.identity(len(yi))
    inverted_matrix = np.linalg.inv(E-h*A)
    return np.dot(inverted_matrix,yi)
# Stability function for trapazoidal rule 
def advance_trap(t,h,yi):
    E = np.identity(len(yi))
    inverted_matrix = np.linalg.inv(E-(h/2)*A)
    RHS_matrix = (E+(h/2)*A)
    return np.dot(np.dot(inverted_matrix,RHS_matrix),yi)
# A function to calculate the global error at various step sizes (for fixed final time t =T)
def global_error_in_h(h_grid,T,solver,exact_solution_at_T):
    def calculate_global_error_for_h(h,T):
        t_grid = np.arange(0,T+h,h) 
        y,t = solver.solve(t_grid)
        return np.linalg.norm(exact_solution_at_T-y[-1,:])
    
    E = [calculate_global_error_for_h(h, T) for h in tqdm(h_grid, desc="Processing")]
    return E
# Generate a grid of step size h that will be equidistant on a log scale
def step_grid(T,exp_i,exp_f,):
    K = np.linspace(exp_i,exp_f,exp_f-exp_i+1)
    N_grid = [2**k for k in K]
    return np.array([T/n for n in N_grid],np.float64)

#%% IVP constants and time grid
# Linear Test ivp: dy/dt = A y
A = np.array([[-0.2,0],[2,5]],dtype=np.float64) 
Y0 = np.array([1,1],dtype=np.float64)
# Final time and number of steps
T,N =  np.float64(2) , np.float64(100)
h = T/N             # step size
t_grid = np.arange(0,T+h,h,np.float64)    # time grid
index = 1 # which coordinate of the solution to plot
#%%
# Seting up the linear initial value problem and its exact solutions
livp = LIVP.LinearIVP(A)
livp.set_initial_conditions(Y0)
y_exact=livp.exact_solution(t_grid,Y0) # evaluates the exact solution (e^(At)) at points t in t_grid
f = livp.Lin_IVP_function() # returns a function f(t,u) = Au
 

# Numerical Euler, RK4 and RK3 solvers
solverE = ode_solver.Euler(f)
solver4 = ode_solver.RungeKutta4(f)
solver3 =ode_solver.RungeKutta3(f)
solver43 = ode_solver.RungeKutta43(f)
solverIm_Eul = ode_solver.Implicit_method(advance_Im_Euler) # For implicit methods pass in R() , the stability function you get by solving the system of equations, that is used to advance the solution y_n+1 = R()y_n
solver_tr = ode_solver.Implicit_method(advance_trap)
# setting the initial conditions
solverE.set_initial_conditions(Y0)
solver4.set_initial_conditions(Y0)
solver3.set_initial_conditions(Y0)
solver43.set_initial_conditions(Y0)
solverIm_Eul.set_initial_conditions(Y0)
solver_tr.set_initial_conditions(Y0)
# Getting numerical solutions
y_E,t = solverE.solve(t_grid)
y_Im_E,t = solverIm_Eul.solve(t_grid)
y_tr,t = solver_tr.solve(t_grid)
y_RK4,t = solver4.solve(t_grid)
y_RK3,t = solver3.solve(t_grid)
y_RK43,t43 = solver43.solve(t_grid[0],T,10**(-7))
#%%
# Calculating the global errors in time
error_E = [np.linalg.norm(y_E[i]-y_exact[i] ) for i in range(len(y_exact))] 
error_RK3 = [np.linalg.norm(y_RK3[i]-y_exact[i] ) for i in range(len(y_exact))] 
error_RK4 = [np.linalg.norm(y_RK4[i]-y_exact[i] ) for i in range(len(y_exact))] 
error_Im_E = [np.linalg.norm(y_Im_E[i]-y_exact[i] ) for i in range(len(y_exact))] 
error_tr = [np.linalg.norm(y_tr[i]-y_exact[i] ) for i in range(len(y_exact))] 
y_exact43=livp.exact_solution(t43,Y0) # Caluclating the exact solution for adaptive RK time grid: since the adaptive RK method evaluates at different set of points t in t43
error_RK43 = [np.linalg.norm(y_RK43[i]- y_exact43[i] ) for i in range(len(y_exact43))] 
# Plotting solutions
plt.plot(t_grid,y_E[:,index],label='Euler')
plt.plot(t_grid,y_Im_E[:,index],label='Implicit Euler')
plt.plot(t_grid,y_tr[:,index],label='Trapazoidal')
plt.plot(t_grid,y_RK3[:,index],label='RK3')
plt.plot(t_grid,y_RK4[:,index],label='RK4')
plt.plot(t43,y_RK43[:,index],label='RK43')
plt.plot(t_grid,y_exact[:,index], label='Exact')
plt.xlabel('time grid')
plt.ylabel('y(t) Exact and numerical solutions')
plt.title('Numerical solutions')
plt.legend()
plt.show()

#plotting the global errors as a function of time
plt.semilogy(t_grid,error_E,label='Euler')
plt.semilogy(t_grid,error_Im_E,label='Implicit Euler')
plt.semilogy(t_grid,error_tr,label='Trapazoidal')
plt.semilogy(t_grid,error_RK3,label='RK3')
plt.semilogy(t_grid,error_RK4,label='RK4')
plt.semilogy(t43,error_RK43,label='RK43')
plt.legend()
plt.title('Global error in t')
plt.show()

# %% 

# %% Global Error study:
h_grid = step_grid(T,0,17)
error_h_E = global_error_in_h(h_grid,T,solverE,y_exact[-1,:])
error_h_RK3 = global_error_in_h(h_grid,T,solver3,y_exact[-1,:])
error_h_RK4 = global_error_in_h(h_grid,T,solver4,y_exact[-1,:])
error_h_trap = global_error_in_h(h_grid,T,solver_tr,y_exact[-1,:])
error_h_Im_E = global_error_in_h(h_grid,T,solverIm_Eul,y_exact[-1,:])
#%%
# Plotting errors in h
Color_blue,col_orange,color_red = '#0059de','#de8100','#de0025'
plt.loglog(h_grid,error_h_E,label='Euler',marker='o',color=Color_blue)
plt.loglog(h_grid,error_h_RK3,label='RK3',marker='o',color=col_orange)
plt.loglog(h_grid,error_h_RK4,label='RK4',marker='o',color=color_red)
plt.loglog(h_grid,error_h_trap,label='Trap',marker='o')
plt.loglog(h_grid,error_h_Im_E,label='Implicit Euler',marker='o')

# plot lines Ch^p curves to check the order of a method
# (Tune these constants by hand, what matters is the slope ) error = O(h^p)
C_E = 0.01*10**(11)
C_RK3 = 2*10**(8)
C_RK4 = 10**(9)
C_tr = 10**(6)
#plt.loglog(h_grid,C_E*h_grid,label= 'Ch^1',color=Color_blue)
#plt.loglog(h_grid,C_RK3*h_grid**3,label= 'Ch^3',color=col_orange)
#plt.loglog(h_grid,C_RK4*h_grid**4,label= 'Ch^4',color=color_red)
plt.title('Global error at t=T for different step size h')
plt.legend()
plt.savefig('Global_error_in_h.png',dpi=400)
plt.show()

# %%
