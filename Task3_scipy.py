import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def van_der_pol(t, u, mu):
    y_1, y_2 = u
    dydt = np.array([y_2, mu * (1 - y_1*y_1) * y_2 - y_1])
    return dydt

mu = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000, 10_000]

Tol = 10e-10
Y0 = np.array([2, 0])
solutions = []
sol_times = []

for idx,i in enumerate(mu):
    f = lambda t, u: van_der_pol(t,u,i)

    T, N = 10*i, 1000
    h = T/N
    t_grid = np.arange(0,T+h,h)
    
    vals = solve_ivp(f, [0,T+h], Y0, method='BDF')
    solutions.append(np.transpose(vals.y))
    sol_times.append(vals.t)

    
    
# It's kinda logarithmic in \mu

index = 6
# Task 3.1
plt.title(r'Plot of $y_2$ for $\mu = 100$')
plt.plot(sol_times[index],solutions[index][:,1])
plt.show()
# Phase plot
plt.title(r'Phase plot for $\mu = 100$')
plt.plot(solutions[index][:,0],solutions[index][:,1])
plt.show()

# Task 3.2
plt.title(r'Phase plots for different $\mu$')
for idx, i in enumerate(solutions):
    plt.plot(i[:,0],i[:,1],label=f'$\mu = {{{mu[idx]}}}$')

plt.legend()
plt.show()

plt.title(r'Number of time-steps taken depending on $\mu$')
plt.plot(mu, [*map(len,sol_times)],'bx')
plt.show()
