import numpy as np
import matplotlib.pyplot as plt
from twopBVP import twopBVP
from scipy.stats import linregress

def step_grid(exp_i,exp_f):
    K = np.linspace(exp_i,exp_f,exp_f-exp_i+1)
    N_grid = [int(2**k) for k in K]
    return N_grid


def test_fun(x, L, alpha, beta):
    return (L * alpha - x * alpha + x * beta + x * np.sin(L) - L * np.sin(x)) / L


L, alpha, beta, N = 10, 1, 10, 10000
# Interior points only, no x_0 x_N+1
grid = np.linspace(0, L, N - 1, endpoint=False)[1:]
# Grid with endpoints
grid_endpoints = np.linspace(0, L, N, endpoint=True)
test_fvec = np.sin(grid)

approximation = twopBVP(test_fvec, alpha, beta, L, N)
analytic = test_fun(grid_endpoints, L, alpha, beta)

plt.plot(grid_endpoints, analytic)
plt.plot(grid_endpoints, approximation)
plt.show()

comparison = analytic / approximation - 1
global_err = analytic - approximation

errors = []
steps =  step_grid(2,20)
h = [L / i for i in steps]
for i in steps:
    L, alpha, beta, N = 10, 1, 10, i
    # Interior points only, no x_0 x_N+1
    grid = np.linspace(0, L, N - 1, endpoint=False)[1:]
    # Grid with endpoints
    test_fvec = np.sin(grid)
    grid_endpoints = np.linspace(0, L, i, endpoint=True)
    analytic = test_fun(grid_endpoints, L, alpha, beta)
    approximation = twopBVP(test_fvec, alpha, beta, L, i)
    global_err = np.abs(analytic - approximation)
    errors.append(max(global_err))


cutoff = 15
reg = linregress(np.log(h)[:cutoff], np.log(errors)[:cutoff])

plt.plot(np.arange(0,-15,-1), reg.slope * np.arange(0,-15,-1) + reg.intercept, label=f'slope = {reg.slope}')
plt.plot(np.log(h), np.log(errors),label=f'glob_err')
plt.legend()
plt.show()

