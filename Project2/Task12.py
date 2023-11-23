import numpy as np
import matplotlib.pyplot as plt
from twopBVP import twopBVP

def I(x):
    return 10e-3 * (3 - 2 * np.cos((np.pi * x)/L)**12)

E = 1.9e11
L, alpha, beta, N = 10, 0, 0, 1001
# Interior points only, no x_0 x_N+1
grid = np.linspace(0, L, N - 1, endpoint=False)[1:]
grid_endpoints = np.linspace(0, L, N, endpoint=True)
load = np.ones(N-2) * -50

bending_moment = twopBVP(load, alpha, beta, L, N)

M = bending_moment[1:-1]/(E*I(grid))

deflection = twopBVP(M, alpha, beta, L, N)

# plt.plot(grid_endpoints, bending_moment, label='Bending Moment')
plt.plot(grid_endpoints, deflection, label='Deflection')
plt.show()
