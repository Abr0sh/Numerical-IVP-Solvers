import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def twopBVP(fvec, alpha, beta, L, N):
    N = N - 2
    # 1/\del x^2 factor
    del_x = L / (N + 1)
    delx_2 = 1 / (del_x * del_x)

    fvec[0] = fvec[0] - alpha * delx_2
    fvec[-1] = fvec[-1] - beta * delx_2

    d1 = np.ones(N) * delx_2
    d0 = -2 * np.ones(N) * delx_2
    T = spdiags([d1, d0, d1], [-1, 0, 1], N, N, format="csc")

    sol = spsolve(T, fvec)
    #Pre-/Append endpoints
    sol = np.append(sol, beta)
    sol = np.insert(sol, 0, alpha)
    
    return sol
