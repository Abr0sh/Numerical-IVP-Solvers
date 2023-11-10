import numpy as np
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
import ode_solver
from tqdm import tqdm
class LinearIVP:
    # This is a help class that contains functions that are useful in the study of linear 
    # initial value problems of the form dy/dt = A y
    def __init__(self,A):
        self.A = A
    def set_initial_conditions(self,y0):
        self.Y0 = y0

    
    def exact_solution(self,t_grid,y0):
        A = self.A
        return np.array([np.dot(expm(t*A),y0) for t in t_grid]) 

    def Lin_IVP_function(self):
     A = self.A
     def f(t,u): 
          return np.dot(A,u)
     return f