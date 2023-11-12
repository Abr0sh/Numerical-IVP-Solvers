import numpy as np
from tqdm import tqdm
class ODE_Solver:
    """
    ODESOLVER Mother class
    
    ODE:
    dy/dt = f(t,y),
    y(t=0)= y0 initial condition
    """

    def __init__(self,f):
        self.f = f

    def set_initial_conditions(self,Y0):
        if isinstance(Y0, (int, float)):
            # 1D ODE
            self.dimension = 1
            Y0 = float(Y0)
        else:
            # n dimensional ODE
            Y0 = np.array(Y0)
            self.dimension = Y0.size
        self.Y0 = Y0 
    
    def solve(self,time_points):
        self.t = np.array(time_points)
        n = self.t.size
        
        self.y = np.zeros((n, self.dimension))
        self.y[0, :] = self.Y0

        for i in range(n-1):
            self.i = i
            self.y[i+1] = self.advance()

        return self.y, self.t
    def advance(self):
        """Advance the solution by one time step"""
        raise NotImplementedError
    

class Euler(ODE_Solver):
    def advance(self):
        y,f,i,t = self.y, self.f, self.i, self.t
        dt = t[i+1]- t[i]
        return y[i, :] + dt*f(t[i],y[i, :])
    def name(self): return "Euler method"

class Implicit_method(ODE_Solver):
    def advance(self):
        y,f,i,t = self.y, self.f, self.i, self.t
        dt = t[i+1]- t[i]
        return self.f(t[i],dt,y[i])
    def name(self): return "Implicit method"

class RungeKutta4(ODE_Solver):
    def advance(self):
        y,f,i,t = self.y, self.f, self.i ,self.t
        dt = t[i+1]-t[i]
        dt2 = dt * 0.5
        K1 = dt*f(t[i],y[i, :])
        K2 = dt*f(t[i]+dt2 ,y[i, :]+ 0.5*K1)
        K3 = dt*f(t[i]+dt2 ,y[i, :]+ 0.5*K2)
        K4 = dt * f(t[i]+ dt, y[i, :]+K3)

        return y[i, :] + (1/6)*(K1 + 2*K2 +2*K3 + K4)
    def name(self): return "Runge Kutta 4 method"

class RungeKutta3(ODE_Solver):
    def advance(self):
        y,f,i,t = self.y, self.f, self.i ,self.t
        dt = t[i+1]-t[i]
        dt2 = dt/2
        K1 = dt*f(t[i],y[i, :])
        K2 = dt*f(t[i]+dt2 ,y[i, :]+ 0.5*K1)
        K3 = dt*f(t[i]+dt2 ,y[i, :]-K1+ 2*K2)
        return y[i, :] + (1/6)*(K1 +4 *K2 +K3 )
    def name(self): return "Runge Kutta 4 method"

class RungeKutta43(ODE_Solver):

    def solve(self,t_0,T,Tol):
        self.TOL = Tol
        self.t = [t_0]
        self.T = T

        # initiate the local errors
        self.l = np.array([self.TOL])
        # initiate the first step size
        self.h = [self.TOL**(0.25)*np.abs(self.T-t_0)/(100*(1+np.linalg.norm(self.f(t_0,self.Y0))))]
        self.y = np.zeros((1, self.dimension))
        self.y[0, :] = self.Y0


        # change this to a while loop and handle last step s.t we end on T
        self.i= 0
        while self.t[self.i] < self.T:
            i = self.i
            self.y = np.concatenate((self.y,np.array([self.advance()])),axis=0)
            self.t.append(self.t[i]+self.h[i])
            self.i = self.i+1
        
        # This block makes sure the last step ends on the specified end point T
        self.i = self.i - 1
        self.t = self.t[:-1]
        self.y = self.y[:-1]
        self.h[i]=self.T-self.t[i]
        self.y = np.concatenate((self.y,np.array([self.advance()])),axis=0)
        self.t.append(self.T)

        return self.y, self.t 
       
    def advance(self):
        i,f = self.i,self.f
        hi,ti,yi = self.h[i],self.t[i],self.y[i,:]
        K1 = hi*f(ti,yi)
        K2 = hi*f(ti+0.5*hi ,yi+ 0.5*K1)
        K3 = hi*f(ti+0.5*hi ,yi+ 0.5*K2)
        Z3 = hi*f(ti+hi ,yi-K1 +2*K2)
        K4 = hi * f(ti+ hi, yi+K3) 
        y_new = yi + (1/6)*(K1 + 2*K2 +2*K3 + K4)
        l = np.array([np.linalg.norm((1/6)*( 2*K2 +Z3 -2*K3 - K4))])
        self.l = np.concatenate((self.l,l),axis =0)
        h_new = self.h[-1]*(self.l[-2]*self.TOL/(self.l[-1])**2)**(1/12)
        self.h.append(h_new)
        
        return y_new
