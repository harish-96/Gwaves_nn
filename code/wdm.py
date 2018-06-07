import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def phi(omega, a=1, b=1, n=2):
    Omega = 2*a + b
    if abs(omega) < a:
        return 1/np.sqrt(Omega)
    if abs(omega) > a+2*b:
        return 0
    else:
        return 1/np.sqrt(Omega) * np.cos(nu(n, (abs(omega) - a)/b)*np.pi/2)
    
def nu(n, x):
    return B(x,n,n)/B(1,n,n)

def B(x,a,b):
    if x==0: return 0
    return odeint(lambda y,t: t**(a-1)*(1-t)**(b-1), 0, np.linspace(0, x, 1000)).T[-1][-1]

n = 1
a = 1
b = 1
Omega = 2*a + b
ans = [phi(omega, a, b, n) for omega in np.linspace(-Omega,Omega,1000)]
plt.plot(np.linspace(-Omega, Omega,1000)/Omega, ans)
plt.show()
