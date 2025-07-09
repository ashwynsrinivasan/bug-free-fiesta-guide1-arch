import numpy as np
from numpy import matrix
# Function to time step on a state space system
def euler(A, B, C, D, dt, xkm1, uk):
    if type(A) is matrix:
        A = np.array(A)
    if type(B) is matrix:
        B = np.array(B)
    if type(C) is matrix:
        C = np.array(C)
    if type(D) is matrix:
        D = np.array(D)

    Na = np.shape(A)
    xk = np.linalg.solve((np.eye(Na[0]) - dt*A), (xkm1 + dt*np.dot(B,uk)))
    yk = C*xk + D*uk
    return yk, xk
