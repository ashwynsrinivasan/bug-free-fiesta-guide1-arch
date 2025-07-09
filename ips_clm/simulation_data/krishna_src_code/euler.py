import numpy as np

# Function to time step on a state space system
def euler(A, B, C, D, dt, xkm1, uk):
    Na = np.shape(A)
    xk = np.linalg.solve((np.eye(Na[0]) - dt*A), (xkm1 + dt*np.dot(B,uk)))
    yk = C*xk + D*uk
    return yk, xk
