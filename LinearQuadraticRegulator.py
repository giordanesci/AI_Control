from scipy.linalg import solve_continuous_are
from Inverted_double_pendulum import step, _xdot, DEFAULT_PARAMS
import numpy as np

def linearize_continuous(x_eq, u_eq, params=None, eps=1e-6):
    p = params if params is not None else DEFAULT_PARAMS
    w = (0.0, 0.0, 0.0)
    n = len(x_eq)
    f0 = _xdot(x_eq, u_eq, w, p)
    
    A = np.zeros((n, n))
    for j in range(n):
        xp = x_eq.copy(); xp[j] += eps
        A[:, j] = (_xdot(xp, u_eq, w, p) - f0) / eps
    
    B = (_xdot(x_eq, u_eq + eps, w, p) - f0) / eps  # shape (6,)
    return A, B.reshape(-1, 1)

# x_eq=np.zeros(6)

def get_K_LQR(x_eq):
    A, B = linearize_continuous(x_eq, 0.0)
    Q = np.diag([1, 10, 10, 1, 1, 1])  # penalize angles heavily
    R = np.array([[0.01]])              # cheap control

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

