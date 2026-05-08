"""
pendulum_plant.py
=================
Inverted double pendulum on a cart — physics only.

The single public function is:

    x_next = step(x, u, dt, params, w)

which advances the state by one time step using RK4.

State vector (length-6 numpy array):
    x[0]  q         cart position          [m]
    x[1]  theta1    rod-1 angle (upright=0) [rad]
    x[2]  theta2    rod-2 angle (upright=0) [rad]
    x[3]  q_dot
    x[4]  theta1_dot
    x[5]  theta2_dot

Sign convention:  theta > 0  →  leaning right,  u > 0  →  force to the right.
"""

import numpy as np
import math

# ── Default physical parameters ──────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    m  = 1.0,    # cart mass               [kg]
    m1 = 0.5,    # mass at rod junction    [kg]
    m2 = 0.3,    # mass at rod tip         [kg]
    l1 = 0.6,    # length of first rod     [m]
    l2 = 0.4,    # length of second rod    [m]
    d1 = 0.1,    # cart damping            [N·s/m]
    d2 = 0.01,   # joint-1 damping         [N·m·s/rad]
    d3 = 0.01,   # joint-2 damping         [N·m·s/rad]
    g  = 9.81,   # gravity                 [m/s²]
)

# ── Internal helpers ─────────────────────────────────────────────────────────

def _mass_matrix(th1, th2, p):
    m, m1, m2 = p['m'], p['m1'], p['m2']
    l1, l2    = p['l1'], p['l2']
    return np.array([
        [m + m1 + m2,              l1*(m1+m2)*math.cos(th1),          m2*l2*math.cos(th2)          ],
        [l1*(m1+m2)*math.cos(th1), l1**2*(m1+m2),                     l1*l2*m2*math.cos(th1-th2)   ],
        [l2*m2*math.cos(th2),      l1*l2*m2*math.cos(th1-th2),        l2**2*m2                     ],
    ])

def _rhs(th1, th2, dq, dth1, dth2, u, w, p):
    m1, m2        = p['m1'], p['m2']
    l1, l2        = p['l1'], p['l2']
    d1, d2, d3    = p['d1'], p['d2'], p['d3']
    g             = p['g']
    w1, w2, w3    = w

    f1 = l1*(m1+m2)*dth1**2*math.sin(th1) + m2*l2*dth2**2*math.sin(th2) - d1*dq + u + w1
    f2 = -l1*l2*m2*dth2**2*math.sin(th1-th2) + g*(m1+m2)*l1*math.sin(th1) - d2*dth1 + w2
    f3 =  l1*l2*m2*dth1**2*math.sin(th1-th2) + g*l2*m2*math.sin(th2)      - d3*dth2 + w3

    return np.array([f1, f2, f3])

def _xdot(x, u, w, p):
    th1, th2        = x[1], x[2]
    dq, dth1, dth2  = x[3], x[4], x[5]

    M   = _mass_matrix(th1, th2, p)
    f   = _rhs(th1, th2, dq, dth1, dth2, u, w, p)
    acc = np.linalg.solve(M, f)          # [q̈, θ̈₁, θ̈₂]

    return np.array([dq, dth1, dth2, acc[0], acc[1], acc[2]])

# ── Public API ────────────────────────────────────────────────────────────────

def step(x, u, dt, params=None, w=(0.0, 0.0, 0.0)):
    """
    Advance the pendulum state by one time step using RK4.

    Parameters
    ----------
    x      : array-like, shape (6,)
             Current state [q, th1, th2, dq, dth1, dth2]
    u      : float
             Control force applied to the cart [N]
    dt     : float
             Time step [s]
    params : dict or None
             Physical parameters. Uses DEFAULT_PARAMS if None.
    w      : array-like, shape (3,)  (optional)
             External disturbances [w1, w2, w3]. Default = zero.

    Returns
    -------
    x_next : np.ndarray, shape (6,)
             State at t + dt
    """
    p = params if params is not None else DEFAULT_PARAMS
    x = np.asarray(x, dtype=float)
    w = tuple(w)

    k1 = _xdot(x,            u, w, p)
    k2 = _xdot(x + dt/2*k1,  u, w, p)
    k3 = _xdot(x + dt/2*k2,  u, w, p)
    k4 = _xdot(x + dt*k3,    u, w, p)

    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)