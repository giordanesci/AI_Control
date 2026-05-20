"""
Double pendulum simulator 

At the moment there is no control force on the cart (u=0 constant), 
so the pendulum simply falls down freely according to the initial position 
(x as initialised) 
"""
from Inverted_double_pendulum import step, _xdot, DEFAULT_PARAMS
import numpy as np
import matplotlib.pyplot as plt
from LinearQuadraticRegulator import LQR 
from scipy.linalg import solve_continuous_are
from matplotlib.animation import FuncAnimation


# Initializing state vector x and parameters
x = [0, 0, 0.05, 0, 0, 0] # small theta_2 angle
u = 0 # constant zero control force on the cart for now 
dt = 0.001
w = (0.0, 0.0, 0.0) 

n_steps = 2500

# Preallocation for data
t_hist = np.zeros(n_steps + 1) 
x_hist = np.zeros((6,n_steps + 1))
u_hist = np.zeros(n_steps + 1)

x_hist[:,0] = x
u_hist[0] = u
t_hist[0] = 0


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

# Set up LQR
x_eq = np.zeros(6)
A, B = linearize_continuous(x_eq, 0.0)

Q = np.diag([1, 10, 10, 1, 1, 1])  # penalize angles heavily
R = np.array([[0.01]])              # cheap control

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

for i in range(n_steps):
    x_new = step(x, u, dt, None, w)

    # Controller
    u_new = float((-K @ (np.array(x_new) - x_eq))[0])

    x_hist[:,i] = x_new
    u_hist[i] = u 
    t_hist[i] = i*dt

    x = x_new
    u = u_new


def get_cartesian_tip_pos(x_hist):

    l1 = DEFAULT_PARAMS["l1"]
    l2 = DEFAULT_PARAMS["l2"]

    x_tip = l1 * np.sin(x_hist[1,:]) + l2 * np.sin(x_hist[2,:])
    y_tip = l1 * np.cos(x_hist[1,:]) + l2 * np.cos(x_hist[2,:])

    return x_tip, y_tip

x_tip, y_tip = get_cartesian_tip_pos(x_hist)

# Figure (trail in time of the tip of the double pendulum, no floor)
t = t_hist

# Figure 1, Static plot
fig1, ax1 = plt.subplots()

sc = ax1.scatter(
    x_tip,
    y_tip,
    s=2,
    c=t,
    cmap='viridis',
    alpha=0.7
)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1.05)
ax1.set_title("Static point trajectory")

plt.colorbar(sc, ax=ax1, label="time")

# FIGURE 2: Animation
fig2, ax2 = plt.subplots()

ax2.set_title("Animation")
ax2.set_xlim(-0.1, 0.1)
ax2.set_ylim(0.99, 1.01)

line, = ax2.plot([], [], 'b-')
point, = ax2.plot([], [], 'ro')

def update(frame):
    idx = frame * 100
    line.set_data(x_tip[:idx+1], y_tip[:idx+1])
    point.set_data([x_tip[idx]], [y_tip[idx]])
    return line, point

ani = FuncAnimation(
    fig2,
    update,
    frames=len(x_tip)//100,
    interval=100,
    blit=True
)

# ani.save("trajectory.gif", writer="pillow", fps=10)

plt.show()
