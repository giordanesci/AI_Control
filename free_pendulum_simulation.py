"""
Double pendulum simulator 


"""
from Inverted_double_pendulum import step, _xdot, DEFAULT_PARAMS
import numpy as np
import matplotlib.pyplot as plt
from LinearQuadraticRegulator import get_K_LQR
from matplotlib.animation import FuncAnimation


# Initializing state vector x and parameters 
x_eq = np.zeros(6)        # equilibrium poin or the project
x = [0, 0, 0.05,0, 0, 0] # [cart position, theta 1, theta 2, x_cart dot, theta 1 dot, theta 2 dot]
u = 0                     # initial control force on the cart now
dt = 0.001                # simulation timestep
w = (0.0, 0.0, 0.0)       # external disturbances

n_steps = 2500            # milliseconds

# Preallocation for data
t_hist = np.zeros(n_steps + 1) 
x_hist = np.zeros((6,n_steps + 1))
u_hist = np.zeros(n_steps + 1)
x_hist[:,0] = x
u_hist[0] = u
t_hist[0] = 0

#____________________________________________________
# Per-step integrator using LQR at the moment
#_____________________________________________________

K = get_K_LQR(x_eq)

for i in range(n_steps):
    x_new = step(x, u, dt, None, w)

    # Controller
    u_new = float((-K @ (np.array(x_new) - x_eq))[0])

    # Update state vectors for plotting
    x_hist[:,i] = x_new
    u_hist[i] = u 
    t_hist[i] = i*dt

    # Reset state vector and action for next step
    x = x_new
    u = u_new


def get_cartesian_tip_pos(x_hist):

    l1 = DEFAULT_PARAMS["l1"]
    l2 = DEFAULT_PARAMS["l2"]

    x_tip = l1 * np.sin(x_hist[1,:]) + l2 * np.sin(x_hist[2,:])
    y_tip = l1 * np.cos(x_hist[1,:]) + l2 * np.cos(x_hist[2,:])

    return x_tip, y_tip

x_tip, y_tip = get_cartesian_tip_pos(x_hist)

#_______________________________________________
# Figures
#______________________________________________
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
