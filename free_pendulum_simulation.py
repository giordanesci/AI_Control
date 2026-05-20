"""
Double pendulum simulator 

This script is the node in which the loop that spins the simulation of an inverted double pendulum runs. 
The double pendulum is initialized with the equilibrium position x_eq picked as the upright extended position 
(there are 2 more unstable configurations and a stable trivial one). 

The dynamics is described by inverted_double_pendulum.py and takes into account full dynamics of a cart with weight 
and two rods with weight only at their end. 
It is possible to control the cart with an horizontal force u.
It is possible to add some disturbances to the cart that at the moment have a dissipation nature, as are dependent
from a velocity term (to be specified).

After initialization the script loops for each timestep the dynamics and the control response at each step. To change
the controller, simply comment out its relevant matrix or function, both before the loop and in the loop itself, 
leaving only one active controller in the form:

    u_new = ... 
    # u_new = ...
    # ...

The script also plots two figures of the trajectory in time, one static and one animated
"""

from Inverted_double_pendulum import step, DEFAULT_PARAMS 
import numpy as np
import matplotlib.pyplot as plt
from LQR_functions import get_K_LQR
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# Initializing state vector x and parameters 
x_eq = np.zeros(6)        # equilibrium poin or the project
x = [0, 0, 0.05,0, 0, 0] # [cart position, theta 1, theta 2, x_cart dot, theta 1 dot, theta 2 dot]
u = 0                     # initial control force on the cart now
dt = 0.001                # simulation timestep
w = (0.0, 0.0, 0.0)       # external disturbances

n_steps = 5000            # milliseconds

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

K = get_K_LQR(x_eq) # Use this for LQR

for i in range(n_steps):
    x_new = step(x, u, dt, None, w)

    # Controller
    u_new = 0 #float((-K @ (np.array(x_new) - x_eq))[0]) # Use this for LQR

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

fps = 20 
n = int((1/(dt))/fps)

ax2.set_title("Animation")
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.grid(True, alpha=0.2)

cmap = plt.cm.viridis
norm = Normalize(vmin=0, vmax=len(x_tip))
lc = LineCollection([], cmap=cmap, norm=norm, linewidth=3)
ax2.add_collection(lc)

line, = ax2.plot([], [], 'b-')
point, = ax2.plot([], [], 'ro', markersize=6)

# Colorbar
cbar = fig2.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax2
)
cbar.set_label("Time [ms]")

def update(frame):

    idx = frame * n

    # Build line segments
    points = np.array([x_tip[:idx+1], y_tip[:idx+1]]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc.set_segments(segments)

    # Color progression
    lc.set_array(np.arange(len(segments)))

    # Moving tip
    point.set_data([x_tip[idx]], [y_tip[idx]])

    return lc, point

ani = FuncAnimation(
    fig2,
    update,
    frames=len(x_tip)//n,
    interval=n, # real time
    blit=True
)

ani.save("trajectory.gif", writer="pillow", fps=fps)

plt.show()
