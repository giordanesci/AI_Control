"""
Double pendulum simulator 

At the moment there is no control force on the cart (u=0 constant), 
so the pendulum simply falls down freely according to the initial position 
(x as initialised) 
"""
from Inverted_double_pendulum import step, DEFAULT_PARAMS
import numpy as np
import matplotlib.pyplot as plt


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

for i in range(n_steps):
    x_new = step(x, u, dt, None, w)

    x_hist[:,i] = x_new
    u_hist[i] = u 
    t_hist[i] = i*dt

    x = x_new


def get_cartesian_tip_pos(x_hist):

    l1 = DEFAULT_PARAMS["l1"]
    l2 = DEFAULT_PARAMS["l2"]

    x_tip = l1 * np.sin(x_hist[1,:]) + l2 * np.sin(x_hist[2,:])
    y_tip = l1 * np.cos(x_hist[1,:]) + l2 * np.cos(x_hist[2,:])

    return x_tip, y_tip

x_tip, y_tip = get_cartesian_tip_pos(x_hist)

# Figure (trail in time of the tip of the double pendulum, no floor)
t = t_hist

plt.scatter(x_tip, y_tip, s=2 ,c=t, cmap = 'viridis', alpha=0.7,)

plt.colorbar(label='time[s]')
plt.show()
