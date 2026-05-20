Personal project to explore control methods for simple and complex dynamical scenarios. Ranging from simple PID controllers to AI/ML based control potentially in the future.

I will try to keep this updated with the latest.

## Basics: PID
In PID_controller_simple_process there is a linear process that is controlled with a PID. Integration and derivatives are approximated per step.
Some simple plots describe the process variable value in time and the control input in time. 
Inspired by [this](https://www.digikey.com/en/maker/tutorials/2024/implementing-a-pid-controller-algorithm-in-python).

## Double inverted Pendulum 
Some simulation files have been set up to per-step  simulate the dynamical behaviour of a double ended pendulum as described in [this](https://www3.math.tu-berlin.de/Vorlesungen/SS12/Kontrolltheorie/matlab/inverted_pendulum.pdf) TU Berlin doc. The cart has mass, and the mass of the pendulums is assumed to be concentrated on the tip. The resulting behaviour of the tip of the pendulum on a cart is displayed in the GIF below.
![til](https://github.com/giordanesci/AI_Control/blob/main/trajectory.gif)

## LQR
Given the non-linear equations of motion of the system, with a linearization around one of the unstable equilibrium points (e.g. fully extended vertical pendulum) it is possible to use an LQR to obtain a force u(t) to balance the pendulum about its unstable equilibrium point. 
The equation of motion linearization and the definition of a suitable K matrix to solve the problem is done in LQR_functions.py, and the functions are callable in the main simulation file. 
Deatils on the definition of the merit figures that direct the optimality of the controller are in the LQR_functions.py file.





