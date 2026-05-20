Personal project to explore control methods for simple and complex dynamical scenarios. Ranging from simple PID controllers to AI/ML based control potentially
I will try to keep this updated with the latest.

## PID
In PID_controller_simple_process there is a linear process that is controlled with a PID. Integration and derivatives are approximated per step.
Some simple plots describe the process variable value in time and the control input in time. 
Inspired by: https://www.digikey.com/en/maker/tutorials/2024/implementing-a-pid-controller-algorithm-in-python

## Double inverted Pendulum 
Some simulation files have been set up to per-step  simulate the dynamical behaviour of a double ended pendulum as described in [https://www3.math.tu-berlin.de/Vorlesungen/SS12/Kontrolltheorie/matlab/inverted_pendulum.pdf]. The resulting behaviour of the tip of the pendulum on a cart is displayed in the GIF below.
![til](https://github.com/giordanesci/AI_Control/blob/main/trajectory.gif)
