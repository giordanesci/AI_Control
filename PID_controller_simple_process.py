import time
import matplotlib.pyplot as plt

def PID_controller(objective, current_state, dt, kp, kd, ki, previous_error, integral):
    error = objective - current_state
    integral += error * dt
    derivative = (previous_error - error)/dt
    control = kp * error + kd * derivative + ki * integral
    return control, error, integral

def main():
    objective = 100
    current_state = 0 
    control = float
    kp = 1
    kd = .1
    ki = 1
    previous_error = 0
    integral = 0
    dt = 0.1
    
    # initialization for plotting
    time_steps = []
    pv_values = []
    control_values = []
    setpoint_values = []

    for i in range(100):
        control, error, integral = PID_controller(objective, current_state, dt, kp, kd, ki, previous_error, integral)
        current_state += ((control**1.1) - 0.5 * control) * dt
        previous_error = error

        time_steps.append(i*dt)
        pv_values.append(current_state)
        control_values.append(control)
        setpoint_values.append(objective)

        # time.sleep(dt) # to simulate real time

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_steps, pv_values, label='Process Variable (PV)')
    plt.plot(time_steps, setpoint_values, label='Setpoint', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Process Variable vs. Setpoint')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_steps, control_values, label='Control Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Output')
    plt.title('Control Output over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()