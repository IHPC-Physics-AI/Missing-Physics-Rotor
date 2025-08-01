# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:52:34 2024

@author: xfeng
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
I = 4.667e-10  # Moment of inertia (kg·m²)
gamma = 0.118  # Friction coefficient (rad/s)
c = I * gamma  # Damping coefficient (N·m·s)
epsilon = 8.854e-12  # Permittivity (F/m)
A = 2e-6  # Effective area (m²)
V0 = 600  # Peak voltage (V)
R = 0.005  # Radius of rotor (m)
d0 = 0.001  # Baseline distance (m)

# Rotor arms and electrode angles
arm_angles = np.array([0, np.pi / 2])  # Arm 1 and Arm 2
electrode_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)  # 6 electrodes

# Precomputed constants
R2 = 2 * R
d0_squared = d0**2
epsilon_A = epsilon * A
R_epsilon_A = epsilon_A * R

# Precompute triangular waveforms on a time grid
time_grid = np.linspace(0, 1, 1000)
V1_table = V0 * (2 * np.abs(2 * (time_grid % 1) - 1) - 1)
V2_table = np.roll(V1_table, int(len(time_grid) / 3))
V3_table = np.roll(V1_table, int(2 * len(time_grid) / 3))

def electrode_voltages_precomputed(t, f):
    idx = int((f * t) % 1 * len(time_grid))
    v1 = V1_table[idx]
    v2 = V2_table[idx]
    v3 = V3_table[idx]
    return np.array([v1, v2, v3, v1, v2, v3])

# Torque computation
def compute_torques_scalar(theta, t, f):
    angle_diff = theta + arm_angles[:, np.newaxis] - electrode_angles[np.newaxis, :]
    sin_half_angle_diff = np.sin(angle_diff / 2)
    distances_squared = d0_squared + (R2 * sin_half_angle_diff)**2
    voltages_squared = electrode_voltages_precomputed(t, f)**2
    torques = R_epsilon_A * voltages_squared / distances_squared * np.sign(np.sin(angle_diff))
    return np.sum(torques)
    #torques = R_epsilon_A * voltages_squared / distances_squared * np.sign(np.sin(angle_diff))*R2*sin_half_angle_diff/np.sqrt(distances_squared)
    
# Define the system of ODEs for theta and theta_dot
def equation_of_motion(t, y, f):
    theta, theta_dot = y
    total_tau = compute_torques_scalar(theta, t, f)
    theta_ddot = (total_tau - c * theta_dot) / I
    return [theta_dot, theta_ddot]

# Simulation parameters
t_span = (0, 30)  # Time range (s)
t_eval = np.linspace(t_span[0], t_span[1], 500)  
frequencies = np.arange(0.5, 8.5, 0.5)  # Frequencies from 0.5 Hz to 8 Hz

# Initial conditions
theta_0 = 0.1  # Initial angular position (rad)
theta_dot_0 = 0.1  # Initial angular velocity (rad/s)
y0 = [theta_0, theta_dot_0]

# Arrays to store results
all_theta = []
all_theta_dot = []
all_time = []
all_frequency = []

# Frequency sweep loop
for frequency in frequencies:
    # Solve the ODE for the current frequency
    solution = solve_ivp(equation_of_motion, t_span, y0, t_eval=t_eval, method='RK45', args=(frequency,))
    
    # Extract results
    time_values = solution.t
    theta_values = solution.y[0]
    theta_dot_values = solution.y[1]
    
    # Append results
    all_time.extend(time_values + (all_time[-1] if all_time else 0))  # Adjust time for continuity
    all_theta.extend(theta_values)
    all_theta_dot.extend(theta_dot_values)
    all_frequency.extend([frequency] * len(time_values))  # Add frequency for each time step
    
    # Update initial conditions for the next frequency
    y0 = [theta_values[-1], theta_dot_values[-1]]

# Convert lists to NumPy arrays for operations
all_theta = np.array(all_theta)
all_theta_dot = np.array(all_theta_dot)
all_time = np.array(all_time)
all_frequency = np.array(all_frequency)

# Calculate overall torque
all_torques = np.array([
    compute_torques_scalar(theta, t, frequency)
    for theta, t, frequency in zip(all_theta, all_time, all_frequency)
])

# Updated plots to include torque
fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Plot Theta with frequency on the secondary y-axis
ax0 = axs[0]
ax1 = axs[0].twinx()
ax0.plot(all_time, all_theta, label="Theta (rad)", color='blue')
ax1.plot(all_time, all_frequency, label="Frequency (Hz)", color='red', linestyle='--')
ax0.set_ylabel("Theta (rad)", color='blue')
ax1.set_ylabel("Frequency (Hz)", color='red')
ax0.set_title("Theta vs Time (Frequency Sweep)")
ax0.legend(loc="upper left")
ax1.legend(loc="upper right")
ax0.grid(True)

# Plot Angular Velocity with frequency on the secondary y-axis
ax2 = axs[1]
ax3 = axs[1].twinx()
ax2.plot(all_time, all_theta_dot / (2 * np.pi), label="Angular Velocity (Hz)", color='orange')
ax3.plot(all_time, all_frequency, label="Frequency (Hz)", color='red', linestyle='--')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Angular Velocity (Hz)", color='orange')
ax3.set_ylabel("Frequency (Hz)", color='red')
ax2.set_title("Angular Velocity vs Time (Frequency Sweep)")
ax2.legend(loc="upper left")
ax3.legend(loc="upper right")
ax2.grid(True)

# Plot Torque
axs[2].plot(all_time, all_torques, label="Torque (N·m)", color='green')
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Torque (N·m)")
axs[2].set_title("Torque vs Time (Frequency Sweep)")
axs[2].legend()
axs[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
