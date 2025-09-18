###ON JAX###
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

# Constants
I = 4.667e-10  # Moment of inertia (kg·m²)
epsilon = 8.854e-12  # Permittivity (F/m)
A = 2e-6  # Effective overlap area between electrodes and rotor arms (m²)
V0 = 600  # Peak voltage (V)
R = 0.005  # Radius of rotor (m)
d0 = 0.001  # Baseline distance (m)
mu = 4.76e-4 * I
m = 0.0000414 #kg
Iz = 4.64e-10
Ix = 2.33e-10
Iy = Ix
fnx = 8.83
fny = 17.89
fnz = 22.31
Qx = 22
Qy = 19
Qz = 13
gamma = 0.118  # Friction coefficient (rad/s)
c = (m*2*jnp.pi*fnx)/Qx  # Damping coefficient (N·m·s)
k = ((2*jnp.pi*fnx)**2)*m


# Rotor arms and electrode angles
arm_angles = jnp.array([0, jnp.pi / 2, jnp.pi, (3*jnp.pi)/2])  # Creates 1D array of arms
electrode_angles = jnp.linspace(0, 2 * jnp.pi, 6, endpoint=False) # 6 electrodes

# Precomputed constants
R2 = 2 * R
d0_squared = d0**2
epsilon_A = epsilon * A
R_epsilon_A = epsilon_A * R

# Define the system of ODEs for theta, theta_dot, theta_ddot, x, x_dot, x_ddot
def equation_of_motion(y, t, *args): # Corrected order of arguments and added *args
    theta, theta_dot, x, x_dot = y
    total_tau = 0
    theta_ddot = (total_tau - mu * theta_dot) / I
    driving_force = jnp.sin(x) + jnp.cos(x) # Corrected exponential syntax
    x_ddot = (driving_force - k * x - c * x_dot) / m
    return jnp.array([theta_dot, theta_ddot, x_dot, x_ddot])

# Simulation parameters
t_eval = jnp.linspace(0, 30, 1000)

# Initial conditions
theta_0 = 0.1  # Initial angular position (rad)
theta_dot_0 = 0.1 # Initial angular velocity (rad/s). Keep constant.
x_0 = 0.1  # Initial x position (m)
x_dot_0 = 0.1  # Initial x velocity (m/s)
y0 = jnp.array([theta_0, theta_dot_0, x_0, x_dot_0])

solution = odeint(equation_of_motion, y0, t_eval, rtol = 1e-6, atol = 1e-4)

# Extract results - solution is now an array where columns are variables
theta_values = solution[:, 0]
theta_dot_values = solution[:, 1]
x_values = solution[:, 2]
x_dot_values = solution[:, 3]

# The lists for appending are not needed with odeint from jax.experimental.ode
all_time = t_eval
all_theta = theta_values
all_theta_dot = theta_dot_values
all_x = x_values
all_x_dot = x_dot_values

# Calculate overall driving force at each time point
all_driving_forces = jnp.sin(all_x) + jnp.cos(all_x) # Corrected exponential syntax

# Check for inf or NaN values before plotting
if jnp.any(jnp.isinf(all_theta)) or jnp.any(jnp.isnan(all_theta)) or \
   jnp.any(jnp.isinf(all_theta_dot)) or jnp.any(jnp.isnan(all_theta_dot)) or \
   jnp.any(jnp.isinf(all_x)) or jnp.any(jnp.isnan(all_x)) or \
   jnp.any(jnp.isinf(all_x_dot)) or jnp.any(jnp.isnan(all_x_dot)):
  print("Simulation results contain infinite or NaN values. Skipping plotting.")
else:
     # Updated plots to include torque
      fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True) #give 5 rows and 1 column of plots

      # Plot Theta against Time
      ax0 = axs[0] #take first subplot from axs array
      ax0.plot(all_time, all_theta, label="Theta (rad)", color='blue')
      ax0.set_ylabel("Theta (rad)", color='blue')
      ax0.set_title("Theta vs Time")
      ax0.legend(loc="upper left")
      ax0.grid(True) #provide grid lines

      # Plot Angular Velocity against Time
      ax1 = axs[1]
      ax1.plot(all_time, all_theta_dot / (2 * jnp.pi), label="Angular Velocity (Hz)", color='red') #divide by 2pi to convert from rad/s to Hz
      ax1.set_xlabel("Time (s)")
      ax1.set_ylabel("Angular Velocity (Hz)", color='orange')
      ax1.set_title("Angular Velocity vs Time")
      ax1.legend(loc="upper left")
      ax1.grid(True)


      # Plot Mode Position against Time
      ax2 = axs[2]
      ax2.plot(all_time, all_x, label="Mode Position", color='pink')
      ax2.set_xlabel("Time (s)")
      ax2.set_ylabel("Mode Position", color='orange')
      ax2.set_title("Mode Position vs Time")
      ax2.legend(loc="upper left")
      ax2.grid(True)

      # Plot Mode Vibration Velocity against Time
      ax3 = axs[3]
      ax3.plot(all_time, all_x_dot, label="Mode Vibration Velocity", color='pink')
      ax3.set_xlabel("Time (s)")
      ax3.set_ylabel("Mode Vibration Velocity", color='orange')
      ax3.set_title("Mode Vibration Velocity vs Time")
      ax3.legend(loc="upper left")
      ax3.grid(True)

      # Plot Driving Force
      axs[4].plot(all_time, all_driving_forces, label="Driving Force (N)", color='green')
      axs[4].set_xlabel("Time (s)")
      axs[4].set_ylabel("Driving Force (N)")
      axs[4].set_title("Driving Force vs Time")
      axs[4].legend()
      axs[4].grid(True)

      # Adjust layout
      plt.tight_layout()
      plt.show()
