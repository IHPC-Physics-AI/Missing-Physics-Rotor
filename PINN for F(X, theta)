## Multiple Trajectories per Batch ##

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
import numpy as np  # for random number generation outside JAX

# Initialize random key
rng = jax.random.PRNGKey(42)

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
c = (m*2*np.pi*fnx)/Qx  # Damping coefficient (N·m·s)
k = ((2*np.pi*fnx)**2)*m

total_tau = 0

# Define the system of ODEs
def equation_of_motion(y, t):
    theta, theta_dot, x, x_dot = y
    total_tau = 0
    driving_force = jnp.sin(x) * jnp.sin(theta)
    theta_ddot = (total_tau - mu * theta_dot) / I
    x_ddot = (driving_force - k * x - c * x_dot) / m
    return jnp.array([theta_dot, theta_ddot, x_dot, x_ddot])

# Generate full training data (over [0,1] seconds)
t_eval = jnp.linspace(0, 1, 100)  # Adjust this to [0,5] if needed
y0_train = jnp.array([0.1, 0.1, 0.1, 0.1])
solution_train = odeint(equation_of_motion, y0_train, t_eval, rtol=1e-3, atol=1e-3)
print("Full solution shape:", solution_train.shape)  # (100, 4)

# Calculate true acceleration for reference
driving_force_train = jnp.sin(solution_train[:, 0]) * jnp.sin(solution_train[:, 2])
x_ddot_train = (driving_force_train - k * solution_train[:, 2] - c * solution_train[:, 3]) / m
theta_ddot_train = (total_tau - mu * solution_train[:, 1]) / I
training_data_full = jnp.column_stack((solution_train[:, 0], solution_train[:, 1], solution_train[:, 2], solution_train[:, 3], theta_ddot_train, x_ddot_train))
print("Full training data shape:", training_data_full.shape)  # (100, 6)

# Neural Network Definition
class DRIVINGFORCEPredictor(nn.Module):
    @nn.compact
    def __call__(self, x):
        angular_position = x[:, 0] / 0.2
        angular_velocity = x[:, 1] / 0.1
        position = x[:, 2] / 2.0 # Get position
        velocity = x[:, 3] / 10.0  # Scale down velocity by 10
        x_scaled = jnp.column_stack((angular_position, angular_velocity, position, velocity))  # Combine position and scaled velocity
        x_scaled = nn.Dense(32)(x_scaled)
        x_scaled = jnp.sin(x_scaled) 
        x_scaled = nn.Dense(16)(x_scaled)
        x_scaled = jax.nn.swish(x_scaled)
        x_scaled = nn.Dense(8)(x_scaled)
        x_scaled = jax.nn.swish(x_scaled)
        return nn.Dense(1)(x_scaled)  # Output driving force

# Initialize model
model = DRIVINGFORCEPredictor()
params = model.init(rng, jnp.ones((1, 4)))

# Optimizer
optimizer = optax.chain(
    optax.clip(1.0),
    optax.adam(learning_rate=1e-4)
)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

# Modified train_step to accept t_eval and use batch initial conditions
@jax.jit
def train_step(state, batch, t_eval):
    def loss_fn(params):
        def pred_equation_of_motion(y, t):
            pred_theta, pred_theta_dot, pred_x, pred_x_dot = y
            total_tau = 0
            inputs = jnp.array([pred_theta, pred_theta_dot, pred_x, pred_x_dot])
            pred_driving_force = state.apply_fn(params, inputs[None, :])  # Shape (1,4)
            pred_theta_ddot = (total_tau - mu * pred_theta_dot) / I
            pred_x_ddot = (pred_driving_force.squeeze() - k * pred_x - c * pred_x_dot) / m
            return jnp.array([pred_theta_dot, pred_theta_ddot, pred_x_dot, pred_x_ddot])

        initial_conditions = batch[0, :4]  # initial pos and vel at t_eval[0]
        pred_solution = odeint(pred_equation_of_motion, initial_conditions, t_eval - t_eval[0], atol=1e-3, rtol=1e-3)
        #Extract x_dot from pred_solution and perform FFT on it. Make FFT into another function outside. # Make sure FFT t_eval here matches the script. Trajectories must be of same size. 
        loss = jnp.mean(((1/0.2)*pred_solution[:, 0] - (1/0.2)*batch[:, 0]) ** 2 + ((1 / 10)*(pred_solution[:, 3]) - (1 / 10)*(batch[:, 3])) ** 2 + ((1/2.0)*pred_solution[:, 2] - (1/2.0)*batch[:, 2]) ** 2 + ((1/0.1)*pred_solution[:, 1] - (1/0.1)*batch[:, 1]) ** 2)
        
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

# Training loop on multiple trajectories
num_trajectories = 17 # Number of trajectories to train on each epoch
for epoch in range(1000):
    epoch_loss = 0.0  # To accumulate loss from multiple trajectories
    for _ in range(num_trajectories):
        t_start_idx = np.random.randint(0, 100)  # Random start point
        interval_length = np.random.randint(5, 8)  # Small interval length
        t_end_idx = t_start_idx + interval_length

        # Make sure we don't exceed the bounds of training data
        if t_end_idx >= 100:
            continue

        state, loss = train_step(state, training_data_full[t_start_idx:t_end_idx], t_eval[t_start_idx:t_end_idx])
        epoch_loss += loss

    # Average loss for the epoch
    avg_loss = epoch_loss / num_trajectories
    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

# Testing parameters (can test on the full interval for visualization)
t_eval_test = jnp.linspace(0, 1, 100)
y0_test = jnp.array([0.1, 0.1, 0.1, 0.1])
solution_test = odeint(equation_of_motion, y0_test, t_eval_test)
test_driving_force = jnp.sin(solution_test[:, 0]) * jnp.sin(solution_test[:, 2])

# Prepare test data
x_ddot_test = (test_driving_force - k * solution_test[:, 2] - c * solution_test[:, 3]) / m
theta_ddot_test = (total_tau - mu * solution_test[:, 1]) / I
test_data = jnp.column_stack((solution_test[:, 0], solution_test[:, 1], solution_test[:, 2], solution_test[:, 3], theta_ddot_test, x_ddot_test))

# Predict driving force using the trained model
pred_driving_force = model.apply(state.params, test_data[:, :4])

plt.figure(figsize=(12, 5))
plt.plot(t_eval_test, test_driving_force, label='Original Driving Force', color='green')
plt.plot(t_eval_test, pred_driving_force, label='Predicted Driving Force', color='orange', linestyle='--')
plt.title('Driving Force vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Driving Force (N)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
