## Single ODE system for a Spinning Rotor
## Simple NN for fixed t = 20.0s and f = 3.0Hz

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
import matplotlib.pyplot as plt
# Initialize random key
rng = jax.random.PRNGKey(42)

# Constants (replace with your actual values)
R2 = 1.0
d0_squared = 0.1
R_epsilon_A = 0.5
arm_angles = jnp.array([0.0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
electrode_angles = jnp.array([0.0, jnp.pi])

# Torque computation function
def electrode_voltages_precomputed(t, f):
    return 10.0 * jnp.sin(2*jnp.pi*f*t)

def compute_torques_scalar(theta, t, f):
    angle_diff = theta + arm_angles[:, jnp.newaxis] - electrode_angles[jnp.newaxis, :]
    sin_half_angle_diff = jnp.sin(angle_diff / 2)
    distances_squared = d0_squared + (R2 * sin_half_angle_diff)**2
    voltages_squared = electrode_voltages_precomputed(t, f)**2
    torques = R_epsilon_A * voltages_squared / distances_squared * jnp.sign(jnp.sin(angle_diff))
    return jnp.sum(torques)
# =============================================
# Enhanced Scaling (Handles Tiny Values)
# =============================================
def scale_torque(torque):
    """Scale torque to [-1, 1] range before normalization"""
    scale_factor = 1e10  # Adjust based on your torque magnitude
    scaled = torque * scale_factor
    return scaled, scale_factor

# Generate and scale training data
theta_train = jnp.linspace(0, 2*jnp.pi, 1000)
torque_train = jax.vmap(partial(compute_torques_scalar, t=20.0, f=3.0))(theta_train)
torque_train_scaled, torque_scale = scale_torque(torque_train)  # Now in ~[-1,1] range
# Reshape data
theta_train = theta_train.reshape(-1, 1)
torque_train_scaled = torque_train_scaled.reshape(-1, 1)
class TorquePredictor(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Rich periodic encoding applied within layers
        x = nn.Dense(256)(x)

        # Add sinusoidal transformations at this stage
        x = jnp.sin(x) + jnp.cos(x)  # Apply sin and cos to output of first layer. Activation functions changes input.
        x = nn.Dense(128)(x) #2 and 4 updated as weights for new inputs.
        x = nn.swish(x)

        # Third layer with periodic encoding
        x = nn.Dense(64)(x)
        x = nn.swish(x)

        # Final output layer
        return nn.Dense(1)(x)  # Linear output
# Initialize model
model = TorquePredictor()
params = model.init(rng, jnp.ones((1, 1)))

# =============================================
# Robust Training Setup
# =============================================
optimizer = optax.chain( #Combines multiple transformations to apply to the optimisation process in order into a single transformation.
    optax.clip(1.0),  # Gradient clipping to prevent them from exceeding a certain magnitude. Avoid issues of numerical instability caused by exploding gradients.
    optax.adam(learning_rate=1e-4)  # Smaller learning rate
)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)
# Training loop
@jax.jit
def train_step(state, batch):
    theta_batch, torque_batch = batch
    def loss_fn(params):
        pred = state.apply_fn(params, theta_batch)
        return jnp.mean((pred - torque_batch)**2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

for epoch in range(3000):
    state, loss = train_step(state, (theta_train, torque_train_scaled))
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
# =============================================
# Evaluation
# =============================================
theta_test = jnp.linspace(0, 2*jnp.pi, 500).reshape(-1, 1)
torque_true = jax.vmap(partial(compute_torques_scalar, t=20.0, f=3.0))(theta_test.squeeze())

# Predict and rescale
torque_pred_scaled = model.apply(state.params, theta_test)
torque_pred = torque_pred_scaled / torque_scale  # Reverse scaling

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(theta_test, torque_true, label='True Torque', linewidth=2)
plt.plot(theta_test, torque_pred, '--', label='NN Prediction', linewidth=2)
plt.xlabel('Theta (radians)')
plt.ylabel('Torque (Original Scale)')
plt.legend()
plt.grid(True)
plt.title('Improved Torque Prediction')
plt.show()

# Metrics
mse = jnp.mean((torque_true - torque_pred)**2)
print(f"Test MSE: {mse:.3e}")
print(f"Max Error: {jnp.max(jnp.abs(torque_true - torque_pred)):.3e}")

## Simple NN for fixed t = 20.0s ONLY
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
from jax import config
config.update("jax_enable_x64", True) #Critical for tiny values
import matplotlib.pyplot as plt

# Initialize random key
rng = jax.random.PRNGKey(42)

# Constants (replace with your actual values)
R2 = 1.0
d0_squared = 0.1
R_epsilon_A = 0.5
arm_angles = jnp.array([0.0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
electrode_angles = jnp.array([0.0, jnp.pi])

# Torque computation function
def electrode_voltages_precomputed(t, f):
    return 10.0 * jnp.sin(2*jnp.pi*f*t)

def compute_torques_scalar(theta, t, f):
    angle_diff = theta + arm_angles[:, jnp.newaxis] - electrode_angles[jnp.newaxis, :]
    sin_half_angle_diff = jnp.sin(angle_diff / 2)
    distances_squared = d0_squared + (R2 * sin_half_angle_diff)**2
    voltages_squared = electrode_voltages_precomputed(t, f)**2
    torques = R_epsilon_A * voltages_squared / distances_squared * jnp.sign(jnp.sin(angle_diff))
    return jnp.sum(torques)  #if jian cheng's calculation is correct, should be -500 to 500.

# Define a function that takes theta and f and uses the fixed t
def compute_torques(theta, f):
    return compute_torques_scalar(theta, t=20.0, f=f)

# =============================================
# Enhanced Scaling (Handles Tiny Values)
# =============================================
def scale_torque(torque):
    log_torque = jnp.log10(jnp.abs(torque) + 1e-20)  # Adjust based on your torque magnitude
    scaled = log_torque / jnp.max(jnp.abs(jnp.log10(jnp.abs(torque) + 1e-20))) #Normalize to [-1, 1]
    return scaled, None

# Generate the training data for theta
theta_train = jnp.linspace(0, 2 * jnp.pi, 1000)

# Generate the training data for frequency f_train
f_train = jnp.linspace(0.5, 8.5, 1000)

# Use vmap to apply compute_torques over the inputs
torque_train = jax.vmap(compute_torques, in_axes=(0, 0))(theta_train, f_train)

# Scale torque values
torque_train_scaled, torque_scaled = scale_torque(torque_train)

# Reshape data
theta_train = theta_train.reshape(-1, 1)
torque_train_scaled = torque_train_scaled.reshape(-1, 1)
f_train = f_train.reshape(-1, 1)  # Reshape f_train

# =============================================
# Enhanced Model Architecture
# =============================================
# =============================================
# Enhanced Model Architecture
# =============================================
class TorquePredictor(nn.Module):
    @nn.compact
    def __call__(self, theta, f):
        # Rich periodic encoding
        theta_encoded = jnp.concatenate([
            theta,
            jnp.sin(theta), jnp.cos(theta),
            jnp.sin(2*theta), jnp.cos(2*theta),
            jnp.sin(4*theta), jnp.cos(4*theta)
        ], axis=-1)

        # Concatenate the encoded theta with f
        x = jnp.concatenate([theta_encoded, f], axis=-1)

        # Larger network with residual connections
        x = nn.Dense(256)(x)
        x = nn.swish(x) #Swish is better than ReLU for physics problems.
        x = nn.Dense(128)(x)
        x = nn.swish(x)
        x = nn.Dense(64)(x)
        x = nn.swish(x)
        return nn.Dense(1)(x)  # Linear output

# Initialize model
model = TorquePredictor()
params = model.init(rng, jnp.ones((1, 1)), jnp.ones((1, 1)))

# =============================================
# Robust Training Setup
# =============================================
optimizer = optax.chain( #Combines multiple transformations to apply to the optimisation process in order into a single transformation.
    optax.clip(1.0),  # Gradient clipping to prevent them from exceeding a certain magnitude. Avoid issues of numerical instability caused by exploding gradients.
    optax.adam(learning_rate=1e-4)  # Smaller learning rate
)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

# Training loop
@jax.jit
def train_step(state, batch):
    theta_batch, torque_batch, f_batch = batch
    def loss_fn(params):
        pred = state.apply_fn(params, theta_batch, f_batch)
        return jnp.mean((pred - torque_batch)**2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

for epoch in range(3000):
    state, loss = train_step(state, (theta_train, torque_train_scaled, f_train))
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# =============================================
# Evaluation
# =============================================

# Generate theta_test values
theta_test = jnp.linspace(0, 2 * jnp.pi, 10).reshape(-1, 1)
f_test = jnp.linspace(0.5, 3.0, 10).reshape(-1, 1)

# Compute true torque with a specific frequency for comparison
#torque_true = jax.vmap((compute_torques))(theta_test.squeeze(), f_test.squeeze())
# Compute true torque with a specific frequency for comparison
torque_true = jnp.array([compute_torques_scalar(theta, t=20.0, f=f) for theta, f in zip(theta_test.squeeze(), f_test.squeeze())])

# Predict using the trained model, including f_test
max_log = jnp.max(jnp.abs(jnp.log10(jnp.abs(torque_train) + 1e-20)))
torque_pred_scaled = model.apply(state.params, theta_test, f_test)
torque_pred = 10**(torque_pred_scaled*max_log)

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot true torque and predicted torque on the primary y-axis
ax1.plot(theta_test, torque_true, label='True Torque', linewidth=2, color='blue')
ax1.plot(theta_test, torque_pred, '--', label='NN Prediction', linewidth=2, color='orange')
ax1.set_xlabel('Theta (radians)')
ax1.set_ylabel('Torque (Original Scale)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a secondary x-axis for frequency
ax2 = ax1.twiny()  # Instantiate a second axes that shares the same y-axis

# Optional: Set limits for clarity if needed
# ax2.set_xlim(0.5, 8.5)  # Adjust x-limits for frequency if needed

# Add legend for frequency
ax2.legend(loc='upper right')

# Show the title and the plot
plt.title('Improved Torque Prediction with Frequency')
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Metrics
mse = jnp.mean((torque_true - torque_pred)**2)
print(f"Test MSE: {mse:.3e}")
print(f"Max Error: {jnp.max(jnp.abs(torque_true - torque_pred)):.3e}")
