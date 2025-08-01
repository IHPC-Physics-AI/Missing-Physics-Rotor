from jax.numpy.fft import fft, fftfreq

def perform_fft_final(time_data, signal_data, window_size=0.5, overlap=0.8):
    # CRITICAL FIX: Proper sampling rate calculation
    fs = 1/(time_data[1] - time_data[0])  # Must use actual time differences
    print(f"Calculated sampling rate: {fs} Hz")  # Should be ~100Hz for your data

    win_N = int(window_size * fs)
    hop_N = int(win_N * (1 - overlap))

    # Window function with energy correction
    win = 0.5 * (1 - jnp.cos(2 * jnp.pi * jnp.arange(win_N) / (win_N - 1)))
    win_gain = jnp.mean(win)  # For amplitude correction

    # FFT processing
    nfft = win_N
    freq_bins = fftfreq(nfft, d=1/fs)[:nfft//2]  # Now with correct fs

    num_windows = (len(signal_data) - win_N) // hop_N + 1
    vel_spec = jnp.zeros((nfft//2, num_windows))
    t_frames = jnp.zeros(num_windows)

    for i in range(num_windows):
        start = i * hop_N
        segment = signal_data[start:start+win_N] * win
        fft_result = fft(segment, n=nfft)[:nfft//2]

        # Correct amplitude scaling
        A_v = (2.0 / (win_N * win_gain)) * jnp.abs(fft_result)
        vel_spec = vel_spec.at[:,i].set(A_v)
        t_frames = t_frames.at[i].set(time_data[start + win_N//2])

    # Convert to displacement
    v2x_scale = jnp.where(freq_bins == 0, jnp.inf, 2 * jnp.pi * freq_bins)
    disp_spec = vel_spec / v2x_scale[:,None]

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

# Define the system of ODEs
def equation_of_motion(y, t):
    theta, theta_dot, x, x_dot = y
    total_tau = 0
    driving_force = jnp.sin(x) * jnp.sin(theta)
    theta_ddot = (total_tau - mu * theta_dot) / I
    x_ddot = (driving_force - k * x - c * x_dot) / m
    return jnp.array([theta_dot, theta_ddot, x_dot, x_ddot])

# Generate full training data (over [0,1] seconds)
t_eval = jnp.linspace(0, 1, 100)  # Adjust this to jnp.linspace(100, 150) for expt data.
y0_train = jnp.array([0.1, 0.1, 0.1, 0.1])
solution_train = odeint(equation_of_motion, y0_train, t_eval, rtol=1e-3, atol=1e-3)
print("Full solution shape:", solution_train.shape)  # (100, 4)

# Calculate true acceleration for reference
driving_force_train = jnp.sin(solution_train[:, 0]) * jnp.sin(solution_train[:, 2])
total_tau_train = 0
x_ddot_train = (driving_force_train - k * solution_train[:, 2] - c * solution_train[:, 3]) / m
theta_ddot_train = (total_tau_train - mu * solution_train[:, 1]) / I
training_data_full = jnp.column_stack((solution_train[:, 0], solution_train[:, 1], solution_train[:, 2], solution_train[:, 3], theta_ddot_train, x_ddot_train))
x_dot_train = solution_train[:, 3]
#print("Full training data shape:", training_data_full.shape)  # (100, 6)

#FFT on Training Data
freq_bins_train, disp_spec_train, t_frames_train, vel_spec_train = perform_fft_final(t_eval, x_dot_train)

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
    optax.clip(0.1),
    optax.adam(learning_rate=1e-5)
)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

@jax.jit
def train_step(state, batch, t_eval_segment, desired_vel_spec_segment):
    def loss_fn(params):
        def pred_equation_of_motion(y, t):
            pred_theta, pred_theta_dot, pred_x, pred_x_dot = y
            total_tau = 0
            inputs = jnp.array([pred_theta, pred_theta_dot, pred_x, pred_x_dot])
            pred_driving_force = state.apply_fn(params, inputs[None, :])
            pred_theta_ddot = (total_tau - mu * pred_theta_dot) / I
            pred_x_ddot = (pred_driving_force.squeeze() - k * pred_x - c * pred_x_dot) / m
            return jnp.array([pred_theta_dot, pred_theta_ddot, pred_x_dot, pred_x_ddot])

        initial_conditions = batch[0, :4]  # Get initial position and velocity
        pred_solution = odeint(pred_equation_of_motion, initial_conditions, t_eval_segment - t_eval_segment[0], atol=1e-5, rtol=1e-3)

        # Extract predicted x_dot
        pred_x_dot = pred_solution[:, 3]

        # Perform FFT on predicted x_dot
        # Pass win_N and hop_N as arguments to perform_fft_final
        freq_bins_pred, disp_spec_pred, t_frames_pred, vel_spec_pred = perform_fft_final(t_eval_segment, pred_x_dot)

        # Calculate the Mean Squared Error loss
        loss = jnp.mean((vel_spec_pred - desired_vel_spec_segment) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

# Training loop on multiple trajectories
# Training loop
num_trajectories = 17  # Number of trajectories for each epoch
for epoch in range(1000):
    epoch_loss = 0.0  # To accumulate loss from multiple trajectories
    for _ in range(num_trajectories):
        t_start_idx = np.random.randint(0, 100)  # Random start point
        interval_length = np.random.randint(20, 40)  # Small interval length
        t_end_idx = t_start_idx + interval_length

        # Ensure we don't exceed the bounds of training data
        if t_end_idx >= 100:
            continue

        # Perform FFT on the segment of the true training data
        # Pass win_N and hop_N to perform_fft_final
        desired_freq_bins_train, desired_disp_spec_train, desired_t_frames_train, desired_vel_spec_segment = perform_fft_final(t_eval[t_start_idx:t_end_idx], x_dot_train[t_start_idx:t_end_idx])

        # Call train_step with the appropriate arguments
        state, loss = train_step(state, training_data_full[t_start_idx:t_end_idx], t_eval[t_start_idx:t_end_idx], desired_vel_spec_segment)
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
total_tau_test = 0

# Prepare test data
x_ddot_test = (test_driving_force - k * solution_test[:, 2] - c * solution_test[:, 3]) / m
theta_ddot_test = (total_tau_test - mu * solution_test[:, 1]) / I
test_data = jnp.column_stack((solution_test[:, 0], solution_test[:, 1], solution_test[:, 2], solution_test[:, 3], theta_ddot_test, x_ddot_test))

# Predict driving force using the trained model
pred_driving_force = model.apply(state.params, test_data[:, :4])

plt.figure(figsize=(12, 5))
plt.plot(t_eval_test, test_driving_force, label='Original Driving Force', color='green')
plt.plot(t_eval_test, pred_driving_force.squeeze(), label='Predicted Driving Force', color='orange', linestyle='--')
plt.title('Driving Force vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Driving Force (N)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

    return freq_bins, disp_spec, t_frames, vel_spec
