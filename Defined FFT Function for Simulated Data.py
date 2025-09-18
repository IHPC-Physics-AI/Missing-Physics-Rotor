# Simulate Data

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.numpy.fft import fft, fftfreq
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

alpha = 1

# Define the system of ODEs
def equation_of_motion(y, t):
    theta, theta_dot, x, x_dot = y
    driving_force = jnp.sin(x) * jnp.sin(theta)
    total_tau = 0 - alpha*driving_force
    theta_ddot = (total_tau - mu * theta_dot) / I
    x_ddot = (driving_force - k * x - c * x_dot) / m
    return jnp.array([theta_dot, theta_ddot, x_dot, x_ddot])

# Generate full training data (over [0,1] seconds)
t_eval = jnp.linspace(0, 1, 100)  # Adjust this to [0,5] if needed #fs for ground truth is 100. Do same sampling rate as trajectory prediction.
y0_train = jnp.array([0.1, 0.1, 0.1, 0.1])
solution_train = odeint(equation_of_motion, y0_train, t_eval, rtol=1e-3, atol=1e-3)
x_dot_train = solution_train[:, 3]
x_train = solution_train[:, 2]
theta_train = solution_train[:, 0]
theta_dot_train = solution_train[:, 1]

# Define FFT

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

    return freq_bins, disp_spec, t_frames, vel_spec

# Modified FFT amplitude scaling
def correct_amplitude_scaling(disp_spec):
    # Apply physical constraints (max displacement < 1mm)
    disp_mm = disp_spec * 1e3  # Convert to mm
    return jnp.where(disp_mm > 1, 1, disp_mm)  # Cap unrealistic values

freq_bins, disp_spec, t_frames, vel_spec = perform_fft_final(t_eval, x_dot_train, window_size=0.5, overlap=0.8)

# Plot with both simulated and experimental frequencies
fig, ax = plt.subplots(figsize=(12, 6))

# Plot spectrogram
im = ax.pcolormesh(t_frames, freq_bins, disp_spec*1e3,
                  shading='auto', norm=LogNorm(vmin=1e-3), #vmax=10),
                  cmap='viridis')

# Experimental frequencies (from your data)
ax.axhline(8.83, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Exp fnx=8.83Hz')
ax.axhline(17.89, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Exp fny=17.89Hz')
ax.axhline(22.31, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Exp fnz=22.31Hz')


ax.set_ylim(0, 60)  # Show full simulated frequency range
ax.set_xlabel('Time [s]', fontsize=12)
ax.set_ylabel('Frequency [Hz]', fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

cbar = plt.colorbar(im, pad=0.02)
cbar.set_label('Displacement [mm]', rotation=270, labelpad=15)
plt.title('Simulated vs Experimental Natural Frequencies', pad=20)
plt.tight_layout()
plt.show()
