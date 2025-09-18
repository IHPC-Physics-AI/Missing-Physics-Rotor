##JAX VERSION##

import pandas as pd
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import lax
from jax.numpy.fft import fft, fftfreq
from matplotlib.colors import LogNorm
import matplotlib as mpl

# -----open and plot time domain signal
# 1.  Define the CSV file path
file_path = './20250521-520RPM-ringdown-1hour-scope.csv'  # Adjust as needed

# ---------------------------------------------------------------
# 2.  Load data from the uploaded file content, skip header lines, force numeric dtype
# ---------------------------------------------------------------
df = pd.read_csv(
    file_path,
    skiprows=3,
    header=None,
    names=["time", "voltage"],
    na_values=["-∞", "∞", "-inf", "inf", "-Inf", "Inf",
               "Infinity", "-Infinity",  # plus the usuals
               "nan", "NaN", ""],
    keep_default_na=True
)
# Drop rows that failed numeric conversion
df = df.dropna(subset=["time", "voltage"])

time_data_raw    = df["time"].to_numpy()       # seconds
signal_data_raw  = df["voltage"].to_numpy()    # V  (velocity‑decoder output)

# ---------------------------------------------------------------
# 3.  Convert decoder output (V) → velocity (m/s)
mm_s_per_V = 10  # 10 mm/s/V
velocity_scale = mm_s_per_V * 1e-3             # → m s‑1 V‑1  = 0.01
velocity_data  = signal_data_raw * velocity_scale

# Plot velocity data
plt.figure(figsize=(10, 6))
plt.plot(time_data_raw, velocity_data, label="Data", marker='', color='b')
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Velocity (m/s)", fontsize=16)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16)  
plt.show()

#%% Perform FFT and convert from velocity to displacement

# ---------------------------------------------------------------
# 1)  Zoom the data you want to analyze
# ---------------------------------------------------------------
t_min, t_max = -2000, 3600       # seconds  ← change if needed

time_raw = jnp.asarray(time_data_raw)      # <-- your raw time stamps
vel_raw  = jnp.asarray(velocity_data)      # <-- velocity in m/s

mask          = (time_raw >= t_min) & (time_raw <= t_max)
time_data     = time_raw[mask]            # zoomed trace
time_data    -= time_data[0]              # t = 0 at start
signal_data   = vel_raw[mask] - vel_raw[mask].mean()   # remove DC

# ---------------------------------------------------------------
# 2)  Rolling-FFT parameters
# ---------------------------------------------------------------
window_size = 100        # seconds per FFT frame
overlap     = 0.99       # 99% overlap
peak_band   = (10, 19)   # Hz band in which to track the peak

# ---------------------------------------------------------------
# 3)  Derived constants (no edits below this block)
# ---------------------------------------------------------------
fs          = 1.0 / jnp.diff(time_data[:2])[0]  # sample rate
win_N       = int(window_size * fs)               # samples / frame
hop_N       = int(win_N * (1 - overlap))          # hop size
nfft        = win_N                                 # FFT length

# Manual implementation of the Hann window
def hann_window(length):
    return 0.5 * (1 - jnp.cos(2 * jnp.pi * jnp.arange(length) / (length - 1)))

win         = hann_window(win_N)                    # Hann taper

freq_bins   = fftfreq(nfft, d=1/fs)[:nfft//2]     # positive freqs
v2x_scale   = jnp.where(freq_bins == 0,
                         jnp.inf,                        # avoid /0
                         2 * jnp.pi * freq_bins)

# Containers for outputs
vel_spec  = []       # spectra → later stacked (N_freq × N_time)
t_frames  = []       # center time of every frame
peak_disp = []       # peak displacement (m)
peak_vel  = []       # peak velocity (m/s)
peak_freq = []       # frequency of that peak (Hz)

# ---------------------------------------------------------------
# 4)  Rolling-FFT loop
# ---------------------------------------------------------------
start = 0
while start + win_N <= len(signal_data):
    seg = signal_data[start:start + win_N] * win
    fft_v = fft(seg, n=nfft)[:nfft//2]
    A_v = (2.0 / win_N) * jnp.abs(fft_v)              # velocity amp (m/s)

    vel_spec.append(A_v)
    t_frames.append(time_data[start + win_N // 2])     # frame center time

    # --- locate strongest bin in chosen band --------------------
    band = (freq_bins >= peak_band[0]) & (freq_bins <= peak_band[1])
    if jnp.any(band):
        idx_rel  = jnp.argmax(A_v[band])               # index inside band
        idx_full = jnp.where(band)[0][idx_rel]         # index in full array

        v_peak   = A_v[idx_full]                      # velocity
        x_peak   = v_peak / v2x_scale[idx_full]       # displacement
        f_peak   = freq_bins[idx_full]                # frequency

        peak_vel.append(v_peak)
        peak_disp.append(x_peak)
        peak_freq.append(f_peak)
    else:
        peak_vel.append(0.0)
        peak_disp.append(0.0)
        peak_freq.append(jnp.nan)

    start += hop_N                                    # slide window

# ---------------------------------------------------------------
# 5)  Stack spectra & convert to displacement
# ---------------------------------------------------------------
vel_spec  = jnp.array(vel_spec).T                     # (N_freq, N_time)
disp_spec = vel_spec / v2x_scale[:, None]            # m

# %% Plot spectrum vs time; log scale
# ---------------------------------------------------------------
# Configure matplotlib for plotting
mpl.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "figure.dpi": 300,
})

# Custom blue–white–orange colormap
blue_white_orange = plt.cm.get_cmap("cividis")

# Data selection for plotting
f_min, f_max = 0.2, 20
mask      = (freq_bins >= f_min) & (freq_bins <= f_max)
disp_mm   = disp_spec[mask] * 1e3
disp_mm   = jnp.where(disp_mm <= 1e-10, 1e-10, disp_mm)  # Avoid log(0)

# Plotting
fig, ax = plt.subplots(figsize=(3.6, 2.0))

pcm = ax.pcolormesh(
    t_frames,
    freq_bins[mask],
    disp_mm,
    shading="auto",
    norm=LogNorm(vmin=1e-5, vmax=jnp.max(disp_mm)),
    cmap=blue_white_orange                       # Use custom map
)

# Set axes and ticks
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_ylim(f_min, f_max)
ax.tick_params(direction="out")

# Normal color-bar
cbar = fig.colorbar(pcm, ax=ax, pad=0.02)    # Default width / aspect
cbar.minorticks_off()                        # Remove minor ticks
cbar.set_label("Displacement [mm]")         # Label in normal font / color

# Final layout
fig.tight_layout(pad=0.2)
plt.show()
