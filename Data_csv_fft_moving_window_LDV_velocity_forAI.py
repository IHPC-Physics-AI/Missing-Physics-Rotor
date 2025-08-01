import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hann
import tkinter as tk
from tkinter import filedialog
from scipy.integrate import cumtrapz          # SciPy’s cumulative trapezoidal rule
from scipy import signal
from matplotlib.colors import LogNorm
import matplotlib as mpl

# -----open and plot time domain signal
# 1.  Pick a CSV file (the dialog title said ".csv", now true to life)
# ---------------------------------------------------------------
def select_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

file_path = select_file()
if not file_path:
    raise SystemExit("No file selected – exiting.")

# ---------------------------------------------------------------
# 2.  Load data, skip header lines, force numeric dtype
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
#     5 mm s‑1 per volt  ×  1 m / 1000 mm  = 0.005 m s‑1 V‑1
# ---------------------------------------------------------------
mm_s_per_V = 10  #5mm/s/V
velocity_scale = mm_s_per_V * 1e-3             # → m s‑1 V‑1  = 0.005
velocity_data  = signal_data_raw * velocity_scale

# Plot x_data_raw vs. y_data_raw
plt.figure(figsize=(10, 6))
plt.plot(time_data_raw, velocity_data, label="Data", marker='', color='b')
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Velocity (m/s)", fontsize=16)
# plt.title("Plot of X Data vs Y Data")
# plt.legend()
# plt.grid(True)
# Make the outer box (spines) thicker
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
# Set font size for x and y tick labels
plt.xticks(fontsize=16)  # X-axis font size
plt.yticks(fontsize=16)  # Y-axis font size
# Save as a high-resolution image for publication
# plt.savefig("plot.png", dpi=300, bbox_inches="tight", transparent=True)
plt.show()




#%% do fft-->convert from velocity to displacement

# ---------------------------------------------------------------
# 1)  Zoom the data you want to analyse   ①
# ---------------------------------------------------------------
t_min, t_max = -2000, 3600       # seconds  ← change if needed

time_raw = np.asarray(time_data_raw)      # <-- your raw time stamps
vel_raw  = np.asarray(velocity_data)      # <-- velocity in m/s

mask          = (time_raw >= t_min) & (time_raw <= t_max)
time_data     = time_raw[mask]            # zoomed trace
time_data    -= time_data[0]              # t = 0 at start
signal_data   = vel_raw[mask] - vel_raw[mask].mean()   # remove DC

# ---------------------------------------------------------------
# 2)  Rolling-FFT parameters          ②
# ---------------------------------------------------------------
window_size = 100        # seconds per FFT frame
overlap     = 0.99        # 95 % overlap  (hop = 0.05 * window)
peak_band   = (10, 19)    # Hz band in which to track the peak

# ---------------------------------------------------------------
# 3)  Derived constants (no edits below this block)
# ---------------------------------------------------------------
fs          = 1.0 / np.diff(time_data[:2])[0]        # sample rate
win_N       = int(window_size * fs)                  # samples / frame
hop_N       = int(win_N * (1 - overlap))             # hop size
nfft        = win_N                                  # FFT length
win         = hann(win_N, sym=False)                 # Hann taper

freq_bins   = fftfreq(nfft, d=1/fs)[:nfft//2]        # positive freqs
v2x_scale   = np.where(freq_bins == 0,
                       np.inf,                       # avoid /0
                       2*np.pi*freq_bins)

# Containers for outputs
vel_spec  = []        # spectra → later stacked (N_freq × N_time)
t_frames  = []        # centre time of every frame
peak_disp = []        # peak displacement (m)
peak_vel  = []        # peak velocity (m/s)
peak_freq = []        # frequency of that peak (Hz)

# ---------------------------------------------------------------
# 4)  Rolling-FFT loop
# ---------------------------------------------------------------
start = 0
while start + win_N <= len(signal_data):
    seg = signal_data[start:start+win_N] * win
    fft_v = fft(seg, n=nfft)[:nfft//2]
    A_v = (2.0 / win_N) * np.abs(fft_v)              # velocity amp (m/s)

    vel_spec.append(A_v)
    t_frames.append(time_data[start + win_N//2])     # frame centre time

    # --- locate strongest bin in chosen band --------------------
    band = (freq_bins >= peak_band[0]) & (freq_bins <= peak_band[1])
    if np.any(band):
        idx_rel  = np.argmax(A_v[band])               # index inside band
        idx_full = np.where(band)[0][idx_rel]         # index in full array

        v_peak   = A_v[idx_full]                      # velocity
        x_peak   = v_peak / v2x_scale[idx_full]       # displacement
        f_peak   = freq_bins[idx_full]                # frequency

        peak_vel.append(v_peak)
        peak_disp.append(x_peak)
        peak_freq.append(f_peak)
    else:
        peak_vel .append(0.0)
        peak_disp.append(0.0)
        peak_freq.append(np.nan)

    start += hop_N                                    # slide window

# ---------------------------------------------------------------
# 5)  Stack spectra & convert to displacement
# ---------------------------------------------------------------
vel_spec  = np.array(vel_spec).T                     # (N_freq, N_time)
disp_spec = vel_spec / v2x_scale[:, None]            # m

# # ---------------------------------------------------------------
# # 6)  Spectrogram (displacement)
# # ---------------------------------------------------------------
f_min, f_max = 0.2, 10                                 # plot band
f_mask = (freq_bins >= f_min) & (freq_bins <= f_max)

# plt.figure(figsize=(14,6))
# plt.pcolormesh(t_frames,
#                 freq_bins[f_mask],
#                 disp_spec[f_mask]*1e3,                # mm
#                 shading='auto', cmap='jet')
# plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
# plt.colorbar(label='Amplitude (mm)')
# plt.title('Rolling-FFT displacement map')
# plt.tight_layout(); plt.show()

# Ensure values are strictly positive before applying log10
# amplitude = disp_spec[f_mask] * 1e3  # Convert to mm
# amplitude[amplitude <= 1e-5] = 1e-5  # Avoid log of zero or negative

# log_amplitude = np.log10(amplitude)

# plt.figure(figsize=(14, 6))
# plt.pcolormesh(t_frames,
#                freq_bins[f_mask],
#                log_amplitude,
#                shading='auto', cmap='gray')

# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.colorbar(label='Log₁₀ Amplitude (mm)')
# plt.title('Rolling-FFT displacement map (log scale)')
# plt.tight_layout()
# plt.show()

#%% plot spectrum vs time; log scale
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator

# ------------------ publishing defaults ------------------ #
mpl.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "figure.dpi": 300,
})

# ------------------ custom blue–white–orange map  🆕 ------------------ #
blue_white_orange = LinearSegmentedColormap.from_list(
    "blue_white_orange",
    [(0.00, "#2D5891"),     # deep blue
     (0.6, "#F2BC6E"),    # orange-gold
     (1.00, "#FFF2BE")],     # white centre
    N=256
)

# ------------------ data selection ------------------ #
f_min, f_max = 0.2, 20
mask     = (freq_bins >= f_min) & (freq_bins <= f_max)
disp_mm  = disp_spec[mask] * 1e3
disp_mm[disp_mm <= 1e-10] = 1e-10           # avoid log(0)

# ------------------ plot ------------------ #
fig, ax = plt.subplots(figsize=(3.6, 2.0))

pcm = ax.pcolormesh(
    t_frames,
    freq_bins[mask],
    disp_mm,
    shading="auto",
    norm=LogNorm(vmin=1e-5, vmax=disp_mm.max()),
    # norm=LogNorm(vmin=1e-5, vmax=10e-3),
    cmap=blue_white_orange                       # 🆕 use custom map
)

# axes & ticks
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_ylim(f_min, f_max)
ax.tick_params(direction="out")

# ------------------ normal colour-bar ------------------ #
cbar = fig.colorbar(pcm, ax=ax, pad=0.02)    # default width / aspect
cbar.minorticks_off()                        # <- remove minor ticks

cbar.set_label("Displacement  [mm]")         # label in normal font / colour


# ------------------ final layout ------------------ #
fig.tight_layout(pad=0.2)
plt.show()
