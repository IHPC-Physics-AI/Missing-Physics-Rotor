# Missing Physics Algorithms

This repository is for the development in modelling a spinning rotor system under a scenario where our understanding of the system is incomplete.

# JAX Conversion of ODE solver for old rotor system

Unable to find rtol, atol that perfectly align with numpy solver (applies to all JAX versions). However, exact alignment is not necessary since it is difficult to attain maximal precision. 

# Implementation of Differential 2-Equations in Code (First Try) #

We no longer have a periodic nature of electrode voltages across time. Therefore, a periodic net torque of the entire rotor system cannot be produced. External driving torque is only induced each instant a rotor arm crosses the laser detector. As a starting simulation, we explore deceleration of the rotor at various rotational frequencies. Assume no net torque here. 

I * theta_ddot + mu * theta_dot = total_tau = 0 (Nm)
Iz = 4.64e-10
Ix = 2.33e-10
Iy = Ix
I = 4.667e-10 
We only consider I as moment of inertia for rotor rotation here. 
theta_ddot = (0-I*(theta_dot))/mu. This line defines equation of motion in code. 
Why mu and not c? Does not matter. Stick with mu as indicated in latest system desciption. 
Initial conditions of theta and theta_dot are inputs of ODE solver. 0.1 each. 

m * x_ddot + c * x_dot + k * x = f(t)
f(t) refers to driving force caused by coupling of modes. It is dependent on rotor frequency and mode vibrations. Causes system to accelerate in mode, decelerate in rotation (due to energy transfer from rotations to vibrations). 
Put x_ddot, x_dot as subjects in equation of motion in code. 
Initial conditions of x and x dot are inputs of ODE solver. 0.1 each.
k = (2*np.pi*fn)**2) * m. 
How do we know what fn is unless we perform Fourier Decomposition? For now, we can explore fn as natural frequency values since they are unique for each mode (uniques Qs and cs). 

First set f = 2 * x 
Plot theta, theta dot, x, x dot against time. t = 30s 
Result: All quantities show straight line across time near 0 with a sharp spike at end of 30s. 

2nd set f = 0.1 * x 
Runtime 22 minutes! 
Result: Rapid fluctuations in x, x_dot and driving force across time. Sharp spikes in theta and theta dot then both go to 0. Possibly due to net torque = 0.

3rd set f = 0.1 * x. Set theta dot (frequency of rotor) as constant: 0.1. 
No difference from 2nd. 

# 2D NN Torque Predictor 

For fixed t, f:
Accurate prediction achieved. 

For fixed t only: 
Parameters explored...
- learning rate 1e-3 most ideal. 
- different activation functions, including sine and cosine within layers
- increase training data to 10,000. No need to run training data and epoch to very high levels. Remember to vary frequency for each theta. 
- increase epoch to 5000
- min-max scaling for f and theta
- linear scaling (1e8) for torque
- logarithmic scaling for torque
Trials & Errors not so successful.
What other modifications to make? Any better methods of trial and error? Can try learning rate scheduler to adapt learning rate. 

# UPDATES: 17/6/2025 #
Tasks in past week: 
1. Achieve accurate 2D NN torque prediction for old rotor system using simulation variables (theta, f)
2. Implement 2-Equation ODE Solver for new system (both Numpy and JAX). Explore possible trends for mode vibrations, forms of driving force f(t) (start with single dependence on x position first). 

## Implementation of Differential 2-Equations in Code (2nd Try) ##
After appropriate changes made to c, Q and fn. 
1st set f = 0.1 * x 
Result: Less rapid fluctuations in x, x_dot and driving force over time with apparent damping. theta and thea_dot still display sharp changes to 0. 

2nd set f = 0.01 * x
Result: More rapid fluctuations of x, x_dot and driving force than 1st. Similar apparent damping over time. theta and theta_dot same trend. Strange observation: theta_0 and theta_dot_0 don't appear to be 0.1 at t = 0s? 

3rd set f = 0.05 * x
Result: Intermediate rapid fluctuations "..."

4th set f = 0.5 * x
Result: ERROR. The simulation results contain infinite or NaN values. This indicates that the issue is not with the plotting code itself, but with the simulation producing values that grow too large or become undefined.

5th set f = 2 * x. Modify code to skip plotting for infinite or NaN values.
Result: Simulation results contain infinite or NaN values. Skipping plotting.

### Next step -- Directly plot experimental data from collaborators to visualise with different forms of f(t), no need to keep just linear in x. Keep track of scenarios where results explode. Change mu. Experimental values won't explode, but may be very noisy. If results explode, means simulation values are unphysical. 

## 2D NN Torque Predictor ##
- NN prediction for theta varied from 0 to 2pi for each frequency in range jnp.linspace(0.5, 8.8, 0.5)
- Only swish activation functions and sinusoidal functions
- learning rate = 1e-4
- Used meshgrid and jnp.stack to assign uniform frequencies for each set of 1000 theta values in NN training.
- Plotted multiple pairs of graphs of torque_true and torque_pred for each frequency
Result: Accurate prediction and same torque_true for all frequencies. Prediction more accurate for manual concatenation of sinusoidal functions compared to including them in hidden layers. 

### Next step -- Convergent history of the training loss. Shows whether training is converging. Check torque computataion, whether remains the same for all frequencies. Relative error (normalizes scaling, computes error wrt to original magnitude, takes away the original scale of the problem) tells error in %, shows how model compares in different length scales. MSE tells error in the original units. Plot ground truth of all frequencies to compare. 

# UPDATES 20/6 - 24/6 25" #
Tasks in Past Week:
- Visualise experimental data for specific mode vibration (figure out which mode it is)
- Explore more variety of driving force forms while still keeping sole dependence on mode position. Attempt to match experimental plot.
- Run collaborators' script to visualise provided experimental data with respect to rotor frequency across time. Compare against results in experiment paper.
- Final improvements to old 2D NN. 

## Implementation of Differential 2-Equations in Code (2nd Try) ##
Plotted of experimental values (x mode displacement against time). Rapid fluctuations with no decay over time. Mode positions between 1.0 and -1.0. Is the torque being applied caused by periodic laser detections of the arms? 
Visually hard to see damping due to noise of high frequencies. No torque in actual experiment. Laser is only used as detection, no voltage involved. 

After appropriate change made to mu (still keep f = 0.01 * x):
Oscillations become more rapid, but same decay trend for driving force, x and x_dot. theta increases linearly. theta_dot decreases linearly. No more sharp changes since mu now matches precision of the system. 

Exploring different forms of f: 
1. f = 0.12<=const<=0.15 * x
   Driving, x, x_dot remain constant near 0, then spike at end of 30s. Can't seem to make much sense of the spiking but logical to see     that quantities are near 0 due to small precision of x, as well as relatively high driving force suppressing vibrations. theta and      theta_dot increase and decrease linearly respectively.
2. f = >=0.16 * x
   Infinite values. Driving force becomes too high relative to x perhaps? Infinities probably come from x, x_dot and f since f purely    dependent on x. 
3. f = x**2
   Driving force fluctuates across positive values (as expected). x_dot fluctuates in similar trend as before and decays. x has larger     fluctuations in positive values. theta and theta_dot increase and decrease linearly respectively (same as before).
4. f = x**3
   Driving force fluctuations decay rapidly. x and x_dot fluctuations decay more gradually and symmetrical about positive & negative       values. Same linear trends for theta and theta_dot.
5. f = sin(x)
   Only positive fluctuation of x. Higher positive fluctuations for x_dot and higher negative fluctuations for driving force.
6. f = cos(x)
   Higher positive fluctuations of driving force. Higher negative fluctuations of x_dot. Only positive fluctuations of x.
8. f = sin(x) + cos(x)
   Positive fluctuations of x. Approximately symmetrical fluctuations in x_dot and driving force. 
10. f = exp(x)
   Infinite values (as expected) 

Attempt at making total_tau positive to mimick non-decay of experimental x:
total_tau = 2
Decay still happens. This is because tau only affects rotation of system. Here, simulated driving force only dependent on x. Clear evidence that x_dot is dependent on theta_dot in actual experiment. 
Need driving frequency (from theta coupling) to match resonant frequency of vibration in order for the decay to be more apparent. Then you get enhanced transfer of energy from 1 mode to the other. 
In experiment, both rotation frequency and vibrational frequency will decay. 
Q factor is degree to which you get amplification of resonant modes. Determines how high the peak goes in real-world. 

## Running Collaborators' Frequency FFT Script on Data ##
Result: Plot of frequency and vibrational displacement against time is exactly the same for omega-z mode of 12mm rotor. Shows that data provided was for omega-z mode. As time passes, frequency decreases along with vibrational displacement. Damping of both theta_dot and x_dot. But if energy is transferred from rotation of rotor to vibration of arms, shouldn't x_dot (and thus x) be increasing? In general, both rotation and vibration will lose energy gradually over time. 
Question: Existence of other significant curves. Noise? 
Clarification: How to interpret FFT codes. 
omega-z mode has jump at 2800s. 
They know that omega-z resonance was not hit. Guessing that energy is transferred from omega to x due to some unknown physical process. 
FFT graph plots amplitudes for all frequencies at each time point. 

## 2D NN (Old System) ##
- Convergent history shows desirable results: Training loss converges to near 0 across epochs.
- Ground truth for torque shows that it is indeed the same across all frequencies. 

Better to plot training scale to log scale. 

## Next Step ## 
- Convert FFT code to JAX code. jnp.fft
- Read paper
