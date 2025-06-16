# Missing Physics Algorithms

This repository is for the development in modelling a spinning rotor system under a scenario where our understanding of the system is incomplete.

#JAX Conversion of ODE solver for old rotor system

Unable to find rtol, atol that perfectly align with numpy solver (applies to all JAX versions). However, exact alignment is not necessary since it is difficult to attain maximal precision. 

# Implementation of Differential Equations in Code (First Try) 

We no longer have a periodic nature of electrode voltages across time. Therefore, a periodic net torque of the entire rotor system cannot be produced. External driving torque is only induced each instant a rotor arm crosses the laser detector. As a starting simulation, we explore deceleration of the rotor at various rotational frequencies. Assume no net torque here. 

I * theta_ddot + mu * theta_dot = tau = 0 (Nm)
Iz = 4.64e-10
Ix = 2.33e-10
Iy = Ix
I = 4.667e-10 
We only consider I as moment of inertia for rotor rotation here. 
theta_ddot = (0-I*(theta_dot))/mu. This line defines equation of motion in code. 
Why mu and not c? Does not matter. Stick with mu as indicated in latest system desciption. 
Initial conditions of theta and theta dot are inputs of ODE solver. 0.1 each. 

m * x_ddot + c * x_dot + k * x = f(t)
f(t) refers to driving force caused by coupling of modes. It is dependent on rotor frequency and mode vibrations. Causes system to accelerate in mode, decelerate in rotation (due to energy transfer from rotations to vibrations). 
Put x_ddot, x_dot as subjects in equation of motion in code. 
Initial conditions of x and x dot are inputs of ODE solver. 0.1 each.
k = (2pi * fn)^2 * m. 
How do we know what fn is unless we perform Fourier Decomposition? For now, we can explore any values for fn. 

First set f = 2 * x 
Plot theta, theta dot, x, x dot against time. t = 30s 
Result: All quantities show straight line across time near 0 with a sharp spike at end of 30s. 

2nd set f = 0.1 * x 
Runtime 22 minutes! 
Result: Rapid fluctuations in x, x dot and driving force across time. Sharp spikes in theta and theta dot then both go to 0. Possibly due to net torque = 0.

3rd set f = 0.1 * x. Set theta dot (frequency of rotor) as constant: 0.1. 
No difference from 2nd. 

#NN Torque Predictor 

For fixed t, f:
Accurate prediction achieved. 

For fixed t only: 
Parameters explored...
- learning rate 1e-3 most ideal. Can try learning rate scheduler to adapt learning rate. 
- different activation functions, including sine and cosine within layers
- increase training data to 10,000. No need to run training data and epoch to very high levels. Remember to vary frequency for each theta. 
- increase epoch to 5000
- min-max scaling for f and theta
- linear scaling (1e8) for torque
- logarithmic scaling for torque
Trials & Errors not so successful.
What other modifications to make? Any better methods of trial and error?  

