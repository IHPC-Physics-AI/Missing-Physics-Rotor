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
- Visualise experimental data for specific mode vibration (figure out which mode it is). In reality data is supposed to be decomposed into the 3 modes. 
- Explore more variety of driving force forms while still keeping sole dependence on mode position. Attempt to match experimental plot.
- Run collaborators' script to visualise provided experimental data with respect to rotor frequency across time. Compare against results in experiment paper.
- Final improvements to old 2D NN. 

## Implementation of Differential 2-Equations in Code (2nd Try) ##
Plot of experimental values (x mode displacement against time). Rapid fluctuations with no decay over time. Mode positions between 1.0 and -1.0. Is the torque being applied caused by periodic laser detections of the arms? 
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
Question: Existence of other significant curves. Noise? Probably.
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
- Read paper to understand physics of experiment. 
- Produce NN predictor that calls ODE solver

# UPDATES: 27/6/2025 #
- Successful conversion of FFT script into JAX
- Rough outline of ODE NN Predictor obtained. Still needs more analysis and fixing.

## ODE NN Predictor ##
Key Features modified: 
- Training data are now random values of initial conditions
- Instead of NN predicting a torque function, now it predicts equation of motion in ODE solver. But am I supposed to predict f(x) directly? Yes. First attempt at NN prediction is done wrongly. Need to redo.

Current Stage:
- Prediction is terrible
- Use of sinusoidal activation functions does not help
- Will attempt to scale initial conditions and output variables: theta, theta_dot, x, x_dot.

# UPDATES: 1/7/2025 #
Tasks in past week: 
1. Restructure Neural Network to properly set up training loop for predicting driving_force. Run a few tests to ensure propor structure is achieved before actively improving the NN.
2. Read experimental paper and summarize insights


## Restructured NN training based on feedback from previous meeting: ##
- Simplified to using only 1 equation first (for x only) 
- Generated training data for x, x_dot, x_ddot and driving_force by using a single set of fixed initial conditions. Training data obtained from solutions of ODE.
- Inputs to NN are x, x_dot, x_ddot. 
- Inserted ODE solver for predicted driving force into training loop. Found losses for ODE solution.
- Some issues with shapes
- Result: High numerical instability. Losses are too huge to produce plots.

Try remove x_ddot first. Print values in the loss_fn to check which parameter causes errors. 
In theory, x should not be varying over a large set of values. 
## Insights from Experiment Paper ##
- The electrostatic force arises from capacitive interac- tions between the rotor’s arms and the underlying elec- trodes. As this force is always attractive, a torque is gen- erated when there is asymmetry in the overlapping area. In our levitation setup, high voltage signals ( typically above 100V) are applied to the electrodes (see Fig. 1), creating a spatially varying electric field that induces a redistribution of charges on the conducting rotor. Collaborators only want us to deal with ringdown part. 
- We find that when the rotor approaches 500RPM, the rotor starts to wobble dramatically and ultimately stops upon contacting the magnets. This critical speed 500RPM corresponds to a rotational frequency of 8.33Hz, suggesting a possible resonance with one of the rotor’s rigid body modes. So when rotational frequency (aka driving frequency matches natural frequency of mode, resonance occurs. 
- Beating frequency traces 2Ωz −fX (starting from 7.87 Hz) and fX − Ωz (starting from 0.73 Hz) can also be spotted. What are these? For now no need to worry about this. 
- From FFT plot of 12mm rotor: At 2800s, a sharp drop in spin frequency is observed, coinciding with the spin frequency approaching approxi- mately one-third of the θ mode frequency and one-fourth of the Z mode frequency (see also Fig. 3a). This suggests a mode coupling and significant energy transfer to other modes.
- In contrast, the smaller rotor exhibits a much cleaner spectrum (Fig. 5e), with the dominant frequency com- ponent closely tracking its rotational speed. The higher baseline RPM (> 14Hz) throughout the decay process keeps it away from low-frequency noise and modal cou- pling. Our NN is not dealing with low_frequency noise.

# Updates 4/7/2025 #
## f(x) NN Predictor Progress ##
- Successfully made corrections to 3rd cell in Colab to resolve shaping errors (using both x and x_dot as inputs). Now able to output comparisons between predictions of driving_force.
- Dr Ooi's new code's predictions are much less accurate than my modified version.
- Questions to ask on both versions (commented in script).

Next step: 
- Tuning of NN
- Try smaller time interval of 5s
- Can try to chunk if possible.

# Updates 8/7/2025 # 
Tasks for the past week: 
- Tune properly structured NN to attain best prediction for driving_force
- Attempt training and testing with smaller time interval of 5s.

## f(x) NN Predictor Progress ##
- Found an issue with my version of the NN. The reason why it has 2 driving forces (1 in pred_equation_of_motion and 1 outside) shows that losses calculated have nothing to do with calling the ODE solver. It directly calculates loss of pred_driving_force.
- Made changes to calculate loss due to x and x_dot. In state.apply_fn(), used batch[0, :2], aka initial conditions of x and x_dot as starting point of training driving_force.

Result: Losses were very extreme: 1000+. Very long runtime despite reducing t_eval to 5s. 

### Tuning Actions: ###
- Input Normalization
- Output Standardization 
- Mini-Batching
- Modifying Hidden Layers
- Tuning learning rate

Result: EXTREMELY LONG RUNTIME (ESP AFTER NORMALIZATION & STANDARDIZATION: 40 mins still no output). 

After another attempt at normalization (between 0 and 1): Losses around 0.3+ 

Observation -- Swish activation leads to losses of 1000+

### Attempt at reducing runtime:
- t_eval reduce 5s to 1s
- Introduce float32 to improve JAX speed on most GPUs/TPUs

Result: Loss -- 80000+

## Next Steps ##
- Do normalization inside hidden layers
- Do mini-batching properly: Choose 1 random trajectory first between 0 to 1s. Ensure code accounts for this random initial condition and recognises it accurately. Play around for number of points for NN to predict within that small time interval.
- Then once settled, introduce more random mini trajectories for NN to use.
- Mini-batching is more efficient since it doesn't force NN to predict all time points across the whole major time trajectory.

## Updates 11/7/2025 ##
- Normalization implemented within hidden layers.
- Managed to implement mini batching for single randomly chosen trajectory per batch within 0 to 5s.
- Epochs updates every one time, shows gradual decreasing losses.
- Prediction of driving_force not very accurate, but shape matches the original.
- Attempt at increasing lengths of single trajectory per epoch. Runtime is slow.

## Next Steps ##
- Reduce t_eval to 0 to 1s
- Do 2 to 4 steps for random fixed time interval. Not very generalizable. See whether network can memorise. But easier to see how to improve network. After tuning can start increasing t_eval slightly. 
- Try adding sinusoidal activation functions
- Can try reducing size of network too. Maybe 1 - 2 layers. Decrease number of nodes.
- Set x from -2pi to 2pi. Plot predicted driving_force against x.
- Scaling only ends at NN level. For the sake of better performance in updating weights. But does not affect ACTUAL prediction of x and x_dot. In rest of ODE no scaling occurs. So attempt to implement scaling for x_dot in MSE. To do this must seperate x and x_dot in MSE calculation. 
- After predicted accurately, implement second equation in theta. driving_force now depends on both theta and x. 

## Updates 15/7/2025 ##
Attempts at tuning NN: 
- Reduced t_eval to 0 to 1s for training
- Plotted driving force against x
- Added sinusoidal activation functons. Result: Slightly better prediction but still not accurate
- Use of swish. Result: Not much better in predictions
- Reduced learning rate: Slightly better prediction but only reduce up to 1e-5 since 1e-3 and 1e-4 give nans.
- Explored different number of hidden layers. Result: Not useful
- Applied scaling 1/100 to x_dot in loss calculation. Result: Significantly smaller losses (0.1 to 20+) but not much difference in prediction 

## Next Steps ##
- Do muliple trajectories
- Apply learning rate scheduler
- Reduce tolerance to check if losses are calculating correctly. Change output layer to jnp.sin for checking
- Reduce NN, number of neurons. Structure currently too big.
- Plot convergence
- Plot loss on log scale against time
- Implement 2nd equation and set driving_force as sin(x)sin(theta).

## Updates 18/7/2025 ##
- Successful prediction of f(x) -- Learning rate = 1e-3, no. of trajectories = 12
- Extended t_eval to 5s. 
- Added 2nd equation in theta
- Added hidden layer and swish activation functions: more accurate
- Scaled all variables: theta, theta_dot, x, x_dot
- No. of trajectories = 17, learning rate = 1e-4, time interval = 5 to 8 time steps: moderately accurate.

## Next Steps ##
- Plot "3d" plot. X take min to max. Theta is -pi to pi. Contour plot. 
- Think of another function to try too. See how well trajectory can match. To ensure don't have strange bugs. Try x dot theta.
- voltage is some output. Measures magnitude of velocity of vibration for specific frequency. V is a function of x_dot. One value in csv only accounts for x mode.
- FFT takes a time window to compute fourier transform. Use same time interval of simulation data. Use same number of sampling points per time window as inside the script they. Generate FFT graph from my simulated data.
- FFT will give set of values at different fs per time point.
- For each time you get 1 set of graphs of voltage against frequencies for expt and simulation.
- Choose 1 time point for 1 trajectory for training.
- def FFT part as a function, then call it in training loop. Error needs to be done on FFT output.
- Magnitude of frequency is just some paramter. Take predicted output of FFT and compare to simulated FFT.
- Probably no need scaling as FFT is normalized. Shouldn't see any weird things.
- First thing to figure out is how to make FFT into a single perfect function. 
