# RK45

Simple RK45 integrator implementation in Python

## usage

Lets use rk45 to simulate the Lorenz Attractor.

The Lorenz Attractor is defined as 

$$\dot{x} = \sigma(y - x)$$

$$\dot{y} = x(\rho - z) - y$$

$$\dot{z} = xy - \beta z$$

to define the system in Python we will return a 3 dimensional vector 
$\langle 
    \dot{x}, 
    \dot{y}, 
    \dot{z} 
\rangle$