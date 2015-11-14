# Midterm project
## Shiva Barzili
In mathematics and computational science, the Euler method is a numerical procedure for solving ordinary differential equations (ODEs) with a given initial value. It is the most basic explicit method for numerical integration of ordinary differential equations and is the simplest Runge–Kutta method. The Euler method is named after Leonhard Euler, who treated it in his book Institutionum calculi integralis.
The Euler method is a first-order method, which means that the local error (error per step) is proportional to the square of the step size, and the global error (error at a given time) is proportional to the step size. The Euler method often serves as the basis to construct more complex methods.
Runge–Kutta methods are methods for the numerical solution of the ordinary differential equation.
Despite a very noticable similarity between the two methods, the Runge kutta Method uses parabolas(of 2nd order) and quadratic curves( of 4th order) as springboards in order to achieve better approximations.

This project aims at investigating the Euler and the Runge Kutta Methods; two well known techniques for solving systems of differential equations. For, study we have investigated the effects of some parameters, such as the initial conditions, the time step size, the choice of the icrement on the solutions of a given system of differential equations.
Despite that the euler methode being very simple to be implimented, it just provide a very shallow approach to the desired solutions of a system of differential of equation.The Runge kutta a much better approach than the euler method.Decreasing the time step maximes the chance to obtain convergence.It is very difficult to conduct some analyses on the x, y, and z plots due to the chaotic features they display.The choice of the initial conditions should be miticulously done in order to avoid eventual pertubations or chaotic situations.
in my code:
Attractor.ipynb : This file contains the Data Type Attractor, which in its turns contains the default constructor; the rhs, euler, rk2, rk4, save, plotx, ploty, plotz, and plot3d 
test_attractor.py : This file is used to verify a single minimal unit of source of code. It serves for isolating and testing the smallest part of our Attractor class in order to see whether or not the above methods are functioning perfectly in isolation. This help us make sure whether or not the methods listed above operate as they were to function, crash and continue to work on invalide data
Explorer Attractor.ipynb : This file displays the empirical results obtained from

