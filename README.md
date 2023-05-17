# Exploration of Differentiable Physics

I created this repo to understand and apply differentiable physics in the context of numerical solvers for newtons equation of motion and the navier stokes equations. 

![alt text](img/trajectories_attractors.png)

In this setup a differentibale version of the verlet integration method was used to find an optimal initial position and velocity which moves an object true a force field created by several attractors to a target position.
You can see the evolution of the trajectory over the course of the gradient descent algorithm until it conveges to a near optimal solution.

The solution is not unique as its based on improving an initial guess, if the initial guess is choosen differently the solution will be different.