# Physics based deep learning

## Introduction
We differentiate between:
- *Superbised* Learning
- *Loss Term* Physiscs informed learning, where the loss is modeled after a physical system
- *Interleaved* TODO

Generally we to approximate an unknown function $f^*$ which models the relationship between $x$ and the ground truth $y^*$:

$$ f(x)^* = y^* $$

We have a loss function $L(y, y^*)$ which is used to find the paramteres $\theta$ of $NN$ network which computes $f(x, \theta) = y$. $\theta$ can be found by minimizing the loss function. 

### Nabla Operators

For a vector field $u$ and a scalar field $\phi$ we have the following:

| Name        | Operator           | From        | To          | Example               
| ----------- | -------------------| ----------- | ----------- | ----------------------
| Gradient    | $\nabla$           | $\phi$      | $u$         | Slope of a height map 
| Divergence  | $\nabla \cdot$     | $u$         | $\phi$      | In and outflow of cell
| Curl        | $\nabla \times$    | $u$         | $u$         | Rotation of a vector field 
| Laplacian   | $\nabla \cdot \nabla$ | $\phi$   | $\phi$      | Curvature        


### Burgers PDE

Advection / diffusion term in dimension $i$:

$$ \frac{\partial u_i}{\partial t} + u \cdot \nabla u_i =  \nu \nabla \cdot \nabla u  + g_i$$

where $u$ is the velocity field, $\nu$ is the diffusion constant and $g_i$ is the external force (e.g. gravity) in dimension $i$.

### Navier-Stokes PDE

$$ \frac{\partial u_i}{\partial t} + u \cdot \nabla u_i = - \frac{1}{\rho}\nabla p + \nu \nabla \cdot \nabla u_i $$

where we have the non compressible constraint:

$$ \nabla \cdot u = 0 $$

where $u$ is the velocity field, $\nu$ is the diffusion constant, $p$ is the pressure, $\rho$ is the density and $g_i$ is the external force (e.g. gravity) in dimension $i$.

### Classical Optimizers

Given a scalar loss function $L(x) : \mathbb{R}^n \rightarrow \mathbb{R}$ with an optimum (minimum) at $x^*$ and $\Delta$ a step in $x$. Given two matrices $A$ and $B$ we define an inversion:

$$ \frac{A}{B} = B^{-1} A $$

Given two vectors $a, b$ we define the inversions:

$$ \frac{a}{b} = \frac{aa^T}{a^Tb}$$

We get the Yacobian of $L$ at $x$ $J(x) = \frac{\partial L}{\partial x}$. We get the Hessian by differentiating the Jacobian:

$$ H(x) = \frac{\partial J}{\partial x} = \frac{\partial^2 L}{\partial x^2} $$

Talyor expansion of $L$ around $x$:

$$ L(x + \Delta) = L(x) + \Delta J(x) + \frac{1}{2} \Delta^2 H(x) + ... $$

The Lagrange Form:

$$ L(x + \Delta) = L(x) + J(x)\Delta + \frac{1}{2} H$$