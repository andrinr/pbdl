import torch
import matplotlib.pyplot as plt
from typing import Callable

def macCormack2D(u : torch.Tensor, t : float , flux : Callable , dt : float , dx : float):

    up = u.clone()
    up[:-1] = u[:-1] - (dt/dx) * (flux(u[1:]) - flux(u[:-1]))
    u[1:] = 0.5 * (u[1:] + up[1:]) - 0.5 * (dt/dx) * (flux(up[1:]) - flux(up[:-1]))
    return u[1:-1]

def macCormack3D(u : torch.Tensor, t : float , flux : Callable , dt : float , dx : float):

    up = u.clone()
    up[:-1, :] = u[:-1, :] - (dt/dx) * (flux(u[1:, :]) - flux(u[:-1, :]))
    u[1:, :] = 0.5 * (u[1:, :] + up[1:, :]) - 0.5 * (dt/dx) * (flux(up[1:, :]) - flux(up[:-1, :]))

    up = u.clone()
    up[:, :-1] = u[:, :-1] - (dt/dx) * (flux(u[:, 1:]) - flux(u[:, :-1]))
    u[:, 1:] = 0.5 * (u[:, 1:] + up[:, 1:]) - 0.5 * (dt/dx) * (flux(up[:, 1:]) - flux(up[:, :-1]))

    return u[1:-1, 1:-1]

n = 1000

flux = lambda u : 0.5 * u**2.0
dt = 0.1
dx = 0.1

u = torch.zeros(n+2)
u = torch.sin(torch.linspace(0, 2*3.1415, n+2))

us = []
steps = 200
for i in range(steps):
    u[1:-1] = macCormack2D(u, i, flux, dt, dx)
    us.append(u.clone())

n_plots = 5

fig, ax = plt.subplots()

for i in range(0, steps, steps//n_plots):
    ax.plot(us[i], label=f"t = {i}")

ax.legend()

plt.show()