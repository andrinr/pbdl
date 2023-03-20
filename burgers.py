from phi.flow import *
import numpy as np

N = 128
dx = 2/N
steps = 32
dt = 1/steps
nu = 0.01/(N*np.pi) # viscosity

initial = np.asarray( 
    [-np.sin(np.pi * x) for x in np.linspace(-1+dx/2,1-dx/2,N)] 
) # 1D numpy array

initial = math.tensor(initial, spatial('x') ) # convert to phiflow tensor

velocity = CenteredGrid(initial, extrapolation.PERIODIC, x=N, bounds=Box(x=(-1,1)))
vt = advect.semi_lagrangian(velocity, velocity, dt)

print("Velocity tensor shape: "   + format( velocity.shape )) # == velocity.values.shape
print("Velocity tensor type: "    + format( type(velocity.values) ))
print("Velocity tensor entries 10 to 14: " + format( velocity.values.numpy('x')[10:15] ))

velocities = [velocity]
age = 0.
for i in range(steps):
    v1 = diffuse.explicit(velocities[-1], nu, dt) # diffuse
    v2 = advect.semi_lagrangian(v1, v1, dt) # advect with itself
    age += dt
    velocities.append(v2)

print("New velocity content at t={}: {}".format( age, velocities[-1].values.numpy('x,vector')[0:5] ))

vels = [v.values.numpy('x,vector') for v in velocities] # gives a list of 2D arrays 

import pylab

fig = pylab.figure().gca()
fig.plot(np.linspace(-1,1,len(vels[ 0].flatten())), vels[ 0].flatten(), lw=2, color='blue',  label="t=0")
fig.plot(np.linspace(-1,1,len(vels[10].flatten())), vels[10].flatten(), lw=2, color='green', label="t=0.3125")
fig.plot(np.linspace(-1,1,len(vels[20].flatten())), vels[20].flatten(), lw=2, color='cyan',  label="t=0.625")
fig.plot(np.linspace(-1,1,len(vels[32].flatten())), vels[32].flatten(), lw=2, color='purple',label="t=1")
pylab.xlabel('x'); pylab.ylabel('u'); pylab.legend()

pylab.show()