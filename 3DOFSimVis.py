import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from util import *

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up plot limits
scale = 10000
ax.set_xlim([-scale, scale])
ax.set_ylim([-scale, scale])
ax.set_zlim([0, scale])

# Initialize empty line
line, = ax.plot([], [], [], lw=2)


# -------------------------------------------------- INIT --------------------------------------------------
# if rocket on north pole pointing up, pitch roll yaw of rocket
# y psi
# p theta
# r phi

# initial rotation of rocket body
rotation = axis_angle_to_quat([1,.2,0], np.pi/2)

#print(getSkinCF(R, L, Rs))


def state_space(t, position):
    v = position[0:3]
    position = position[3:6]

    # gravity rotated to body frame
    # ax = Gx + T/m
    # ay = Gy
    # az = Gz
    position = np.array(position)
    body_acc = np.zeros(3)

    # add thrust
    thrust = 500/m
    if t > 50:
        thrust = 0
    
    body_acc += np.array([thrust,0,0])

    # rotate acceleration to world inertial frame
    #world_acc = body2Inertial(rotation, net_force_body)
    world_acc = rotation.apply(body_acc)
    world_acc += np.array([-g,0,0]) # gravity

    # check ground collision
    if position[0] < 0:
        return np.zeros((1,6))


    return np.append(world_acc, v)

# world consts
g = 9.81

# rocket consts
m = 10   

# in rocket body frame, x is towards nosecone
#       ax,ay,az
acceleration = [0,0,0]
dt = .0001


# ECI frame
# x is up
#           x,y,z
position = np.array([1.0,0.0,0.0])
velocity = np.array([0.0,0.0,0.0])

# state 
# [vx,vy,vz,x,y,z]
# state deriv
# [ax,ay,az,vx,vy,vz]


#print(func(0,np.append(position, rotation)))
sol = RK45(state_space, t0=0, y0=np.append(velocity, position), t_bound=1000, max_step=1)

t = []
pos = []
while True:
    sol.step()
    t.append(sol.t)
    pos.append(sol.y[3:6])

    if sol.status == "finished":
        break

print(f"Len of pos: {len(pos)}")
print(f"Last time : {t[-1]}")


# Data to store the points
x_data, y_data, z_data = [], [], []


# Update function for the animation
def update(num):
    x_data.append(pos[num][1])
    y_data.append(pos[num][2])
    z_data.append(pos[num][0])

    

    # Update the line with new data
    line.set_data(x_data, y_data)
    line.set_3d_properties(z_data)


    lims = list(ax.get_xlim()) + list(ax.get_ylim()) + list(ax.get_zlim())
    lims += [min(x_data), max(x_data), min(y_data), max(y_data), min(z_data), max(z_data)]

    a = abs(min(lims))
    if max(lims) > a:
        a = max(lims)

    ax.set_xlim(-a,a)
    ax.set_ylim(-a,a)
    ax.set_zlim(0,a)



    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, len(pos), 1), interval=1, repeat=False)

# Show the plot
plt.show()