import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import cos, sin
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
from scipy.integrate import RK45
import matplotlib.pyplot as plt


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

rotation = [np.pi/100,np.pi/100,0]

# skin friction coefficient
Cf = 1 / (1.5 * np.log())

def body2Inertial(rotation, x):
    """Rotate vector x to inertial frame given rotation"""
    psi, theta, phi = rotation
    rotation_mat = np.array(
        [
            [cos(psi)*cos(theta), cos(theta)*sin(psi), -sin(theta)],
            [cos(psi)*sin(phi)*sin(theta)-cos(phi)*sin(psi), cos(phi)*cos(psi)+sin(phi)*sin(psi)*sin(theta), cos(theta)*sin(phi)],
            [sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta), cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi), cos(phi)*cos(theta)]
        ]
    )

    return np.matmul(rotation_mat,x)

def inertial2Body(rotation, x):
    """Given angle of rocket and force in world reference frame, rotate force into body reference frame"""
    # inverse of body2Inertial mat
    psi, theta, phi = rotation  
    world_mat = np.array(
        [
            [cos(psi)*cos(theta), cos(theta)*sin(psi), -sin(theta)],
            [cos(psi)*sin(phi)*sin(theta)-cos(phi)*sin(psi), cos(phi)*cos(psi)+sin(phi)*sin(psi)*sin(theta), cos(theta)*sin(phi)],
            [sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta), cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi), cos(phi)*cos(theta)]
        ]
    )
    body_mat = inv(world_mat)

    return np.matmul(body_mat, x)

def state_space(t, position):
    v = position[0:3]
    position = position[3:6]

    # gravity rotated to body frame
    # ax = Gx + T/m
    # ay = Gy
    # az = Gz
    position = np.array(position)
    net_force_body = np.zeros(3)

    # rotate gravity to body frame
    # unit vec from center to position of rocket
    #u = position / np.sqrt(position.dot(position))
    #G = g * (-u)
    #G_body = inertial2Body(rotation,)
    #print(G_body)
    #net_force_body += G_body

    # add thrust
    thrust = 500/m
    if t > 50:
        thrust = 0
    
    net_force_body += np.array([thrust,0,0])

    # rotate acceleration to world inertial frame
    world_acc = body2Inertial(rotation, net_force_body)
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
# z is north pole
#           x,y,z
position = np.array([1.0,0.0,0.0])
velocity = np.array([0.0,0.0,0.0])

# state 
# [vx,vy,vz,x,y,z]
# state deriv
# [ax,ay,az,vx,vy,vz]


#print(func(0,np.append(position, rotation)))
sol = RK45(state_space, t0=0, y0=np.append(velocity, position), t_bound=1000)

t = []
pos = []
for i in range(100):
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