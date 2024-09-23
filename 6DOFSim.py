import numpy as np
from numpy import cos, sin
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
from scipy.integrate import RK45
import matplotlib.pyplot as plt

"""
Initializations
"""
# if rocket on north pole pointing up, pitch roll yaw of rocket
# p theta
# r phi
# y psi
rotation = [0,0,0]

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
    u = position / np.sqrt(position.dot(position))
    G = g * (-u)
    G_body = inertial2Body(rotation,G)
    net_force_body += G_body

    # add thrust
    net_force_body += np.array([100/m,0,0])

    # rotate acceleration to world inertial frame
    world_acc = body2Inertial(rotation, net_force_body)


    return np.append(world_acc, v)





#print(func(0,np.append(position, rotation)))
sol = RK45(state_space, t0=0, y0=np.append(velocity, position), t_bound=10, max_step=1)

t = []
pos = []
for i in range(100):
    sol.step()
    t.append(sol.t)
    pos.append(sol.y[3:6])

    if sol.status == "finished":
        break

print(pos)
print(state_space(0, np.append(velocity, position)))



"""Functions"""
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
