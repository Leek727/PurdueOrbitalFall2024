import numpy as np
from numpy import cos, sin
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

# world consts
g = 9.81

# rocket consts
m = 10   

# in rocket body frame, x is towards nosecone
#       ax,ay,az
acceleration = [0,0,0]


# ECI frame
# z is north pole
#           x,y,z
position = [0,0,1]
#          psi,theta,phi -> alpha,beta,gamma
# if rocket on north pole pointing up
# alpha, psi is roll
# beta, theta is yaw
# gamma, phi is pitch
rotation = [0,0,0]

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



def state_update(position, rotation, dt):
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

    return world_acc

# rotation matrix takes current angle of rocket and returns force on rocket in the earth inertial reference frame

#print(state_update([0,0,1], [0,0,0], 0.01))
#          psi,theta,phi -> alpha,beta,gamma
print(body2Inertial([0,np.pi/10,0], [1,0,0]))
#print(body2Inertial([np.pi/6,0,0], [1,0,0]))
#r = R.from_euler('y', -90, degrees=True)
#print(np.matmul(r.as_matrix(), [1,0,0]))