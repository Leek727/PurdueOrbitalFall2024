import numpy as np
from numpy import cos, sin
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from util import *

"""
Initializations
"""
# if rocket on north pole pointing up, pitch roll yaw of rocket
rotation = axis_angle_to_quat([1,0,0], 0)


# world consts
g = 9.81

# rocket consts
m = 10   

# in rocket body frame, x is towards nosecone
# ECI frame
# x is up
#           x,y,z
#position = np.array([0,0,0,  1.0,0.0,0.0])
#velocity = np.array([0,0,0, 0.0,0.0,0.0])
# init quat rotation to one
q1,q2,q3,q4 = axis_angle_to_quat([1, 0, 0], 0).as_quat()
state = np.array([0,0,0,  0,0,0,  q1,q2,q3,q4,  0,0,0])

# new state
# [w1, w2, w3,    vx,vy,vz, q1,    q2,    q3,    q4, x,y,z]
# state deriv
# [dw1, dw2, dw3, ax,ay,az, q1dot, q2dot, q3dot, q4dot,     vx,vy,vz]



def qderiv_from_angular(wx, wy, wz, q):
    """takes w1,w2,w3 -> quaternion derivative"""
    Omega = np.array([
        [0, -wx,-wy,-wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])

    return .5 * np.matmul(Omega, q)


def state_space(t, state):
    w1, w2, w3, vx,vy,vz, q1, q2, q3, q4, x,y,z = state
    rotation = Rotation.from_quat(np.array([q1,q2,q3,q4]))
    state_deriv = np.zeros(9)

    # ---------------------------- translation --------------------------
    # gravity rotated to body frame
    body_acc = np.zeros(3)

    # add thrust
    thrust = 500/m
    if t > 50:
        thrust = 0

    body_acc += np.array([thrust,0,0])
    world_acc = rotation.apply(body_acc)
    world_acc += np.array([-9.8, 0, 0])

    if x < 0:
        vx = 0
        vy = 0 
        vz = 0
        world_acc = np.array([0,0,0])

    # ----------------------------- rotation -----------------------------------
    # [w1, w2, w3,    vx,vy,vz, phi1,phi2,phi3, x,y,z]
    # eulers equations of motion with no external forces
    dw1 = ((I2 - I3) * w2 * w3) / I1
    dw2 = ((I3-I1) * w3 * w1) / I2
    dw3 = ((I1-I2) * w1 * w2) / I3
    
    qdot = qderiv_from_angular(dw1, dw2, dw3, np.array([q1,q2,q3,q4]))


    state_deriv = np.array([dw1, dw2, dw3, 
                            world_acc[0], world_acc[1], world_acc[2],
                            qdot[0], qdot[1], qdot[2], qdot[3],
                            vx, vy, vz
    ])
    # state deriv
    # all rotations are in body frame, all translations in world
    # [dw1, dw2, dw3, ax,ay,az, q1dot, q2dot, q3dot, q4dot,     vx,vy,vz]

    return state_deriv



#print(func(0,np.append(position, rotation)))
sol = RK45(state_space, t0=0, y0=state, t_bound=10000, max_step=1)

t = []
pos = []
while True:
    sol.step()
    t.append(sol.t)
    w1, w2, w3, vx,vy,vz, q1, q2, q3, q4, x,y,z = sol.y

    pos.append([x,y,z])

    if sol.status == "finished":
        break

print(pos[-1])