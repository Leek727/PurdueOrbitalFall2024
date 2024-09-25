import numpy as np
from numpy import cos, sin
from constants import *
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

# utility functions
def getSkinCd(R: float, v: float) -> float:
    """Reynolds num, abs velocity of rocket -> skin friction drag coefficient"""
    characteristic_len = 2*radius 
    R_crit = 51 * (Rs / characteristic_len) ** (-1.039)

    # skin friction coefficient for turbulent flow
    Cf = 1.48 * 10 ** (-2)
    if R > 10**4 and R < R_crit:
        Cf = 1 / ((1.5 * np.log(R) - 5.6) ** 2)

    elif R > R_crit:
        Cf = 0.032 * (Rs / characteristic_len) ** 0.2

    # apply compressibility corrections
    M = v / 343  # mach number
    if M < 1:  # subsonic
        Cf = Cf(1 - 0.1 * M**2)

    elif M >= 1:
        # check if "roughness-limited" or above R_crit
        if R > R_crit:
            Cf = Cf / (1 + 0.15 * M**2) ** 0.58
        else:
            Cf = Cf / (1 + 0.18 * M**2)

    # get actual drag coefficient
    fB = body_length / diameter  # fineness ratio / how slim it is
    Cd = Cf * ((1 + 1 / (2 * fB) * A_wet_body + (1 + (2 * fin_thickness) / mean_chord_len) * A_wet_fins)) / (A_wet_fins + A_wet_body)

    return Cd

def getNoseCd(v: float) -> float:
    """Takes rocket velocity v, outputs nose pressure drag cd"""
    # openrocket 3.86
    mach = v/343
    Cd = 0.8 * (np.sin(nose_angle)**2)
    if mach <= .8:
        return Cd

    return -1 # TODO  

# scalar last - i,j,k,scalar
def axis_angle_to_quat(v, angle):
    """axis angle rep to quat"""
    v = v / np.linalg.norm(v) # get unit vec
    v *= np.cos(angle/2)
    return Rotation.from_quat(np.append(v, [np.sin(angle/2)]))


def body2Inertial(rotation, x):
    """Rotate vector x to inertial frame given rotation"""
    psi, theta, phi = rotation
    rotation_mat = np.array(
        [
            [cos(psi) * cos(theta), cos(theta) * sin(psi), -sin(theta)],
            [
                cos(psi) * sin(phi) * sin(theta) - cos(phi) * sin(psi),
                cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta),
                cos(theta) * sin(phi),
            ],
            [
                sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta),
                cos(phi) * sin(psi) * sin(theta) - cos(psi) * sin(phi),
                cos(phi) * cos(theta),
            ],
        ]
    )

    return np.matmul(rotation_mat, x)


def inertial2Body(rotation, x):
    """Given angle of rocket and force in world reference frame, rotate force into body reference frame"""
    # inverse of body2Inertial mat
    psi, theta, phi = rotation
    world_mat = np.array(
        [
            [cos(psi) * cos(theta), cos(theta) * sin(psi), -sin(theta)],
            [
                cos(psi) * sin(phi) * sin(theta) - cos(phi) * sin(psi),
                cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta),
                cos(theta) * sin(phi),
            ],
            [
                sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta),
                cos(phi) * sin(psi) * sin(theta) - cos(psi) * sin(phi),
                cos(phi) * cos(theta),
            ],
        ]
    )
    body_mat = inv(world_mat)

    return np.matmul(body_mat, x)
