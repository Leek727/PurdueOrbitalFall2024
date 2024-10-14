import numpy as np
from numpy import cos, sin
from constants import *
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from ISA import ISA_data

# utility functions


# ------------------------- aero -----------------------------
def getReynolds(v: float, alt: int) -> float:
    """v: flow speed, T: temp in Kelvin -> Reynolds num"""
    # pretty close to the nasa applet

    temp, pressure, density_rho, speed_sound = ISA_data(alt)
    # https://www.grc.nasa.gov/www/k-12/airplane/reynolds.html
    dynamic_visc = (
        air_visc_ref
        * ((temp / sutherland_t0) ** (3 / 2))
        * ((sutherland_t0 + sutherland_const) / (temp + sutherland_const))
    )
    kinematic_visc = dynamic_visc / density_rho

    return (
        v * body_length
    ) / kinematic_visc


def getSkinDrag(v: float, alt: int) -> float:
    """Abs velocity of rocket m/s, Altitude m-> skin friction drag FORCE"""
    # get reynolds
    Re = getReynolds(v, alt)
    temp, pressure, density_rho, speed_sound = ISA_data(alt)
    R_crit = 51 * (Rs / body_length) ** (-1.039)
    
    # skin friction coefficient for turbulent flow
    if Re < 10 **4:
        Cf = 1.48 * 10 ** (-2)
        #print("low reynolds")
    elif Re > 10**4 and Re < R_crit:
        #print("normal")
        Cf = 1 / ((1.5 * np.log(Re) - 5.6) ** 2)

    elif Re >= R_crit:
        #print("critical")
        Cf = 0.032 * ((Rs / body_length) ** 0.2)


    # apply compressibility corrections
    M = v / speed_sound  # mach number
    if M < 1:  # subsonic
        Cf = Cf * (1 - 0.1 * M**2)

    elif M >= 1:
        # check if "roughness-limited" or above R_crit
        if Re > R_crit:
            Cf = Cf / (1 + 0.15 * M**2) ** 0.58
        else:
            Cf = Cf / (1 + 0.18 * M**2)

    

    # get actual drag coefficient
    fB = body_length / diameter  # fineness ratio / how slim it is
    Cd = (
        Cf
        * (
            (
                1
                + 1 / (2 * fB) * A_wet_body
                + (1 + (2 * fin_thickness) / mean_chord_len) * A_wet_fins
            )
        )
        / (A_wet_fins + A_wet_body)
    )

    # https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation-2/
    # maybe right
    return Cd * ((density_rho * v**2)/2) * (A_wet_body + A_wet_fins)


def getNoseCd(v: float) -> float:
    """Takes rocket velocity v, outputs nose pressure drag cd"""
    # openrocket 3.86
    mach = v / 343
    Cd = 0.8 * (np.sin(nose_angle) ** 2)
    if mach <= 0.8:
        return Cd

    return -1  # TODO


# ------------------------- rotation  -----------------------------
# scalar last - i,j,k,scalar
def axis_angle_to_quat(v, angle):
    """axis angle rep to quat"""
    v = v / np.linalg.norm(v)  # get unit vec
    v *= np.cos(angle / 2)
    return Rotation.from_quat(np.append(v, [np.sin(angle / 2)]))


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

if __name__ == "__main__":
    for i in range(1000):
        print(getSkinDrag(i,i*39))