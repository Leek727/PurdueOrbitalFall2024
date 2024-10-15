import numpy as np
import pandas as pd

df = pd.read_csv("TheseusVars.csv")
m = df["Mass (g)"][0] / 1000
I1 = df["Longitudinal moment of inertia (kg·m²)"][0]
I2 = df["Rotational moment of inertia (kg·m²)"][0]
I3 = I2


def getThrust(t: float):  # get thrust at time in s
    time = list(df["# Time (s)"])
    thrust_arr = list(df["Thrust (N)"])
    if t > time[-1]:
        return 0
    
    time_ind = np.searchsorted(time, t)
    thrust = thrust_arr[time_ind]

    return thrust

diameter = 15.7 / 100  # m

# R critical based on Rs (surface roughness in m) and length
Rs = 5 * 10 ** (
    -6
)  # roughness of optimum paint-spray surface https://openrocket.sourceforge.net/thesis.pdf - pg 54

# body vars
radius = diameter / 2
inner_radius = diameter / 2 - 0.014
body_length = 284 / 100  # m

mean_chord_len = (
    0.7  # https://en.wikipedia.org/wiki/Chord_(aeronautics)#Mean_aerodynamic_chord
)
fin_thickness = 0.001
A_wet_fins = 0.2  # surface area of fins accounting for thickness
A_wet_body = (
    2 * np.pi * radius * body_length
)  # surface area of the entire rocket body in contact with air - not including fins
# A_ref = A_wet_body + A_wet_fins # entire surface area of rocket

air_visc_ref = 1.716 * 10 ** (
    -5
)  # Ns/m^2 mu0 https://www.grc.nasa.gov/www/k-12/airplane/viscosity.html
sutherland_const = 111  # K
sutherland_t0 = 273  # K

# nose cone vars
# openrocket 3.86
nose_angle = np.pi / 6  # rads

# inertia
# m = 20 # kg

# density = 2700 #kg/m^3
"""
I1 = ((np.pi * density * body_length)/2) * (radius**4-inner_radius**4)
I2 = ((np.pi * density * body_length)/12) * (3 * (radius**4 - inner_radius**4) + (body_length**2) * (radius**2-inner_radius**2))
I3 = I2
"""
