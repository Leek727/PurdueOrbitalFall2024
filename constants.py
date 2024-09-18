import numpy as np

diameter = .05

# Reynolds num
R = 10

# R critical based on Rs (surface roughness Height/um) and length
Rs = 5 # roughness of optimum paint-spray surface https://openrocket.sourceforge.net/thesis.pdf - pg 54

# body vars
radius = .1
body_length = 1

A_ref = np.pi * radius**2 # max cross sectional area? reference area
mean_chord_len = .7 # https://en.wikipedia.org/wiki/Chord_(aeronautics)#Mean_aerodynamic_chord
fin_thickness = .001
A_wet_fins = .2 # surface area of fins accounting for thickness
A_wet_body = 2 * np.pi * radius * body_length # surface area of the entire rocket body in contact with air probably
