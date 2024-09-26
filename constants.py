import numpy as np

diameter = .05

# Reynolds num
R = 10

# R critical based on Rs (surface roughness Height/um) and length
Rs = 5 # roughness of optimum paint-spray surface https://openrocket.sourceforge.net/thesis.pdf - pg 54

# body vars
radius = .1524
inner_radius = .14986
body_length = 1

mean_chord_len = .7 # https://en.wikipedia.org/wiki/Chord_(aeronautics)#Mean_aerodynamic_chord
fin_thickness = .001
A_wet_fins = .2 # surface area of fins accounting for thickness
A_wet_body = 2 * np.pi * radius * body_length # surface area of the entire rocket body in contact with air - not including fins
#A_ref = A_wet_body + A_wet_fins # entire surface area of rocket


# nose cone vars
# openrocket 3.86
nose_angle = np.pi/6 # rads

# inertia
m = 20 # kg

density = 2700 #kg/m^3

I1 = ((np.pi * density * body_length)/2) * (radius**4-inner_radius**4)
I2 = ((np.pi * density * body_length)/12) * (3 * (radius**4 - inner_radius**4) + (body_length**2) * (radius**2-inner_radius**2))
I3 = I2
