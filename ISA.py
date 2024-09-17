def ISA_data(alt: int):
    """
    Internation standard Atmosphere

    Inputs: Altitude in meters integer only

    Outputs: 
    Temperature (Kelvin)
    Pressure (Pa)
    Density (kg/m^3)
    a (Speed of Sound) (m/s)

    Syntax: [Temp,Pressure,Density,a] = ISA_data(altitude)
    """
    import numpy as np
    ## Initialization
    #Height
    h_1 = np.linspace(0, 11000, 11000) # m
    h_2 = np.linspace(11000, 25000, (25000-11000)) # m
    h_3 = np.linspace(25000, 39620, (39620-25000)) # m
    height = np.hstack((h_1, h_2, h_3))
    h_english = np.multiply(height, 3.281)

    # Constant values
    p_0 = 1.01325 * (10**5) # N/m^2 or Pa (Pressure at H=0)
    rho_0 = 1.2250 # Kg/m^3 (density at H=0)
    t_0 = 288.16 # Kelvin (Temperature at H=0)
    h_0 = 0 # sea level (m)
    a_1 = -6.5 * 10**-3 # Lapse rate (K/m)
    a_2 = 3*10**-3 # lapse rate (K/m)
    R = 287 # (J/kg*K)
    g = 9.81 #Gravity on earth

    ## Calculations
    # Temperature 
    Temp_1 = t_0 + a_1 * (h_1)
    Temp_2 = 216.66 + np.linspace(0,0, (25000-11000))
    Temp_3 = 216.66 + a_2 * ((h_3 - 25000))
    T = np.hstack((Temp_1, Temp_2, Temp_3)) # Kelvin
    T_english = (T - 273.15) * (9/5) + 32; # Fahrenheit

    # Pressure
    Pressure_1 = p_0 * (np.divide(Temp_1, t_0) ** -(g / (R * a_1)))
    Pressure_2 = Pressure_1[11000 - 1] * np.exp(-(g / (R * Temp_2[14000-1])) * (h_2 - 11000))
    Pressure_3 = Pressure_2[14000-1] * ((Temp_3 / Temp_2[14000-1]) ** -(g / (R * a_2)))
    P = np.hstack((Pressure_1, Pressure_2, Pressure_3)) # Pa
    P_english = P / 6895; # PSI

    # Density
    Density_1 = rho_0 * (Temp_1 / t_0) ** -((g / (a_1 * R)) + 1)
    Density_2 = Density_1[11000-1] * np.exp( -(g / (R * Temp_2)) * (h_2 - h_1[11000-1] ))
    Density_3 = Density_2[14000-1] * ((Temp_3 / Temp_2[14000-1]) ** -((g / (R * a_2)) + 1))
    D = np.hstack((Density_1, Density_2, Density_3)) # Kg/m^3
    D_english = D / 515.4 # slug/ft^3

    # Speed of Sound
    a = np.sqrt(1.4 * R * T) # speed of sound in (m/s)
    a_english = a / 3.281 # ft/s
    h_find = 2743
    # fprintf("At #d height (m) - Temperature: #d\n\nPressure - #d \n\nDensity - #d\n\nSpeed of Sound - #d\n", h_find, T_english(h_find), P_english(h_find), D_english(h_find), a_english(h_find))

    return T[alt], P[alt], D[alt], a[alt]