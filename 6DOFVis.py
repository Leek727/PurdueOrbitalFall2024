import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from util import *
from _6DOFSim import pos

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