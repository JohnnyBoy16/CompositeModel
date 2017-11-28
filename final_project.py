import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# the number of layers in the composite
n_layers = 10

# thickness of the sample in mm
# thickness = 6.35  # 6.35 mm = 1/4 inch

# thickness of the fiberglass layers in mm
t_fiberglass = 0.25

# thickness of the epoxy layers in mm
t_epoxy = 0.05

# thickness of the sample in mm
thickness = n_layers*(t_fiberglass+t_epoxy) - 2*t_epoxy

# diameter of the beam in mm
d_beam = 1.5

# width of the fiberglass in mm
width = 5

# number of voids to have in the simulation
n_voids = 15

# width & height of the void in mm
void_size = np.array([0.5, 0.25])

###################################################
# Begin simulation

# x location of the void center
x_centers = np.random.rand(n_voids) * width

# y location of the void center
# assume that voids can only occur at layer boundaries, so this is the layer at
# which the void will occur

# random.randint() high is exclusive
y_centers_int = np.random.randint(low=1, high=n_layers, size=n_voids).astype(float)
y_centers = y_centers_int * ((t_fiberglass + t_epoxy) - (t_epoxy / 2))

x_corners = x_centers - void_size[0] / 2
y_corners = y_centers - void_size[1] / 2

beam_corner = (width/2 - d_beam/2, 0)

# build the voids and store them in a list
# this allows us to keep track of their info, so we can see if the THz beam
# hits their boundary box
void_collection = list()
for i in range(n_voids):
    void = Rectangle((x_corners[i], y_corners[i]), void_size[0], void_size[1])
    void_collection.append(void)

# create a patch collection object so we can plot the voids
pc = PatchCollection(void_collection, alpha=0.5)

fig = plt.figure('Location of void centers')
axis = fig.add_subplot(111)
for i in range(n_layers+1):  # put a line at each layer boundary
    layer_height = i * ((t_fiberglass+t_epoxy) - (t_epoxy/2))
    plt.axhline(layer_height, color='k', linestyle='--', linewidth=0.5)

# add the voids to the plot
axis.add_collection(pc)

# add another rectangle to represent THz beam
beam = Rectangle(beam_corner, d_beam, n_layers, facecolor='r', alpha=0.5)
axis.add_patch(beam)

# use a scatter plot to show void center
axis.scatter(x=x_centers, y=y_centers, color='g', label='Void Center')

plt.xlabel('X Location (mm)')
plt.ylabel('Layer Boundary')
plt.title('Diagram of Simulation')
plt.xlim(0, width)
# plt.yticks(np.arange(0, n_layers+1))
plt.ylim(thickness, 0)  # flip y-axis so layer 0 is on top
plt.legend()
