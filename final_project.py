import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import sm_functions as sm

# the file that contains the reference waveform off of an aluminum plate
ref_file = 'Refs\\ref 21JUN2016\\200 ps waveform.txt'

# the number of layers in the composite
n_layers = 10

# thickness of the sample in mm
# thickness = 6.35  # 6.35 mm = 1/4 inch

# thickness of the fiberglass layers in mm
t_fiberglass = 0.25

# thickness of the epoxy layers in mm
t_epoxy = 0.05

# thickness of the sample in mm
thickness = n_layers*(t_fiberglass+t_epoxy) - t_epoxy

# diameter of the beam in mm
beam_width = 1.5

# width of the fiberglass in mm
width = 5

# number of voids to have in the simulation
n_voids = 15

# width & height of the void in mm
void_size = np.array([0.5, t_epoxy])  # for now have void occupy same thickness as epoxy

###################################################
# Begin simulation

# start be handling the reference waveform that is needed

# first thing to do is read in the reference waveform
ref_data = pd.read_csv(ref_file, delimiter='\t')
time = ref_data['Optical Delay/ps'].values
ref_amp = ref_data['Raw_Data/a.u.'].values

# shift the time array so it starts at zero
time -= time[0]

dt = time[1]

df = 1 / (len(time)*dt)

freq = np.linspace(0, len(time)/2*df, len(time)//2+1)

# gate to remove the front artifact signal
gate = 315

# remove the false signal
ref_amp[:gate] = 0

t1 = time[gate-25]
t2 = time[gate]

m = ref_amp[gate] / (t2-t1)  # slope
b = -m*t2 + ref_amp[gate]

ref_amp[gate-25:gate] = m*time[gate-25:gate] + b

dt = time[1]

# convert reference signal to frequency domain
e0 = np.fft.rfft(ref_amp) * dt

plt.figure('Reference Signal In Time Domain')
plt.plot(time, ref_amp, 'r')
plt.axvline(time[315], color='k', linestyle='--')
plt.title('Reference Signal In Time Domain')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Reference Signal Spectrum')
plt.plot(freq, np.abs(e0), 'r')
plt.title('Reference Signal in Frequency Domain')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

# x location of the void center
x_centers = np.random.rand(n_voids) * width

# y location of the void center
# assume that voids can only occur at layer boundaries, so this is the layer at
# which the void will occur

# random.randint() high is exclusive
y_centers_int = np.random.randint(low=1, high=n_layers, size=n_voids).astype(float)
y_centers = y_centers_int * (t_fiberglass + t_epoxy) - t_epoxy/2

y_centers.sort()

# the rectangles that represent voids are constructed by passing through the lower left corner to
# the Rectangle class below in the for loop
x_corners = x_centers - void_size[0] / 2
y_corners = y_centers - void_size[1] / 2

beam_corner = (width/2 - beam_width/2, 0)

# build the voids and store them in a list
# this allows us to keep track of their location, so we can see if the THz beam
# hits their boundary box later
void_list = list()
for i in range(n_voids):
    void = Rectangle((x_corners[i], y_corners[i]), void_size[0], void_size[1])
    void_list.append(void)

# create a patch collection object so we can plot the voids
pc = PatchCollection(void_list, alpha=0.5)

fig = plt.figure('Location of void centers')
axis = fig.add_subplot(111)
for i in range(1, n_layers):  # put a line at each layer boundary (middle of epoxy layer)
    layer_height = i * (t_fiberglass+t_epoxy) - (t_epoxy/2)
    plt.axhline(layer_height, color='k', linestyle='--', linewidth=0.5)

# add the voids to the plot
axis.add_collection(pc)

# add another rectangle to represent THz beam
beam = Rectangle(beam_corner, beam_width, n_layers, facecolor='r', alpha=0.25)
axis.add_patch(beam)

# use a scatter plot to show void center
axis.scatter(x=x_centers, y=y_centers, color='g', label='Void Center')

plt.xlabel('X Location (mm)')
plt.ylabel('Sample Depth')
plt.title('Diagram of Simulation')
plt.xlim(0, width)
# plt.yticks(np.arange(0, n_layers+1))
plt.ylim(thickness, 0)  # flip y-axis so layer 0 is on top
plt.legend()

# now we want to see if the THz beam hits any of the voids
in_beam = [False] * n_voids  # create a list stating whether the beam hits the void or not

# beam left and right bounds
beam_lb = beam.get_x()
beam_rb = beam_lb + beam_width

voids_hit = list()  # contains the voids actually hit by the beam

for i, void in enumerate(void_list):
    void_lb = void.get_x()
    void_rb = void_lb + void.get_width()
    if beam_lb < void_lb < beam_rb or beam_lb < void_rb < beam_rb:
        in_beam[i] = True
        voids_hit.append(void)

print('Number of voids hit by beam = %d' % np.sum(in_beam))

# max number of layers that can have voids is the number of voids hit
n_layers_with_voids = len(voids_hit)
for i in range(1, len(voids_hit)):
    if voids_hit[i].get_y() == voids_hit[i-1].get_y():
        n_layers_with_voids -= 1

# print a yellow star on the voids that are in the beam
for i in range(n_voids):
    if in_beam[i]:
        plt.scatter(x=x_centers[i], y=y_centers[i], marker='*', color='y')

# start by assuming sample is homogeneous
# start with angle straight on

# air to fiberglass reflection and transmission coefficients
r01 = sm.reflection_coefficient(1, 1.55)
t01 = sm.transmission_coefficient(1, 1.55)

# fiberglass to air reflection and transmission coefficients
r10 = sm.reflection_coefficient(1.55, 1)
t10 = sm.transmission_coefficient(1.55, 1)

coverage = np.zeros(n_layers_with_voids)
t_layers = np.zeros(n_layers_with_voids*2+1)

j = 0
for i, void in enumerate(voids_hit):

    void_lb = void.get_x()
    void_rb = void_lb + void.get_width()

    if i != 0 and voids_hit[i].get_y() != voids_hit[i-1].get_y():
        j += 1

    # determine percentage of beam that will hit void
    if void_lb < beam_lb and void_rb > beam_rb:  # void covers entire beam
        print()
        print('Void covers entire beam')
        coverage[j] += 1
        # entire beam will be multiplied by reflection coefficient

    elif void_lb > beam_lb and void_rb < beam_rb:  # void in completely inside of beam
        print()
        print('Void completely inside of beam')
        coverage[j] += void.get_width() / beam_width
        print('Coverage = %0.2f' % coverage[j])

    elif void_lb < beam_lb and void_rb < beam_rb:  # right part of void is inside beam
        print()
        print('Beam hits right part of void')
        coverage[j] += (void_rb - beam_lb) / beam_width
        print('Coverage = %0.2f' % coverage[j])

    elif void_lb > beam_lb and void_rb > beam_rb:  # left part of void is inside beam
        print()
        print('Beam hits left part of void')
        coverage[j] += (beam_rb - void_lb) / beam_width
        print('Coverage = %0.2f' % coverage[j])

coverage[np.where(coverage > 1)] = 1

# for i, void in enumerate(voids_hit):
#     t_layers[i*2+1] = void.get_height()
#
#     if i == 0:
#         t_layers
