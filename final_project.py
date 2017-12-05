import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import sm_functions as sm

# the file that contains the reference waveform off of an aluminum plate
ref_file = 'Refs\\ref 11JUL2016\\110 ps waveform.txt'

# the number of layers in the composite
# including epoxy layers
n_layers = 19

# thickness of the fiberglass layers in mm
t_fiberglass = 0.25

# thickness of the epoxy layers in mm
t_epoxy = 0.05

# diameter of the beam in mm
beam_width = 1.5

# width of the fiberglass in mm
width = beam_width

# the chance of having a void in each layer
p_void = 0.4

# width & height of the void in mm
# for now have void occupy same thickness as epoxy, so the entire thickness of the composite
# layer is occupied by the void if there is one
void_size = np.array([0.5, t_epoxy])

n_fiberglass = 1.85 - 1j*0.02  # index of refraction of fiberglass
n_epoxy = 1.45 - 1j*0.01  # index of refraction of the epoxy

# the angle of the THz system in the lab in 17.5 deg., but for now assume straight on for
# simplicity

# incoming angle of the THz beam in degrees
theta0 = 0.0
theta0 *= np.pi / 180  # convert to radians

# thickness of the sample in mm
thickness = n_layers//2*(t_fiberglass+t_epoxy) + t_fiberglass

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
gate = 400

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
plt.axvline(time[gate], color='k', linestyle='--')
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

# whether or not there is a void in the composite layer
void_in_layer = np.zeros(n_layers)
for i in range(1, n_layers, 2):
    void_in_layer[i] = np.random.binomial(1, p_void)

# print a warning if no voids were created in the simulation
if not np.sum(void_in_layer):
    print('No voids created in simulation!')

for i in range(len(void_in_layer)):
    if void_in_layer[i]:
        print('void in layer', (i+1))

# for now we will assume that the void will be fully covered by the beam
beam_corner = (-beam_width / 2, 0)
beam = Rectangle(beam_corner, beam_width, thickness, facecolor='r', alpha=0.25)

void_list = list()
for i in range(len(void_in_layer)):
    if void_in_layer[i]:
        x_corner = -void_size[0] / 2
        y_corner = (i+1)//2 * (t_fiberglass+t_epoxy)
        void = Rectangle((x_corner, y_corner), void_size[0], -void_size[1])
        void_list.append(void)

# TODO create a patch to add layers to diagram

# create a patch collection object so we can plot the voids
pc = PatchCollection(void_list)

fig = plt.figure('Diagram of Simulation')
axis = fig.add_subplot(111)

# add the beam to the diagram
# just a semi-transparent red rectangle
# axis.add_patch(beam)

# add the voids to the diagram

axis.add_collection(pc)

# add dotted lines where the layer boundaries are
for i in range(1, n_layers):  # put a line at each layer boundary (middle of epoxy layer)
    layer_height = i * (t_fiberglass+t_epoxy) - (t_epoxy/2)
    plt.axhline(layer_height, color='k', linestyle='--', linewidth=0.5)

plt.title('Diagram of Simulation')
plt.xlabel('X Location (mm)')
plt.ylabel('Depth into the composite (mm)')
plt.ylim(thickness, 0)  # flip y-axis so layer 0 is on top
plt.xlim(beam_corner[0], beam_corner[0]+beam_width)

# start by constructing a model of the composite sample with no voids present

# the angle of the beam in each layer, including air on each side of sample
# for now let theta be all zeros for simplicity
theta_array = np.zeros(n_layers+2)

# create the array of index of refraction values for each layer in the simulation
n = np.ones(n_layers+2, dtype=complex)
for i in range(1, n_layers+1, 2):
    n[i] = n_fiberglass
    n[i+1] = n_epoxy

# the for loop above makes the last layer an epoxy layer, when it needs to be air
# so correct that here
n[-1] = 1.0

# create an array of layer thicknesses
thickness_array = np.zeros(n_layers)
for i in range(n_layers):
    if i % 2 == 0:  # if i is even it is a fiberglass layer
        thickness_array[i] = t_fiberglass
    else:
        thickness_array[i] = t_epoxy

# build an array that contains how large each void is relative to the beam size
# leave this as zeros and pass to reflection model equation to create sample model with no defects
coverage = np.zeros(n_layers)

gamma = sm.global_reflection_model(n, theta_array, freq, thickness_array, n_layers, coverage)

e1 = e0 * gamma[0]

return_amp = np.fft.irfft(e1) / dt

plt.figure('Sample with No Defects in Frequency Domain')
plt.plot(freq, np.abs(e1), 'r')
plt.title('Sample with No Defects in Frequency Domain')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

plt.figure('Sample with No Defects in Time Domain')
plt.plot(time, return_amp, 'r')
plt.title('Sample with No Defects in Time Domain')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

###################################################################################################

# now change coverage based on whether this is a void or not and build the model with defects
j = 0
for i in range(len(void_in_layer)):
    if void_in_layer[i]:
        coverage[i] = void_list[j].get_width() / beam_width
        j += 1

gamma = sm.global_reflection_model(n, theta_array, freq, thickness_array, n_layers, coverage)

e1 = e0 * gamma[0]

return_amp = np.fft.irfft(e1) / dt

plt.figure('Sample with Defects in Frequency Domain')
plt.plot(freq, np.abs(e1), 'r')
plt.title('Sample with Defects in Frequency Domain')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

plt.figure('Sample with Defects in Time Domain')
plt.plot(time, return_amp, 'r')
plt.title('Sample with Defects in Time Domain')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

"""
coverage = np.zeros(n_layers_with_voids+2)
t_layers = np.zeros(n_layers_with_voids*2+1)

coverage[0] = 1
coverage[-1] = 1

j = 1
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

# coverage can not be greater than 1
coverage[np.where(coverage > 1)] = 1

d = np.zeros(2*n_layers_with_voids+1)
d[0] = voids_hit[0].get_y() + voids_hit[0].get_height()
d[1] = voids_hit[1].get_height()
d[-1] = thickness - voids_hit[-1].get_y()

j = 1
for i in range(1, len(voids_hit)):

    if voids_hit[i].get_y() != voids_hit[i-1].get_y():
        d[j*2] = voids_hit[i].get_y()+voids_hit[i].get_height() - voids_hit[i-1].get_y()
        d[j*2+1] = voids_hit[i].get_height()

        j += 1

theta = np.zeros(d.size+2)
n = np.zeros(theta.shape, dtype=complex)
n[0] = 1
n[-1] = 1
for i in range(1, len(n)-1):
    if i % 2:  # odd
        n[i] = n_fiberglass
    else:
        n[i] = 1.0

gamma = sm.global_reflection_model(n, theta, freq, d, 2*n_layers_with_voids+1)

e1 = e0 * gamma[0]

return_amp = np.fft.irfft(e1) / dt

plt.figure('Return Signal in Frequency Domain')
plt.plot(freq, np.abs(e1), 'r')
plt.title('Return Signal in Frequency Domain')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

plt.figure('Return Signal in Time Domain')
plt.plot(time, return_amp, 'r')
plt.title('Return Signal in Time Domain')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()
"""