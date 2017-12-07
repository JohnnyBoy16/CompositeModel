import pdb
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import sm_functions as sm

sys.path.insert(0, 'C:\\PycharmProjects\\THzProcClass')
from THzData import THzData

basedir = 'C:\\Work\\Faint Defect Testing\\Yellow Composite'
filename = 'Scan with Two Tape Defects F@FS (res=0.5mm).tvl'

data = THzData(filename, basedir, gate=[[3111, 3302], [700, 900]])

plt.figure('C-Scan Test')
plt.imshow(data.c_scan, interpolation='none', cmap='gray', extent=data.c_scan_extent)
plt.xlabel('X Scan Location (mm)')
plt.ylabel('Y Scan Location (mm)')
plt.colorbar()
plt.grid()

defect = data.c_scan[22:41, 46:60]
normal_area = data.c_scan[22:41, 19:33]

weights = np.ones_like(defect.flatten()) / defect.size

plt.figure('Histogram')
plt.hist(defect.flatten(), 15, weights=weights, facecolor='b', label='Tape')
plt.hist(normal_area.flatten(), 15, weights=weights, facecolor='g', label='Normal')
plt.title('Normalized Histogram of Pixel Values')
plt.ylabel('Probability')
plt.xlabel('Peak to Peak Voltage')
plt.legend()

# the file that contains the reference waveform off of an aluminum plate
ref_file = 'Refs\\ref 11JUL2016\\110 ps waveform.txt'

noise_file = 'NoiseWaveforms\\110 ps system noise.txt'

# the number of layers in the composite
# including epoxy layers, needs to be an odd number
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

noise_data = pd.read_csv(noise_file, delimiter='\t')
noise_amp = noise_data['Raw_Data/a.u.'].values

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

e1_flawed = e0 * gamma[0]

return_amp_flawed = np.fft.irfft(e1_flawed) / dt

plt.figure('Sample with Defects in Frequency Domain')
plt.plot(freq, np.abs(e1_flawed), 'r')
plt.title('Sample with Defects in Frequency Domain')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

plt.figure('Sample with Defects in Time Domain')
plt.plot(time, return_amp_flawed, 'r')
plt.axvline(time[1750], color='k', linestyle='--')
plt.axvline(time[2500], color='k', linestyle='--')
plt.title('Sample with Defects in Time Domain')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

print()
print('Peak to Peak Value with no defects')
print(return_amp[1750:2500].max() - return_amp[1750:2500].min())
print()
print('Peak to Peak value with defects')
print(return_amp_flawed[1750:2500].max() - return_amp_flawed[1750:2500].min())

plt.figure('System Noise')
plt.plot(time, noise_amp, 'r', linewidth=0.5)
plt.title('System Noise')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

noise_mean = np.mean(noise_amp)
noise_std = np.std(noise_amp)

noise_amp -= noise_mean

corr = np.correlate(noise_amp, noise_amp, mode='full')
plt.figure('Noise Correlation')
plt.plot(corr, linewidth=0.5)
plt.xlabel('Point')
plt.ylabel('Correlation')
plt.grid()

# estimate alpha, time consuming
alpha1 = 0.766452  # this is the alpha that is calculated below

# for i in range(len(noise_amp)):
#     correlation = np.correlate(np.roll(noise_amp, i), np.roll(noise_amp, i+1), mode='full')
#     alpha1 += correlation[len(noise_amp)-1] / correlation.max()
#
# alpha1 /= len(noise_amp)

noise_sim = np.zeros(len(noise_amp))
noise_sim[0] = np.random.randn(1) * noise_std*10 + noise_mean  # initialize noise values

w = np.random.randn(len(noise_amp))*noise_std*10 + noise_mean

for i in range(1, len(noise_amp)):
    noise_sim[i] = alpha1*noise_sim[i-1] + w[i]

plt.figure('Noise Simulation')
plt.plot(time, noise_sim, 'r', linewidth=0.5)
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

plt.figure('Sample with Defects and Noise in Time Domain')
plt.plot(time, return_amp_flawed+noise_sim, 'r')
plt.title('Sample with Defects in Time Domain')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

# generate a large time series of the noise simulation and estimate the psd
n_points = 10000
nfft = 4096

# generate the noise
v = noise_std*10 * np.random.randn(n_points) + noise_mean

n_hat = np.zeros(n_points)
for i in range(1, n_points):
    n_hat[i] = alpha1*n_hat[i-1] + v[i]

n_hat2 = n_hat.reshape((50, 200))

N_hat = np.abs(np.fft.rfft(n_hat2, nfft))**2

Sn_hat = np.mean(N_hat, axis=0)

plt.figure('Noise PSD Estimate')
plt.plot(freq, Sn_hat, 'r')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

e1_flawed_noise = np.fft.rfft(