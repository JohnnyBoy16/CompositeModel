import pdb
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.optimize import curve_fit

import sm_functions as sm

# the file that contains the reference waveform off of an aluminum plate
ref_file = 'Refs\\ref 11JUL2016\\110 ps waveform.txt'

# file that contains the noise taken when the laser was highly out of focus
# should be just noise
noise_file_base = 'NoiseWaveforms\\110 ps Noise Waveform '

# number of noise waveforms to use
n_noise_waveforms = 25

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
void_size = np.array([beam_width, t_epoxy])

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

# read in the noise data files
noise_amp = np.zeros((25, len(ref_amp)))
for i in range(n_noise_waveforms):
    noise_file = noise_file_base + str(i+1) + '.txt'
    noise_data = pd.read_csv(noise_file, delimiter='\t')
    # already have the time array from the reference signal, so don't need it here
    noise_amp[i, :] = noise_data['Raw_Data/a.u.'].values

# shift the time array so it starts at zero
time -= time[0]

dt = time[1]

df = 1 / (len(time)*dt)

freq = np.linspace(0, len(time)/2*df, len(time)//2+1)

omega = freq * 2 * np.pi * 1e12  # create omega array for plotting

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
plt.plot(freq, np.abs(e0[:len(freq)]), 'r')
plt.title('Reference Signal in Frequency Domain')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

# whether or not there is a void in the composite layer
# use a binomial random variable for simplicity
void_in_layer = np.zeros(n_layers)
for i in range(1, n_layers, 2):
    void_in_layer[i] = np.random.binomial(1, p_void)

# print a warning if no voids were created in the simulation
if np.sum(void_in_layer) == 0:
    print('No voids created in simulation!')

for i in range(len(void_in_layer)):
    if void_in_layer[i]:
        print('void in layer', (i+1))

# for now we will assume that the void will be fully covered by the beam
beam_corner = (-beam_width / 2, 0)
beam = Rectangle(beam_corner, beam_width, thickness, facecolor='r', alpha=0.25)

# build voids and add them to a list to keep track of
void_list = list()
for i in range(len(void_in_layer)):
    if void_in_layer[i]:
        x_corner = -void_size[0] / 2
        y_corner = (i+1)//2 * (t_fiberglass+t_epoxy)
        void = Rectangle((x_corner, y_corner), void_size[0], -void_size[1])
        void_list.append(void)

# build the layers for plotting
layer_list = list()
for i in range(1, n_layers, 2):
    x_corner = -beam_width / 2
    y_corner = (i + 1) // 2 * (t_fiberglass + t_epoxy)
    layer = Rectangle((x_corner, y_corner), beam_width, -t_epoxy)
    layer_list.append(layer)

# create a patch collection object so we can plot the voids
void_collection = PatchCollection(void_list, facecolor='blue', edgecolor='black')
layer_collection = PatchCollection(layer_list, facecolor='brown', alpha=0.5)

fig = plt.figure('Diagram of Simulation')
axis = fig.add_subplot(111)

# add the beam to the diagram
# just a semi-transparent red rectangle
# axis.add_patch(beam)

# add the voids to the diagram

axis.add_collection(layer_collection)
axis.add_collection(void_collection)

# add dotted lines where the layer boundaries are
for i in range(1, n_layers):  # put a line at each layer boundary
    layer_height = i * (t_fiberglass+t_epoxy) - (t_epoxy/2)
    plt.axhline(layer_height, color='k', linestyle='--', linewidth=0.5)

# plt.title('Diagram of Simulation')
plt.xlabel('X Location (mm)')
plt.ylabel('Depth into the composite (mm)')
plt.ylim(thickness, 0)  # flip y-axis so layer 0 is on top
plt.xlim(beam_corner[0], beam_corner[0]+beam_width)

# start by constructing a model of the composite sample with no voids present

# the angle of the beam in each layer, including air on each side of sample
# for now let theta be all zeros for simplicity
theta_array = np.zeros(n_layers+2)

# create the array of index of refraction values for each layer in
# the simulation
n = np.ones(n_layers+2, dtype=complex)
for i in range(1, n_layers+1, 2):
    n[i] = n_fiberglass
    n[i+1] = n_epoxy

# the for loop above makes the last layer an epoxy layer, when it needs to
# be air so correct that here
n[-1] = 1.0

# create an array of layer thicknesses
thickness_array = np.zeros(n_layers)
for i in range(n_layers):
    if i % 2 == 0:  # if i is even it is a fiberglass layer
        thickness_array[i] = t_fiberglass
    else:
        thickness_array[i] = t_epoxy

# build an array that contains how large each void is relative to the beam size
# leave this as zeros and pass to reflection model equation to create sample
# model with no defects
coverage = np.zeros(n_layers)

gamma = sm.global_reflection_model(n, theta_array, freq, thickness_array,
                                   n_layers, coverage)

e1 = e0 * gamma[0]

return_amp = np.fft.irfft(e1) / dt

peak_times = [15.95, 19.44, 23.01, 26.59, 30.16, 33.72, 40.84]
peak_amps = [0.391, 0.317, 0.229, 0.162, 0.109, 0.079, 0.027]

# create a exponential fit to the data from the peaks of the modeled response
p = curve_fit(lambda t, a, b: a*np.exp(b*t), peak_times, peak_amps,
              p0=(1, -0.05))[0]

x = np.linspace(10, 60, 500)
y = p[0]*np.exp(p[1]*x)

plt.figure('Sample with No Defects in Frequency Domain')
plt.plot(freq, np.abs(e1), 'r')
plt.title('Sample with No Defects in Frequency Domain')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

plt.figure('Sample with No Defects in Time Domain')
plt.plot(time, return_amp, 'r')
plt.plot(x, y, 'k--')
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
plt.plot(x, y, 'k--')
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

# Start analysis of the system noise

plt.figure('Example of System Noise')
plt.plot(time, noise_amp[0], 'r', linewidth=0.5)
plt.title('System Noise')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

noise_mean = np.mean(noise_amp)
print(noise_mean)

# estimate the standard deviation
noise_std = np.std(noise_amp, axis=1)
noise_std = np.mean(noise_std)

# subtract the mean noise level from the system noise so we
# can determine AR1 bandwith parameter
noise_amp -= noise_mean

# plot an example correlation from the first noise waveform
correlation = np.zeros(len(noise_amp[0])*2-1)
for i in range(n_noise_waveforms):
    correlation += np.correlate(noise_amp[i, :], noise_amp[i, :], mode='full')

correlation /= n_noise_waveforms

plt.figure('Average Noise Correlation')
plt.plot(correlation[correlation.argmax():], 'ro-')
plt.title('Noise Correlation')
plt.xlabel('Data Point')
plt.ylabel('Correlation')
plt.grid()

corr = np.correlate(noise_amp[0], noise_amp[0], mode='full')
plt.figure('Noise Correlation')
plt.plot(corr[corr.argmax():], 'ro-')
plt.title('Noise Correlation')
plt.xlabel('Point')
plt.ylabel('Correlation')
plt.grid()

# take a look at the psd by taking splitting noise into 50, 2048 point arrays
noise_reshaped = noise_amp.reshape((50, 2048))
noise_psd = np.abs(np.fft.fft(noise_reshaped, n=len(noise_amp[0]), axis=1))**2
noise_psd = noise_psd.mean(axis=0)

N_hat_dB = 10*np.log10(noise_psd)

plt.figure('Noise PSD Estimate')
plt.plot(omega, N_hat_dB[:len(omega)], 'r', linewidth=0.5)
plt.xlabel(r'Frequency ($\omega$)')
plt.ylabel('Power (dB)')
plt.grid()

alpha = 0
for i in range(n_noise_waveforms):
    correlation = np.correlate(noise_amp[i, :], noise_amp[i, :], mode='full')
    alpha += correlation[len(noise_amp[0])] / correlation.max()  # R(1) / R(0)

alpha /= n_noise_waveforms

# determine the variance of the white noise
var_w = noise_std**2 - alpha**2 * noise_std**2
sigma_w = np.sqrt(var_w)

noise_sim = np.zeros(len(noise_amp[0])+500)

w = np.random.normal(0, sigma_w, len(noise_amp[0])+500)

# determine the constant in the AR process
c = noise_mean - alpha*noise_mean

for i in range(1, len(noise_amp[0])+500):
    noise_sim[i] = c + alpha*noise_sim[i-1] + w[i]

noise_sim = noise_sim[500:]  # throw away the first 500 points

plt.figure('Noise Simulation')
plt.plot(time, noise_sim, 'r', linewidth=0.5)
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

noise_sim *= 25

# measure signal from composite
signal_and_noise = return_amp_flawed+noise_sim

# Plot an example of the measured signal with noise
plt.figure('Sample with Defects and Noise in Time Domain')
plt.plot(time, signal_and_noise, 'r')
plt.title('Sample with Defects in Time Domain')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

# generate a large time series of the noise simulation and estimate the psd
n_points = 2048 * 50
nfft = 4096  # 4096 points in the noise waveforms I have captured

w = np.random.normal(0, sigma_w, n_points+500)

n_hat = np.zeros(n_points+500)
for i in range(1, n_points+500):
    n_hat[i] = c + alpha*n_hat[i-1] + w[i]

n_hat = n_hat[500:]  # throw aways first 500 points

n_hat2 = n_hat.reshape((50, 2048)) * 25

N_hat = np.abs(np.fft.fft(n_hat2, nfft))**2

Sn_hat = np.mean(N_hat, axis=0)

Sn_hat_dB = 10*np.log10(Sn_hat)

plt.figure('Noise Simulation PSD Estimate')
plt.plot(omega, Sn_hat_dB[:len(omega)], 'r', linewidth=0.5)
plt.xlabel(r'Frequency ($\omega$)')
plt.ylabel('Power (dB)')
plt.grid()

# simulate measuring the sample area of the sample multiple
# times to get a better estimate of Sx
n_sim = 50
sx = np.zeros((n_sim, len(return_amp)+500))

w = np.random.normal(0, sigma_w, (n_sim, len(return_amp)+500))
for i in range(n_sim):
    for j in range(1, len(return_amp)):
        sx[i, j] = c + alpha*sx[i, j-1] + w[i, j]

# throw away the extra points at the beginning
sx = sx[:, 500:]

sx *= 25  # scale up noise

# add the composite with voids model to the noise
for i in range(n_sim):
    sx[i, :] += return_amp_flawed

# create psd of measured_signal
Sx_hat = np.abs(np.fft.fft(sx, nfft))**2

# take the mean of each fft, down column
Sx_hat = np.mean(Sx_hat, axis=0)

Sx_hat_dB = 10*np.log10(Sx_hat)

plt.figure('Model with Noise PSD Estimate')
plt.plot(omega, Sx_hat_dB[:len(omega)], 'r', linewidth=0.5)
plt.xlabel('Frequency ($\omega$)')
plt.ylabel('Power (dB)')
plt.grid()

plt.figure('Noise & Signal+Noise PSD Estimates')
plt.plot(omega, Sn_hat_dB[:len(omega)], 'k', linewidth=0.5)
plt.plot(omega, Sx_hat_dB[:len(omega)], 'b', linewidth=0.5)
plt.xlabel(r'Frequency ($\omega$)')
plt.ylabel('Power (dB)')
plt.grid()

H_hat = 1 - Sn_hat / Sx_hat

for i in range(len(H_hat)):
    if H_hat[i] < 0:
        H_hat[i] = 0

plt.figure('Estimated Wiener Filter')
plt.plot(omega, H_hat[:len(omega)], 'r', linewidth=0.75)
plt.xlabel(r'Frequency ($\omega$)')
plt.ylabel('Filter Estimate')
plt.grid()

# generate a new noise set to use with wiener filter
sx = np.zeros(len(return_amp)+500)

w = np.random.normal(0, sigma_w, len(return_amp)+500)
for i in range(1, len(return_amp)):
    sx[i] = c + alpha*sx[i] + w[i]

sx = sx[500:] * 25

sx += return_amp_flawed  # add the model with flaws to the noise

S_hat_estimate = H_hat * np.fft.fft(sx)

s_estimate = np.fft.ifft(S_hat_estimate)

plt.figure('Estimated Signal')
plt.plot(time, s_estimate, 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()
