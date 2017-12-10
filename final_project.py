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

# number of noise waveforms collected
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
p_void = 1.0

# width & height of the void in mm
# for now have void occupy same thickness as epoxy, so the entire thickness of the composite
# layer is occupied by the void if there is one
void_size = np.array([beam_width/2, t_epoxy])

n_fiberglass = 1.85 - 1j*0.02  # index of refraction of fiberglass
n_epoxy = 1.45 - 1j*0.01  # index of refraction of the epoxy

# the angle of the THz system in the lab in 17.5 deg., but for now assume straight on for
# simplicity

# incoming angle of the THz beam in degrees
theta0 = 0.0
theta0 *= np.pi / 180  # convert to radians

# number to multiply noise waveforms by
scale_factor = 50

# thickness of the sample in mm
thickness = n_layers//2*(t_fiberglass+t_epoxy) + t_fiberglass

###################################################
# Begin simulation

# start be handling the reference waveform that is needed

# first thing to do is read in the reference waveform
ref_data = pd.read_csv(ref_file, delimiter='\t')
time = ref_data['Optical Delay/ps'].values
ref_amp = ref_data['Raw_Data/a.u.'].values

data_length = len(ref_amp)

# read in the noise data files
noise_amp = np.zeros((25, data_length))
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

omega = freq * 2 * np.pi * 1e12  # create omega array for plotting, convert from THz

# gate to remove the front artifact signal
gate = 400

# remove the false signal in the front that is always there
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

plt.figure('Sample with No Defects in Frequency Domain')
plt.plot(freq, np.abs(e1), 'r')
plt.title('Sample with No Defects in Frequency Domain')
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.xlim(0, 3.5)
plt.grid()

model_peaks = sm.peak_detect(return_amp, delta=0.015, t=time, dt=3, max_t=50)

plt.figure('Sample with No Defects in Time Domain')
plt.plot(time, return_amp, 'r')
plt.plot(model_peaks[:, 0], model_peaks[:, 1], 'g*')
plt.title('Sample with No Defects')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.ylim(-0.70, 0.55)
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

# run peak detection of model with flaws
peak_values = sm.peak_detect(return_amp_flawed, delta=0.015, t=time, dt=3, max_t=50)

plt.figure('Sample with Defects in Time Domain')
plt.plot(time, return_amp_flawed, 'r')
plt.plot(peak_values[:, 0], peak_values[:, 1], 'g*')
plt.title('Sample with Defects in Time Domain')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.ylim(-0.70, 0.55)
plt.grid()

# for i in range(1, len(peak_values)):
#     if peak_values[i, 1] > model_peaks[i, 1]:
#         print('Void in Epoxy Layer %d' % i)

# Start analysis of the system noise

plt.figure('Example of System Noise')
plt.plot(time, noise_amp[0], 'r', linewidth=0.5)
plt.title('System Noise')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

noise_mean = np.mean(noise_amp)

# estimate the standard deviation along each measurement
noise_std = np.std(noise_amp, axis=1)
noise_std = np.mean(noise_std)
noise_var = noise_std**2

# subtract the mean noise level from the system noise so we
# can determine AR1 bandwidth parameter
noise_amp -= noise_mean

# plot an example correlation from the first noise waveform
correlation = np.zeros(len(noise_amp[0])*2-1)
for i in range(n_noise_waveforms):
    correlation += np.correlate(noise_amp[i, :], noise_amp[i, :], mode='full')

correlation /= n_noise_waveforms

plt.figure('Average Noise Correlation')
plt.plot(correlation[correlation.argmax():], 'ro-')
plt.title('Noise Correlation Averaged')
plt.xlabel('Data Point')
plt.ylabel('Correlation')
plt.grid()

corr = np.correlate(noise_amp[0], noise_amp[0], mode='full')
plt.figure('Noise Correlation')
plt.plot(corr[corr.argmax():], 'ro-')
plt.title('Noise Correlation of a Single Measurement')
plt.xlabel('Point')
plt.ylabel('Correlation')
plt.grid()

# take a look at the psd by taking splitting noise into 50, 2048 point arrays
noise_reshaped = noise_amp.reshape((50, 2048)) * scale_factor
noise_psd = np.abs(np.fft.fft(noise_reshaped, n=len(noise_amp[0]), axis=1))**2
noise_psd = noise_psd.mean(axis=0)

N_hat_dB = 10*np.log10(noise_psd)

plt.figure('Noise PSD Estimate')
plt.plot(omega, N_hat_dB[:len(omega)], 'r', linewidth=0.5)  # only plot one side of spectrum
plt.title('True Noise PSD Estimate')
plt.xlabel(r'Frequency ($\omega$)')
plt.ylabel('Power (dB)')
plt.grid()

alpha = 0
for i in range(n_noise_waveforms):
    correlation = np.correlate(noise_amp[i, :], noise_amp[i, :], mode='full')
    alpha += correlation[data_length] / correlation.max()  # R(1) / R(0)

alpha /= n_noise_waveforms

# determine the variance of the white noise
var_w = noise_std**2 - alpha**2 * noise_std**2
sigma_w = np.sqrt(var_w)

noise_sim = np.zeros(data_length+500)

w = np.random.normal(0, sigma_w, data_length+500)

# determine the constant in the AR process
c = noise_mean - alpha*noise_mean

for i in range(1, data_length+500):
    noise_sim[i] = c + alpha*noise_sim[i-1] + w[i]

noise_sim = noise_sim[500:]  # throw away the first 500 points

plt.figure('AR1 Noise Simulation')
plt.plot(time, noise_sim, 'r', linewidth=0.5)
plt.title('AR1 Noise Simulation')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid()

noise_sim *= scale_factor

# measure signal from composite
signal_and_noise = return_amp_flawed + noise_amp[1, :]*scale_factor

# Plot an example of the measured signal with noise
plt.figure('Sample with Defects and Noise Time Domain')
plt.plot(time, signal_and_noise, 'r')
plt.title('Sample with Defects and Noise')
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
n_hat -= n_hat.mean()  # subtract mean from itself

n_hat2 = n_hat.reshape((50, 2048)) * scale_factor

N_hat = np.abs(np.fft.fft(n_hat2, nfft))**2

Sn_hat = np.mean(N_hat, axis=0)

Sn_hat_dB = 10*np.log10(Sn_hat)

plt.figure('Noise Simulation AR(1) PSD Estimate')
plt.plot(omega, Sn_hat_dB[:len(omega)], 'r', linewidth=0.5)  # plot one-sides spectrum
plt.title('AR(1) Noise Simulation PSD Estimate')
plt.xlabel(r'Frequency ($\omega$)')
plt.ylabel('Power (dB)')
plt.grid()

Ss = np.abs(np.fft.fft(return_amp))**2  # expected signal psd is fft of return amplitude
Ss_dB = np.log10(Ss)

H = Ss / (Ss + Sn_hat)

plt.figure('Wiener Filter')
plt.plot(omega, H[:len(omega)], 'r', linewidth=0.75)
plt.xlabel(r'Frequency ($\omega$)')
plt.ylabel('Filter Value')
plt.grid()

sx = return_amp_flawed + noise_amp[0] * scale_factor  # add the model with flaws to the noise

# now take signal estimate with true wiener filter
S_hat_estimate = H * np.fft.fft(sx)
s_estimate = np.fft.ifft(S_hat_estimate)

filtered_peaks = sm.peak_detect(s_estimate, delta=0.015, t=time, dt=3, min_t=14.5, max_t=50)

for i in range(1, len(model_peaks)):
    if filtered_peaks[i, 1] > model_peaks[i, 1]:
        print('Void in Epoxy Layer %d' % i)

plt.figure('Estimated Signal')
plt.plot(time, s_estimate, 'r')
# plt.plot(filtered_peaks[:, 0], filtered_peaks[:, 1], 'g*')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.ylim(-0.62, 0.45)
plt.grid()

# try and use Kalman filter to estimate ar1 parameter of actual system noise
Q = 0.1
xhat_old = 0.764
P_old = Q
F = 1
I = 1
z = noise_amp[0, :]

K = np.zeros(data_length)
x_hat = np.zeros(data_length)
for i in range(1, data_length):
    H = z[i-1]
    R = np.max([0, 1-xhat_old**2])
    Kk = P_old*H*(H*P_old*H+R)**-1
    K[i] = Kk
    xhatk = xhat_old + Kk*(z[i]-H*xhat_old)
    x_hat[i] = xhatk
    Pk = (I-Kk*H)*P_old
    xhat_old = F*xhatk
    P_old = F*Pk*F + Q

plt.figure()
plt.plot(x_hat)