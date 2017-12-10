"""
Module that contains functions used in signal modelling codes
"""
import pdb

import numpy as np


def reflection_coefficient(n1, n2, theta1=0.0, theta2=0.0):
    """
    Determine the reflection coefficient of a media transition with parallel polarized light
    :param n1: the refractive index of the media in which coming from
    :param n2: the refractive index of the media in which going to
    :param theta1: the angle of the incident ray, in radians
    :param theta2: the angle of the transmitted ray, in radians
    :return: The reflection coefficient
    """
    num = n1*np.cos(theta2) - n2*np.cos(theta1)
    denom = n1*np.cos(theta2) + n2*np.cos(theta1)
    return num / denom


def transmission_coefficient(n1, n2, theta1=0, theta2=0):
    """
    Determine the transmission coefficient of a media transmission, independent of polarization
    :param n1: the refractive index of the media in which coming from
    :param n2: the refractive index of the media in which going to
    :param theta1: the angle of the incident ray, in radians
    :param theta2: the angle of the transmitted ray, in radians
    :return: The transmission coefficient
    """
    return 2*n1*np.cos(theta1) / (n1*np.cos(theta2) + n2*np.cos(theta1))


def get_theta_out(n0, n1, theta0):
    """
    Uses Snell's law to calculate the outgoing angle of light
    :param n0: The index of refraction of the incident media
    :param n1: The index of refraction of the outgoing media
    :param theta0: The angle of the incident ray in radians
    :return: theta1: The angle of the outgoing ray in radians
    """
    # make sure that we do float division
    if type(n1) is int:
        n1 = float(n1)

    return np.arcsin(n0/n1 * np.sin(theta0))


def global_reflection_model(n, theta, freq, d, n_layers, coverage, c=0.2998):
    """
    Calculates the global reflection coefficient given in Orfanidis
    "Electromagnetic Waves & Antennas". The global reflection coefficient is used to solve
    multilayer problems.
    :param n: The index of refraction of each of the layers, including the two half space media on
                either side, expected to be len(n_layers + 2)
    :param d: The thickness of each of the slabs, should be of length n_layer
    :param theta: The angle of the beam in each material including the two media on either side,
                length is n_layers+2
    :param freq: An array of frequencies over which to calculate the coefficient
    :param n_layers: The number of layers in the structure
    :param coverage: Percentage of the beam path occupied by the void
    :param c: The speed of light (default = 0.2998 mm/ps)
    :return: The global reflection coefficient over the supplied frequency range
    """
    try:
        r = np.zeros((n_layers+1, len(freq)), dtype=complex)
        gamma = np.zeros((n_layers+1, len(freq)), dtype=complex)
    except TypeError:
        r = np.zeros(n_layers+1, dtype=complex)
        gamma = np.zeros(n_layers+1, dtype=complex)

    for i in range(n_layers + 1):
        # determine the local reflection
        r[i] = reflection_coefficient(n[i], n[i+1], theta[i], theta[i+1])

        if i == n_layers:
            continue

        if coverage[i] != 0:
            # if there is a void at that layer, make reflection coefficient a weighted average
            # between fiberglass-epoxy interface and fiberglass-air interface

            # reflection coefficient of the void (air, n = 1)
            r_void = reflection_coefficient(n[i], 1.0, theta[i], theta[i+1])
            r[i] = r[i]*(1-coverage[i]) + r_void*coverage[i]
            continue

        if coverage[i-1] != 0:
            # if there was a void in the previous layer we need to account for the reflection as
            # the THz beam leaves the void
            r_void = reflection_coefficient(1.0, n[i], theta[i], theta[i+1])
            r[i] = r[i]*(1-coverage[i]) + r_void*coverage[i]

    # define the last global reflection coefficient as the local reflection coefficient
    gamma[-1, :] = r[-1]

    # calculate global reflection coefficients recursively
    for i in range(n_layers - 1, -1, -1):
        # delta is Orfanidis eq. 8.1.2, with cosine
        delta = 2*np.pi * freq/c * d[i]*n[i+1]*np.cos(theta[i+1])
        z = np.exp(-2j * delta)
        gamma[i, :] = (r[i] + gamma[i+1, :]*z) / (1+r[i]*gamma[i+1, :]*z)

    return gamma


def peak_detect(data, delta, t=None, dt=0, min_t=-np.inf, max_t=np.inf):
    """
    Detect max peaks in a vector
    :param data: the y values
    :param delta: if the value preceding a higher value is less than the higher value by delta,
        the preceding value is declared a peak
    :param t: if t is passed to function the peak locations will be replaced with appropriate
        locations from this array
    :param dt: time difference expected between peaks, if a peak occurs but at spacing between
        previous peak is smaller than dt, it will be ignored
    :param min_t: algorithm will not look for peaks until this time value is passed
    :param max_t: maximum time to consider looking for peaks. If this is passed function will
        end, once it reaches this value
    :return: array with locations in first column and values in second
    """

    if t is None:
        t = np.arange(len(data))

    max_val = -np.inf

    max_pos = np.nan

    max_tab = list()

    old_max_pos = 0.0

    for i in range(len(data)):
        cur_val = data[i]
        cur_pos = t[i]

        if cur_pos < min_t:
            continue

        if cur_pos > max_t:  # if we move past max_t: end function
            return np.array(max_tab)

        if cur_pos - old_max_pos < dt:
            max_val = -np.inf
            continue

        if cur_val > max_val:
            max_val = cur_val
            max_pos = t[i]

        if cur_val < max_val-delta:
            # pdb.set_trace()
            max_tab.append((max_pos, max_val))
            old_max_pos = max_pos

            max_val = cur_val
            max_pos = cur_pos

    return np.array(max_tab)

