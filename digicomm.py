#! /usr/bin/env python

#
# License: see LICENSE.md
#
# Copyright (C) 2021  Neal Patwari
#
# 
# Author: Neal Patwari, npatwari@wustl.edu
#
# Version History:
#
# Version 1.0:  Initial Release.  Feb 2021.  For Python 3.6.10
#

import sys
import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

# PURPOSE: Convert a text string to a stream of 1/0 bits.
# INPUT: string message
# OUTPUT: Numpy array of 1's and 0's.
def text2bits(message):

    # Initialize an empty list and iterate for each character.
    outlist = np.array([])
    for x in message:

        # ord(x) converts a character to int, 
        # "{:07b}".format() converts to a string with "1" and "0" for 7 bits
        # int(b) for b in converts each bit to a 1 or 0 integer
        # then we append to the outlist
        outlist = np.append(outlist, [int(b) for b in "{:07b}".format(ord(x))])

    return outlist


# PURPOSE: insert OS_RATE-1 zeros before each sample of the input 
#   in order to increase the sampling rate by a factor of OS_Rate
def oversample(x, OS_Rate):

    # Initialize output
    x_s = np.zeros(len(x)*OS_Rate)

    # Fill in one out of every OS_Rate samples with the input values
    x_s[range(OS_Rate-1, OS_Rate*len(x), OS_Rate)]= x

    return x_s


# PURPOSE:  Plot an eye diagram of a signal
#
# INPUTS:
#   y_s:    vector of signal samples out of the matched filter
#   N:      the number of samples per symbol.  Assumes that time 0 is 
#           at sample y_s(1).  If not, you must send in an offset integer.
#   offset: the number of samples at the start of y_s to ignore
#
def plot_eye_diagram(y_s, N, offset=0):

    # Each window should start N//2 before offset (+ integer*N)
    # and end at N//2 after offset.  We don't care much about the
    # very beginning and the very end.  The symbol period is N.
    start_indices = range( (N//2)+offset-1, len(y_s)-N-1, N )
    # What to plot on the time axis.
    time_vals     = np.arange(N+1)/N - 0.5

    for i in start_indices:
        plt.plot(time_vals, y_s[range(i, i+N+1)], 'b-', linewidth=2)
        plt.xlabel('Time t / T_s', fontsize=20)
        plt.ylabel('Matched Filter Output', fontsize=20)

#
# PURPOSE:  Convert a vector of (float) zeros and ones to a string (char array)
# INPUT:    Expects a row vector of zeros and ones, a multiple of 7 length
#   The most significant bit of each character is always first.
# OUTPUT:   A string
#
def binvector2str(binvector):

    totalbits = len(binvector)
    if ((totalbits%7) > 1e-6):
        sys.exit('Length of bit stream must be a multiple of 7 to convert to a string.  You must be missing a few bits or have a few extra.')
    
    # We are using 7 bits per character in ASCII encoding
    # Put 7 1/0s into each row of the row_per_char 2D array
    row_per_char = binvector.reshape( totalbits//7, 7)

    str_out = ''  # Initialize character vector
    for bit_ints in row_per_char:
        # Convert the vector of 1.0/0.0 into a string of '1' and '0'
        bitstring   = "".join(str(int(z)) for z in bit_ints)
        # Convert the string into an integer using base 2
        # then into an ascii character
        # Then add it to the end of str_out
        str_out    += chr(int( bitstring, 2))
    
    return str_out


# PURPOSE:  Square-root raised cosine (SRRC) filter design
# INPUT:    
#           alpha: roll-off factor between 0 and 1.
#                  it indicates how quickly its frequency content transitions from its max to zero.
#           SPS:   samples per symbol = sampling rate / symbol rate
#           span:  the number of symbols in the filter to the left and right of the center tap.
#		         the SRRC filter will have 2*span + 1 symbols in the impulse response.
# OUTPUT:   SRRC filter taps
#
def SRRC(alpha, SPS, span):
    if ((int(SPS) - SPS) > 1e-10):
        SPS = int(SPS)
    if ((int(span) - span) > 1e-10):
        span = int(span)
    if (alpha < 0):
       raise ValueError('SRRC design: alpha must be between 0 and 1')
    if (SPS <= 1):
        raise ValueError('SRRC design: SPS must be greater than 1')

    n = np.arange(-SPS*span, SPS*span+1)+1e-9

    h = (1/np.sqrt(SPS))*( (np.sin(np.pi*n*(1-alpha)/SPS)) + \
        (4*alpha*n/SPS)*(np.cos(np.pi*n*(1+alpha)/SPS)) ) /\
        ( (np.pi*n/SPS) * (1 - (4*alpha*n/SPS)**2) )
    return h

def DD_carrier_sync(z, M, BnTs, zeta=0.707, mod_type = 'MPSK', type = 0, open_loop = False):
    """
    z_prime,a_hat,e_phi = DD_carrier_sync(z,M,BnTs,zeta=0.707,type=0)
    Decision directed carrier phase tracking
    
           z = complex baseband PSK signal at one sample per symbol
           M = The PSK modulation order, i.e., 2, 8, or 8.
        BnTs = time bandwidth product of loop bandwidth and the symbol period,
               thus the loop bandwidth as a fraction of the symbol rate.
        zeta = loop damping factor
        type = Phase error detector type: 0 <> ML, 1 <> heuristic
    
     z_prime = phase rotation output (like soft symbol values)
       a_hat = the hard decision symbol values landing at the constellation
               values
       e_phi = the phase error e(k) into the loop filter

          Ns = Nominal number of samples per symbol (Ts/T) in the carrier 
               phase tracking loop, almost always 1
          Kp = The phase detector gain in the carrier phase tracking loop; 
               This value depends upon the algorithm type. For the ML scheme
               described at the end of notes Chapter 9, A = 1, K 1/sqrt(2),
               so Kp = sqrt(2).
    
    Mark Wickert July 2014
    Updated for improved MPSK performance April 2020
    Added experimental MQAM capability April 2020

    Motivated by code found in M. Rice, Digital Communications A Discrete-Time 
    Approach, Prentice Hall, New Jersey, 2009. (ISBN 978-0-13-030497-1).
    """
    Ns = 1
    z_prime = np.zeros_like(z)
    a_hat = np.zeros_like(z)
    e_phi = np.zeros(len(z))
    theta_h = np.zeros(len(z))
    theta_hat = 0

    # Tracking loop constants
    Kp = 2.7 # What is it for the different schemes and modes?
    K0 = 1 
    K1 = 4*zeta/(zeta + 1/(4*zeta))*BnTs/Ns/Kp/K0 # C.46 in the rice book
    K2 = 4/(zeta + 1/(4*zeta))**2*(BnTs/Ns)**2/Kp/K0
    
    # Initial condition
    vi = 0
    # Scaling for MQAM using signal power
    # and known relationship for QAM.
    if mod_type == 'MQAM':
        z_scale = np.std(z) * np.sqrt(3/(2*(M-1)))
        z = z/z_scale
    for nn in range(len(z)):
        # Multiply by the phase estimate exp(-j*theta_hat[n])
        z_prime[nn] = z[nn]*np.exp(-1j*theta_hat)
        if mod_type == 'MPSK':
            if M == 2:
                a_hat[nn] = np.sign(z_prime[nn].real) + 1j*0
            elif M == 4:
                a_hat[nn] = (np.sign(z_prime[nn].real) + \
                             1j*np.sign(z_prime[nn].imag))/np.sqrt(2)
            elif M > 4:
                # round to the nearest integer and fold to nonnegative
                # integers; detection into M-levels with thresholds at mid points.
                a_hat[nn] = np.mod((np.rint(np.angle(z_prime[nn])*M/2/np.pi)).astype(np.int),M)
                a_hat[nn] = np.exp(1j*2*np.pi*a_hat[nn]/M)
            else:
                print('M must be 2, 4, 8, etc.')
        elif mod_type == 'MQAM':
            # Scale adaptively assuming var(x_hat) is proportional to 
            if M ==2 or M == 4 or M == 16 or M == 64 or M == 256:
                x_m = np.sqrt(M)-1
                if M == 2: x_m = 1
                # Shift to quadrant one for hard decisions 
                a_hat_shift = (z_prime[nn] + x_m*(1+1j))/2
                # Soft IQ symbol values are converted to hard symbol decisions
                a_hat_shiftI = np.int16(np.clip(np.rint(a_hat_shift.real),0,x_m))
                a_hat_shiftQ = np.int16(np.clip(np.rint(a_hat_shift.imag),0,x_m))
                # Shift back to antipodal QAM
                a_hat[nn] = 2*(a_hat_shiftI + 1j*a_hat_shiftQ) - x_m*(1+1j)
            else:
                print('M must be 2, 4, 16, 64, or 256');
        if type == 0:
            # Maximum likelihood (ML) Rice
            e_phi[nn] = z_prime[nn].imag * a_hat[nn].real - \
                        z_prime[nn].real * a_hat[nn].imag
        elif type == 1:
            # Heuristic Rice
            e_phi[nn] = np.angle(z_prime[nn]) - np.angle(a_hat[nn])
            # Wrap the phase to [-pi,pi]  
            e_phi[nn] = np.angle(np.exp(1j*e_phi[nn]))
        elif type == 2:
            # Ouyang and Wang 2002 MQAM paper
            e_phi[nn] = np.imag(z_prime[nn]/a_hat[nn])
        else:
            print('Type must be 0 or 1')
        vp = K1*e_phi[nn]      # proportional component of loop filter
        vi = vi + K2*e_phi[nn] # integrator component of loop filter
        v = vp + vi        # loop filter output
        theta_hat = np.mod(theta_hat + v,2*np.pi)
        theta_h[nn] = theta_hat # phase track output array
        if open_loop:
            theta_hat = 0 # for open-loop testing
    
    # Normalize MQAM outputs
    if mod_type == 'MQAM': 
        z_prime *= z_scale
    return z_prime, a_hat, e_phi, theta_h

def Freq_Offset_Correction(samples, fs, order=2, debug=False):
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples**order)))
    f = np.linspace(-fs/2.0, fs/2.0, len(psd))

    max_freq = f[np.argmax(psd)]
    Ts = 1/fs # calc sample period
    t = np.arange(0, Ts*len(samples), Ts) # create time vector
    corrsamp = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0)
    
    if debug:
        print('offset estimate: {} KHz'.format(max_freq/2000) )
        plt.figure()
        plt.plot(f, psd, label='before correction')
        plt.plot(f, np.fft.fftshift(np.abs(np.fft.fft(corrsamp**2))), label='after correction')
        plt.legend()
        plt.show()

    return corrsamp

# PURPOSE:  Plot an eye diagram of a signal
#
# INPUTS:
#   y_s:    vector of signal samples out of the matched filter
#   N:      the number of samples per symbol.  Assumes that time 0 is 
#           at sample y_s[0].  If not, you must send in an offset integer.
#   offset: the number of samples at the start of y_s to ignore
#
# OUTPUTS:  none
def plot_eye_diagram(y_s, N, offset=0):

    start_indices = range( int(np.floor(N/2.0)) + offset - 1, len(y_s) - N, N)
    time_vals     = np.arange(-0.5, 0.5+1.0/N, 1.0/N)

    plt.figure()
    for i, start_i in enumerate(start_indices):
        plt.plot(time_vals, y_s[start_i:(start_i+N+1)], 'b-', linewidth=2)
        
    plt.xlabel(r'Time $t/T_s$', fontsize=16)
    plt.xlim([-0.5, 0.5])
    plt.ylabel('Matched Filter Output', fontsize=16)
    plt.grid(True)
    plt.show()


def write_complex_binary(data, filename):
    '''
    Open filename and write array to it as binary
    Format is interleaved float IQ e.g. each I,Q should be 32-bit float 
    INPUT
    ----
    data:     data to be wrote into the file. format: (length, )
    filename: file name
    '''

    re = np.real(data)
    im = np.imag(data)
    binary = np.zeros(len(data)*2, dtype=np.float32)
    binary[::2] = re
    binary[1::2] = im
    binary.tofile(filename)

def get_samps_frm_file(filename): 
    '''
    load samples from the binary file
    '''
    # File should be in GNURadio's format, i.e., interleaved I/Q samples as float32
    samples = np.fromfile(filename, dtype=np.float32)
    samps = (samples[::2] + 1j*samples[1::2]).astype((np.complex64)) # convert to IQIQIQ
        
    return samps