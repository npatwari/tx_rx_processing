import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math




def coarse_freq_synch(signal_withfreqoff, sample_rate):
    M = 2
    signal_withfreqoff_powerM = signal_withfreqoff**2
    psd_sq = np.fft.fftshift(np.abs(np.fft.fft(signal_withfreqoff_powerM)))

    f_sq = np.linspace(-sample_rate/2.0, sample_rate/2.0, len(psd_sq))
    max_freq = f_sq[np.argmax(psd_sq)] #### THIS IS THE ESTIMATED OFFSET!!!

    # result_fft = np.fft.fftshift(np.abs(np.fft.fft(signal_withfreqoff, len(signal_withfreqoff))))     # result_magsq = np.square(np.abs(result))
    # freqs = np.fft.fftshift(np.fft.fftfreq(len(signal_withfreqoff), 1/sample_rate))
    # max_freq = f_sq[np.argmax(result_fft)] #### THIS IS THE ESTIMATED OFFSET!!!
    # print(max_freq, max_freqq, "\n", f_sq, "\n", freqs)
    # pdb.set_trace()
    
    Ts = 1/sample_rate # calc sample period
    t = np.arange(0, Ts*len(signal_withfreqoff), Ts) # create time vector
    samples_freqoff_coarsecorrected = signal_withfreqoff * np.exp(-1j*2*np.pi*max_freq*t/M) # NOT to the  samples_freqoff_sq but to the samples_freqoff

    return samples_freqoff_coarsecorrected




def time_synch_muller(samples):
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=np.complex)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    while i_out < len(samples) and i_in+16 < len(samples):
        out[i_out] = samples[i_in + int(mu)] # grab what we think is the "best" sample
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        sps = 8 #8 samples per symbol,
        mu += sps + 0.3*mm_val 
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    return out



# For QPSK
def phase_detector_4(sample):
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    err = a * sample.imag - b * sample.real
    print("err", err)
    return err


def fine_freq_synch_costas(samples_with_freqoffset, rate):
    N = len(samples_with_freqoffset)
    samples_fine_corrected = np.zeros(N, dtype=np.complex)

    phase = 0
    freq = 0

    bpsk = True
    alpha = 0.132 # adjust to make the feedback loop faster or slower (which impacts stability)
    beta = 0.00932 # adjust to make the feedback loop faster or slower (which impacts stability)

    freq_log = []

    for i in range(N):
        samples_fine_corrected[i] = samples_with_freqoffset[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset    
        
        if bpsk:
            error = np.real(samples_fine_corrected[i]) * np.imag(samples_fine_corrected[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
        else:
            error = phase_detector_4(samples_with_freqoffset[i])

        freq += (beta * error) # Advance the loop (recalc phase and freq offset)
        freq_log.append(freq * rate / (2*np.pi)) # convert from angular velocity to Hz for logging
        phase += freq + (alpha * error)

        while phase >= 2*np.pi: # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi

    return samples_fine_corrected
