import json
import numpy as np
from scipy import signal
import h5py
import matplotlib.pyplot as plt
from matplotlib import rc
import datetime

rc('xtick', labelsize=14) 
rc('ytick', labelsize=14)


def get_time_string(timestamp):
    '''
    Helper function to get data and time from timestamp
    INPUT: timestamp
    OUTPUT: data and time. Example: 01-04-2023, 19:50:27
    '''
    date_time = datetime.datetime.fromtimestamp(int(timestamp))
    return date_time.strftime("%m-%d-%Y, %H:%M:%S")

def JsonLoad(folder, json_file):
    '''
    Load parameters from the saved json file
    INPUT
    ----
        folder: path to the measurement folder. Example: "SHOUT/Results/Shout_meas_01-04-2023_18-50-26"
        json_file: the json file with all the specifications. Example: '/save_iq_w_tx_gold.json'
    OUTPUT
    ----
        samps_per_chip: samples per chip
        wotxrepeat: number of repeating IQ sample collection w/o transmission. Used as an input to traverse_dataset() func
        rxrate: sampling rate at the receiver side
    '''
    config_file = folder+'/'+json_file
    config_dict = json.load(open(config_file))[0]
    nsamps = config_dict['nsamps']
    rxrate = config_dict['rxrate']
    rxfreq = config_dict['rxfreq']
    wotxrepeat = config_dict['wotxrepeat']
    rxrepeat = config_dict['rxrepeat']

    return rxrepeat, rxrate

def traverse_dataset(meas_folder):
    '''
    Load data from hdf5 format measurement file
    INPUT
    ----
        meas_folder: path to the measurement folder. Example: "SHOUT/Results/Shout_meas_01-04-2023_18-50-26"
    OUTPUT
    ----
        data: Collected IQ samples w/ transmission. It is indexed by the transmitter name
        noise: Collected IQ samples w/o transmission. It is indexed by the transmitter name
        txrxloc: transmitter and receiver names
    '''
    data = {}
    noise = {}
    txrxloc = {}

    dataset = h5py.File(meas_folder + '/measurements.hdf5', "r") #meas_folder
    print("Dataset meta data:", list(dataset.attrs.items()))
    for cmd in dataset.keys():
        print("Command:", cmd)
        if cmd == 'saveiq':
            cmd_time = list(dataset[cmd].keys())[0]
            print("  Timestamp:", get_time_string(cmd_time))
            print("  Command meta data:", list(dataset[cmd][cmd_time].attrs.items()))
            for rx_gain in dataset[cmd][cmd_time].keys():
                print("   RX gain:", rx_gain)
                for rx in dataset[cmd][cmd_time][rx_gain].keys():
                    print("     RX:", rx)
                    print("       Measurement items:", list(dataset[cmd][cmd_time][rx_gain][rx].keys()))
        elif cmd == 'saveiq_w_tx':
            cmd_time = list(dataset[cmd].keys())[0]
            print("  Timestamp:", get_time_string(cmd_time))
            print("  Command meta data:", list(dataset[cmd][cmd_time].attrs.items()))
            for tx in dataset[cmd][cmd_time].keys():
                print("   TX:", tx)
                
                if tx == 'wo_tx':
                    for rx_gain in dataset[cmd][cmd_time][tx].keys():
                        print("       RX gain:", rx_gain)
                        #print(dataset[cmd][cmd_time][tx][rx_gain].keys())
                        for rx in dataset[cmd][cmd_time][tx][rx_gain].keys():
                            print("         RX:", rx)
                            print("           Measurement items:", list(dataset[cmd][cmd_time][tx][rx_gain][rx].keys()))
                            repeat = np.shape(dataset[cmd][cmd_time][tx][rx_gain][rx]['rxsamples'])[0]
                            print("         repeat", repeat)

                            samplesNotx =  dataset[cmd][cmd_time][tx][rx_gain][rx]['rxsamples'][:repeat, :]
                            namelist = rx.split('-')
                            noise[namelist[1]] = samplesNotx
                else:
                    for tx_gain in dataset[cmd][cmd_time][tx].keys():
                        print("     TX gain:", tx_gain)
                        for rx_gain in dataset[cmd][cmd_time][tx][tx_gain].keys():
                            print("       RX gain:", rx_gain)
                            #print(dataset[cmd][cmd_time][tx][tx_gain][rx_gain].keys())
                            for rx in dataset[cmd][cmd_time][tx][tx_gain][rx_gain].keys():
                                print("         RX:", rx)
                                print("         Measurement items:", list(dataset[cmd][cmd_time][tx][tx_gain][rx_gain][rx].keys()))
                                print("         samples shape", np.shape(dataset[cmd][cmd_time][tx][tx_gain][rx_gain][rx]['rxsamples']))
                                # print("         rxloc", (dataset[cmd][cmd_time][tx][tx_gain][rx_gain][rx]['rxloc'][0]))
                                repeat = np.shape(dataset[cmd][cmd_time][tx][tx_gain][rx_gain][rx]['rxsamples'])[0]
                                print("         repeat", repeat)

                                # peak avg check
                                txrxloc.setdefault(tx, []).append([rx]*repeat)
                                rxsamples = dataset[cmd][cmd_time][tx][tx_gain][rx_gain][rx]['rxsamples'][:repeat, :]
                                data.setdefault(tx, []).append(np.array(rxsamples))

        else:                       
            print('Unsupported command: ', cmd)

    return data, noise, txrxloc

# PURPOSE: convert input data stream to signal space values for
#          a particular modulation type (as specified by the inputVec
#          and outputVec).
# INPUT: data (groups of bits)
# OUTPUT: signal space values
def lut(data, inputVec, outputVec):
    if len(inputVec) != len(outputVec):
        print('Input and Output vectors must have identical length')
    # Initialize output
    output = np.zeros(data.shape)
    # For each possible data value
    eps = np.finfo('float').eps
    for i in range(len(inputVec)):
        # Find the indices where data is equal to that input value
        for k in range(len(data)):
            if abs(data[k]-inputVec[i]) < eps:
                # Set those indices in the output to be the appropriate output value.
                output[k] = outputVec[i]
    return output

def oversample(x, OS_Rate):
    # Initialize output
    x_s = np.zeros(len(x)*OS_Rate)
    x_s[::OS_Rate] = x
    return x_s

def SRRC(alpha, N, Lp):
    # Add epsilon to the n values to avoid numerical problems
    n = np.arange(-N*Lp+ (1e-9), N*Lp+1)
    h = np.zeros(len(n))
    coeff = 1/np.sqrt(N)
    for i, each in enumerate(n):
        sine_term = np.sin(np.pi * each * (1-alpha) / N)
        cosine_term = np.cos(np.pi * each * (1+alpha) / N)
        cosine_coeff = 4 * alpha * each / N
        numerator = sine_term + (cosine_coeff * cosine_term)
        denom_coeff = np.pi * each / N
        denom_part = 1 - cosine_coeff**2
        denominator = denom_coeff * denom_part
        h[i] = coeff * numerator / denominator

    return h

# PURPOSE: Plot an eye diagram of a signal
# INPUT: y_s: vector of signal samples out of the matched filter
#        N: the number of samples per symbol. Assumes that time 0 is at sample
#        y_s[0]. If not, you must send in an offset integer.
# OUTPUT: none
def plot_eye_diagram(y_s, N, offset=0):
    start_indices = range(int(np.floor(N/2.0)) + offset - 1, len(y_s) - N, N)
    time_vals     = np.arange(-0.5, 0.5+1.0/N, 1.0/N)

    plt.figure()
    for i, start_i in enumerate(start_indices):
        plt.plot(time_vals, y_s[start_i:(start_i+N+1)], 'b-', linewidth=2)
        
    plt.xlabel(r'Time $t/T_s$', fontsize=16)
    plt.xlim([-0.5, 0.5])
    plt.ylabel('Matched Filter Output', fontsize=16)
    plt.grid(True)
    plt.show()

# PURPOSE: Convert binary data to M-ary by making groups of log2(M)
#          bits and converting each bit to one M-ary digit.
# INPUT: Binary digit vector, with length as a multiple of log2(M)
# OUTPUT: M-ary digit vector
def binary2mary(data, M):

    log2M   = round(np.log2(M))
    # integer number of bits per group
    if (len(data) % log2M) != 0:
        print('Input to binary2mary must be divisible by log2(m).')
    data.shape = (len(data)//log2M, log2M)
    binaryValuesArray = 2**np.arange(log2M)
    marydata = data.dot(binaryValuesArray)
    return marydata


# Purpose: Convert M-ary data to binary data
#          each m-ary value input in "data" is converted to
#          log2(M) binary values.
# INPUT: M-ary digit vector
# OUTPUT: Binary digit vector, with length equal to the number
#         of values in data multiplied by log2(M)
def mary2binary(data, M):
    length = len(data) # number of values in data
    log2M = round(np.log2(M)) # integer number of bits per data value
    format_string = '0' + str(log2M) + 'b'
    binarydata = np.zeros((1,length*log2M))
    count = 0
    for each in data:
        binval = format(int(each), format_string)
        for i in range(log2M):
            binarydata[0][count+i] = int(binval[i])
        count = count + log2M
    return binarydata


def createPreambleSignal(A, N, alpha, Lp):

    # We defined the preamble as this repeating bit signal:
    preamble     = np.tile([1, 1, 0, 0], 16)

    ###########################################
    ### Signal Generation
    ### INPUT: binary data
    ### OUTPUT: 4-ary data (0..3) values
    data = binary2mary(preamble, 4)

    ###########################################
    ### Modulation
    ### INPUT: data
    ### OUTPUT: modulated values, x
    inputVec   = [0, 1, 2, 3]
    outputVecI = [A, -A, A, -A]
    outputVecQ = [A, A, -A, -A]
    xI         = lut(data, inputVec, outputVecI)
    xQ         = lut(data, inputVec, outputVecQ)

    ###########################################
    ### Upsample
    ### INPUT: modulated values, x
    ### OUTPUT: modulated values at sampling rate, x_s
    x_s_I = oversample(xI, N)
    x_s_Q = oversample(xQ, N)

    ###########################################
    ### Pulse-shape filter
    ### INPUT: modulated values at sampling rate, x_s
    ### OUTPUT: baseband transmit signal s
    pulse = SRRC(alpha, N, Lp)
    s_0_I = np.convolve(x_s_I, pulse, mode='full')
    s_0_Q = np.convolve(x_s_Q, pulse, mode='full')

    return (s_0_I + 1j*s_0_Q), pulse

# Plot the dataset
def plotAllLinks(rx_data, txrxloc):

    # rx_data is a dict indexed by transmitter name
    for txloc in rx_data:
        # check one specific transmitter
        # if txloc != 'cbrssdr1-hospital-comp':continue
        print('TX: {}'.format(txloc))
        rx_data[txloc] = np.vstack(rx_data[txloc])
        print('[Debug] data shape', np.shape(rx_data[txloc]))
        txrxloc[txloc] = np.vstack(txrxloc[txloc]).flatten()
            
        # measurements vs. distance
        # plt.figure()
        for j, rxloc in enumerate(txrxloc[txloc][::rxrepeat]):
            # check one specific receiver
            # if rxloc != 'cnode-wasatch-dd-b210':continue
            print('RX: {}'.format(rxloc))

            namelist = (rxloc.split('-'))
            TXnamelist = (txloc.split('-'))
                
            # custom processing functions implemented here
            plt.figure(figsize=(5,4))
            plt.plot(np.abs(rx_data[txloc][j,:]))
            plt.title('TX: {} RX: {}'.format(TXnamelist[1], namelist[1]))
            plt.ylabel('ampltiude')
            plt.xlabel('Sample Index')
            plt.tight_layout()

            plt.figure(figsize=(5,4))
            plt.plot(np.angle(rx_data[txloc][j,:]))
            plt.ylabel('angle')
            plt.xlabel('Sample Index')
            plt.title('TX: {} RX: {}'.format(TXnamelist[1], namelist[1]))
            plt.tight_layout()

            plt.figure(figsize=(5,4))
            plt.psd(rx_data[txloc][j,:], Fs = 240)
            plt.title('TX: {} RX: {}'.format(TXnamelist[1], namelist[1]))
            plt.xlabel('Frequency (KHz)')
            plt.tight_layout()
    plt.show()


# PURPOSE: perform frequency offset estimation and correction.
#          Uses the (complex-valued) preamble signal. The product of 
#          the preamble signal and the received signal (at the time
#          when the preamble is received) has a frequency component near
#          zero at the frequency offset.  Find it from the max of the DFT.
#          We need a very fine resolution on that frequency, so we don't
#          use the FFT, we calculate it from the DFT definition.
# INPUT:   rx0: received signal (with a frequency offset)
#          preambleSignal: complex, known, transmitted preamble signal 
#          lagIndex: the index of rx0 where the preamble signal has highest 
#              cross-correlation
# OUTPUT:  rx1: Frequency-corrected received signal
#          frequencyOffset
#
def estimateFrequencyOffset(rx0, preambleSignal, lagIndex):

    # Estimate a frequency offset using the known preamble signal
    #f_hat  = estimateFrequencyOffset(x0, preambleSignal, lagIndex)
    rx0_part = np.conjugate(rx0[lagIndex:(lagIndex + len(preambleSignal))])
    N        = len(preambleSignal)
    prod_rx0_preamble = rx0_part*preambleSignal

    # Frequencies at which freq content is calc'ed.
    # We'll multiply the generated matrix by the data to calculate PSD
    # frequencies are normalized to sampling rate
    # MUST BE SET BY USER.  For POWDER with frequency synched nodes, we
    # expect at most 200e-9 frequency offset, which at center frequency of 3.5 GHz
    # and 240k sample rate, is 3.5e9 * 200e-9 / 240e3 = 0.003.  But we can
    # be conservative and make it larger, no problem.  We want to get the offset
    # down to at most 5 Hz b/c the packet duration is about 20 ms, so that would
    # keep the drift to about 1/10 of a rotation over the whole packet.
    maxFreqOffset   = 0.01000
    deltaFreqOffset = 0.00002
    freqRange    = np.arange(-maxFreqOffset, maxFreqOffset, deltaFreqOffset)
    temp         = (-1j*2*np.pi) * freqRange
    expMat       = np.transpose(np.array([np.exp(temp*i) for i in np.arange(N)]))
    PSD_prod     = np.abs(expMat.dot(prod_rx0_preamble))**2
    maxIndexPSD  = np.argmax(PSD_prod)
    maxIndexFreq = freqRange[maxIndexPSD]

    print('Frequency offset estimate: ' + str(maxIndexFreq*samp_rate) + ' Hz')

    # Do frequency correction on the input signal
    expTerm = np.exp((1j*2*np.pi * maxIndexFreq) * np.arange(len(rx0)))
    rx1 = expTerm * rx0

    return rx1, maxIndexFreq


# PURPOSE: perform preamble synchronization
#          Uses the (complex-valued) preamble signal. The cross-correlation 
#          of the preamble signal and the received signal (at the time
#          when the preamble is received) should have highest magnitude
#          at the index delay where the preamble approximately starts.  
# INPUT:   rx0: received signal (with a frequency offset)
#          preambleSignal: complex, known, transmitted preamble signal 
# OUTPUT:  lagIndex: the index of rx0 where the preamble signal has highest 
#              cross-correlation
#
def crossCorrelationMax(rx0, preambleSignal):

    # Cross correlate with the preamble to find it in the noisy signal
    lags      = signal.correlation_lags(len(rx0), len(preambleSignal), mode='same')
    xcorr_out = signal.correlate(rx0, preambleSignal, mode='same')
    xcorr_mag = np.abs(xcorr_out)
    # There may be two preambles because the packet repeats. 
    # Pick only a peak in the first half
    maxIndex = np.argmax(xcorr_mag[:len(xcorr_mag)//2])
    lagIndex = lags[maxIndex]

    print('Max crosscorrelation with preamble at lag ' + str(lagIndex))

    # Plot the selected signal.
    plt.figure()
    plt.plot(np.imag(rx0), label='RX Signal')
    plt.plot(range(lagIndex, lagIndex + len(preambleSignal)), 0.004*np.real(preambleSignal), label='Preamble')
    plt.legend()
    plt.ylabel('Real Signal Value', fontsize=14)
    plt.xlabel('Sample Index', fontsize=14)
    plt.tight_layout()

    return lagIndex


# PURPOSE: Plot an eye diagram of a signal
# INPUT: y_s: vector of signal samples out of the matched filter
#        N: the number of samples per symbol. Assumes that time 0 is at sample
#        y_s[0]. If not, you must send in an offset integer.
# OUTPUT: none
def plot_eye_diagram(y_s, N, offset=0):
    start_indices = range(int(np.floor(N/2.0)) + offset - 1, len(y_s) - N, N)
    time_vals     = np.arange(-0.5, 0.5+1.0/N, 1.0/N)

    plt.figure()
    for i, start_i in enumerate(start_indices):
        plt.plot(time_vals, y_s[start_i:(start_i+N+1)], 'b-', linewidth=2)
        
    plt.xlabel(r'Time $t/T_s$', fontsize=16)
    plt.xlim([-0.5, 0.5])
    plt.ylabel('Matched Filter Output', fontsize=16)
    plt.grid(True)
    plt.show()

# PURPOSE: Find the symbols which are closest in the complex plane 
#          to the measured complex received signal values.
# INPUT:   Received r_hat values (output of matched filter downsampled),
#          and possible signal space complex values. 
# OUTPUT:  m-ary symbol indices in 0...length(outputVec)-1
def findClosestComplex(r_hat, outputVec):
    data_out = np.array([np.argmin(np.abs(r-outputVec)) for r in r_hat])
    return data_out

def plot_constellation_diagram(rx4):

    plt.figure(figsize=(5,5))
    ax = plt.gca() 
    ax.set_aspect(1.0) 
    plt.plot(np.real(rx4), np.imag(rx4),'ro', label='RX Signal')
    plt.ylabel('Imag(Symbol Sample)', fontsize=14)
    plt.xlabel('Real(Symbol Sample)', fontsize=14)
    plt.grid('on')
    plt.tight_layout()




# MAIN
plt.ion()

# load parameters from the json script
folder = "Shout_meas/Shout_meas_01-11-2023_21-06-07" 
jsonfile = 'save_iq_w_tx_file.json'
rxrepeat, samp_rate = JsonLoad(folder, jsonfile)

# load data from the .json, save IQ sample arrays and tx/rx names
rx_data, _, txrxloc = traverse_dataset(folder)

# Pick one received signal to demodulate
txloc = 'cbrssdr1-hospital-comp'
rxloc = 'cbrssdr1-honors-comp'

# The dictionary element is a list with one element.
# Then, there are multiple received signals, pick one
rx0 = rx_data[txloc][0][1]   


#### Synchronization
A         = np.sqrt(9/2)
N         = 8   # Samples per symbol
alpha     = 0.5 # SRRC rolloff factor
Lp        = 6   # Number of symbol durations on each side of peak of SRRC pulse
preambleSignal, pulse = createPreambleSignal(A, N, alpha, Lp)
lagIndex = crossCorrelationMax(rx0, preambleSignal)
rx1, freqOffsetEst = estimateFrequencyOffset(rx0, preambleSignal, lagIndex)

# Plot the frequency corrected signal
plt.figure()
plt.plot(np.imag(rx1), label='RX Signal')
plt.title('TX: {} RX: {}'.format(txloc.split('-')[1], rxloc.split('-')[1]), fontsize=14)
plt.ylabel('Imag', fontsize=14)
plt.xlabel('Sample Index', fontsize=14)
plt.tight_layout()

###########################################
# Matched filter
# INPUT: frequency synchronized received signal rx1
# OUTPUT: matched-filtered signal rx2
rx2 = signal.lfilter(pulse, 1, rx1)

# Plot the matched filter output in an eye diagram (looking at each symbol period)
preambleStart = lagIndex + Lp*N*2  # There's also the delay b/c the pulse is this long.
plot_eye_diagram(np.imag(rx2), N, offset=preambleStart)

###########################################
# Downsample
# INPUT: Synched matched filter output
# OUTPUT: Symbol Samples (at n*T_sy)
rx3 = rx2[preambleStart::N]

# Ignore initial samples that are very close to the origin, compared to later samples.
rx3 = rx3 / np.median(np.abs(rx3))  # "AGC" to make symbol values close to +/- 1
magThreshold = 0.2
startsymbol = np.where(np.abs(rx3)>magThreshold)[0][0]
rx4 = rx3[startsymbol:]

# Plot a constellation diagram
plot_constellation_diagram(rx4)

###########################################
# Symbol Decisions
# INPUT: Symbol Samples
# OUTPUT: Bits
outputVec = np.array([1+1j, -1+1j, 1-1j, -1-1j])
mary_out  = findClosestComplex(r_hat, outputVec)
bits_out  = mary2binary(mary_out, 4)


###########################################
# Sync Word Discovery
# INPUT: Bit estimates from the received signal. 
#        Must have sync word used at the transmitter.
# OUTPUT: Bits from the data

# You have to know these things about the packet you are receiving
sync         = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
numDataBits  = 686  
