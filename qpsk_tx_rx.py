# Mobile & Wireless Week, Spring 2023
# Author: Neal Patwari
# Translated from MATLAB
# Translator: Cassie Jeng

###########################################
# Libraries:
import math
import numpy as np
import matplotlib as plt
###########################################

###########################################
# Functions:

# PURPOSE: Convert a text string to a stream of 1/0 bits.
# INPUT: string message
# OUTPUT: vector of 1's and 0's.
def text2bits(message):
    # Convert to characters of '1' and '0' in a vector.
    temp_message = []
    final_message = []
    for each in message:
        temp_message.append(format(ord(each), '07b'))
    for every in temp_message:
        for digit in every:
            final_message.append(int(digit))
    return final_message

# PURPOSE: Convert binary data to M-ary by making groups of log2(M)
#          bits and converting each bit to one M-ary digit.
# INPUT: Binary digit vector, with length as a multiple of log2(M)
# OUTPUT: M-ary digit vector
def binary2mary(data, M):
    length     = len(data)
    log2M   = round(math.log2(M)) # integer number of bits per group
    if (length % log2M) != 0:
        print('Input to binary2mary must be divisible by log2(m).')
    binvalues = np.zeros((log2M,1))
    values = []
    newdata = []
    start = log2M-1
    i = 0
    while start >= 0:
        binvalues[i] = int(math.pow(2,start))
        start=start-1
        i = i + 1
    print(binvalues)
    for each in data:
        newdata.append(int(each))
    newdata = np.array(newdata)
    temp = np.reshape(newdata, (log2M, int(length/log2M)))
    print(temp.shape)
    temp = np.transpose(temp)
    print(temp)
    marydata = temp.dot(binvalues)
    marydata = np.transpose(marydata)
    print(marydata)
    return marydata

# PURPOSE: convert input data stream to signal space values for
#          a particular modulation type (as specified by the inputVec
#          and outputVec).
# INPUT: data (groups of bits)
# OUTPUT: signal space values
def lut(data, inputVec, outputVec):
    if len(inputVec) != len(outputVec):
        print('Input and Output vectors must have identical length')
    # Initialize output
    data = np.array(data)
    data_shape = data.shape
    output = np.zeros(data_shape)
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
    length = len(x)
    x_s = np.zeros((1,length*OS_Rate))
    x_s = np.transpose(x_s)
    # Fill in one out of every OS_Rate samples with the input values
    count = 0
    h = 0
    for k in range(len(x_s)):
        count = count + 1
        if count == 8:
            x_s[k] = x[h]
            count = 0
            h = h + 1
    return x_s

def SRRC(alpha, N, Lp):
    # Add epsilon to the n values to avoid numerical problems
    ntemp = list(range(-N*Lp, N*Lp+1))
    n = []
    for each in ntemp:
        n.append(each + math.pow(10,-9)) # UP TO HERE WORKS
    # Plug into time domain formula for the SRRC pulse shape
    h = []
    coeff = 1/math.sqrt(N)
    for each in n:
        sine_term = math.sin(math.pi * each * (1-alpha) / N)
        cosine_term = math.cos(math.pi * each * (1+alpha) / N)
        cosine_coeff = 4 * alpha * each / N
        numerator = sine_term + (cosine_coeff * cosine_term)
        denom_coeff = math.pi * each / N
        denom_part = 1 - math.pow(cosine_coeff, 2)
        denominator = denom_coeff * denom_part
        pulse = coeff * numerator / denominator
        h.append(pulse)
    return h

# Purpose: Find the symbols which are closest in symbol space
#          to the received signal space values.
# INPUT: Received r hat values (output of matched filter),
#        and possible signal space values. Assumes signal space
#        estimate vectors and signal space vectors are in the columns
#        of matrices r_hat and outputVec
# OUTPUT: m-ary symbol indices in 0...length(outputVec)-1
def findClosestMary(r_hat, outputVec):
    symbols = len(outputVec[0])
    data_out = np.ones((1, len(r_hat)))
    # For each r_hat value
    for i in range(1, len(r_hat[0])):
        # Find the index of the symbol space value with the
        # lowest distance to this r_hat.
        distances = r_hat[:,i]*np.ones((1,symbols))
        distances = np.subtract(distances, outputVec)
        distances = np.square(distances)
        distances = sum(distances) # sum values in each column
        min_val = min(distances)
        min_ind = np.where(distances == min_val)
        # Output values 0, 1, ..., length(outputVec)-1
        data_out[i] = min_ind[0][0]
    return data_out

# Purpose: Convert M-ary data to binary data
#          each m-ary value input in "data" is converted to
#          log2(M) binary values.
# INPUT: M-ary digit vector
# OUTPUT: Binary digit vector, with length equal to the number
#         of values in data multiplied by log2(M)
def mary2binary(data, M):
    length = len(data) # number of values in data
    log2M = round(math.log2(M)) # integer number of bits per data value
    format_string = '0' + str(log2M) + 'b'
    binarydata = np.zeros((1,length*log2M))
    count = 0
    for each in data:
        binval = format(each, format_string)
        for i in range(log2M):
            binarydata[0][count+i] = int(binval[i])
        count = count + log2M
    return binarydata

# PURPOSE: Convert a vector of zeros and ones to a string (char array)
# INPUT: Expects a row vector of zeros and ones, a multiple of 7 length
#        The most significant bit of each character is always first.
# OUTPUT: A string
def binvector2str(binvector):
    binvector = binvector[0]
    length = len(binvector)
    eps = np.finfo('float').eps
    if abs(length/7 - round(length/7)) > eps:
        print('Length of bit stream must be a multiple of 7 to convert to a string.')
    # Each character requires 7 bits in standard ASCII
    num_characters = round(length/7)
    # Maximum value is first in the vector. Otherwise would use 0:1:length-1
    start = 6
    bin_values = []
    while start >= 0:
        bin_values.append(int(math.pow(2,start)))
        start = start - 1
    bin_values = np.array(bin_values)
    bin_values = np.transpose(bin_values)
    str_out = '' # Initialize character vector
    for i in range(num_characters):
        single_char = temp[i*7:i*7+7]
        value = 0
        for counter in range(len(single_char)):
            value = value + int(single_char[counter]) * int(bin_values[counter])
        str_out[i] = chr(int(value))
    return str_out
###########################################

###########################################

# QPSK transmitter

###########################################

# Signal Generation
# INPUT: none
# OUTPUT: binary data

if 1:
    temp         =  'You Are Strong&Resilient'
    data_bits    = text2bits(temp)
    print('Message Sent: ')
    print(temp)

    ###########################################
    ### Signal Generation
    ### INPUT: binary data
    ### OUTPUT: 4-ary data (0..3) values

    data = binary2mary(data_bits, 4)
    print(data)

    ###########################################
    ### Modulation
    ### INPUT: data
    ### OUTPUT: modulated values, x

    A = math.sqrt(9/2)
    A = int(A)
    inputVec   = [0, 1, 2, 3]
    outputVecI = [A, -A, A, -A]
    outputVecQ = [A, A, -A, -A]
    xI         = lut(data, inputVec, outputVecI)
    xQ         = lut(data, inputVec, outputVecQ)

    ###########################################
    ### Upsample
    ### INPUT: modulated values, x
    ### OUTPUT: modulated values at sampling rate, x_s

    x_s_I = oversample(xI,8)
    x_s_Q = oversample(xQ,8)

    ###########################################
    ### Pulse-shape filter
    ### INPUT: modulated values at sampling rate, x_s
    ### OUTPUT: baseband transmit signal s

    pulse = SRRC(0.5, 8, 6)
    s_0_I = np.convolve(x_s_I, pulse, mode='full')
    s_0_Q = np.convolve(x_s_Q, pulse, mode='full')

    ###########################################
    ### Up-convert
    ### INPUT: s, baseband signal
    ### OUTPUT: up_s, bandpass signal

    f_0 = 0.25
    n = list(range(0, len(s_0_I)-1))
    s_0_I = np.array(s_0_I)
    s_0_Q = np.array(s_0_Q)
    stemp = []
    s = []
    for each in n:
        val1 = math.sqrt(2) * s_0_I[each] * math.cos(2 * math.pi * f_0 * each)
        val2 = math.sqrt(2) * s_0_Q[each] * math.sin(2 * math.pi * f_0 * each)
        stemp.append(val1 - val2)
    randn = np.random.normal(loc=0.0, scale=1.0, size=stemp.shape)
    randnn = []
    for each in randn:
        randnn.append(0.2 * each)
    randnn = np.array(randnn)
    stemp = np.array(stemp)
    for i in range(len(stemp)-1):
        s.append(stemp[i] + randnn[i])

###########################################
# Load from one of M. Rice's mat files. (in MATLAB version)
# load qpskdata_Rice.mat
# s = qpskdata[2:]

###########################################
# Plot for project handout

s = np.array(s)
plt.figure()
plt.plot(s, marker = 'o', linestyle = 'dashed')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.grid()
plt.show()

###########################################

###########################################

# QPSK receiver

###########################################
# Down-convert
# INPUT: up_s, bandpass signal
# OUTPUT: s, baseband signal

f_0 = 0.2499
n = list(range(0, len(s)-1))
s_rx_I = []
s_rx_Q = []
for each in n:
    I_part = math.sqrt(2) * s[each] * math.cos(2 * math.pi * f_0 * each)
    Q_part = -1 * math.sqrt(2) * s[each] * math.sin(2 * math.pi * f_0 * each)
    s_rx_I.append(I_part)
    s_rx_Q.append(Q_part)

###########################################
# Matched filter
# INPUT: baseband transmitted signal s
# OUTPUT: matched-filtered signal y
pulse = SRRC(0.5, 8, 6)
yI = filter(pulse, 1, s_rx_I)
yQ = filter(pulse, 1, s_rx_Q)

###########################################
# Plot for project handout

plt.figure()
plt.plot(yI, marker = 'o', linestyle = 'dashed')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.grid()
plt.show()

plt.figure()
plt.plot(yQ, marker = 'o', linestyle = 'dashed')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.grid()
plt.show()

###########################################
# Time Synch
# INPUT: Matched Filter output
# OUTPUT: Synched MF output with samples at US_Rate, 2*US_Rate, ...

y_s_I = yI[97:]
y_s_Q = yQ[97:]

# Plot eye-diagram
# Skipped for Python Version -- no found MATLAB code for function plot_eye_diagram in repository

###########################################
# Downsample
# INPUT: Synched matched filter output
# OUTPUT: Symbol Samples (at n*T_sy)

US_Rate = 8
length = len(y_s_I)
r_hat_I = []
count = 0
for k in range(length):
	count = count + 1
	if count == US_Rate:
		r_hat_I.append(y_s_I[k])
		count = 0
r_hat_Q = []
count = 0
for k in range(length):
	count = count + 1
	if count == US_Rate:
		r_hat_Q.append(y_s_Q[k])
		count = 0
expected_symbols = 84
if len(r_hat_I) > expected_symbols:
    r_hat_I = r_hat_I[:expected_symbols]
    r_hat_Q = r_hat_Q[:expected_symbols]

###########################################
# Symbol Decisions
# INPUT: Symbol Samples
# OUTPUT: Bits
A = math.sqrt(9/2)
outputVecI = [A, -A, A, -A]
outputVecQ = [A, A, -A, -A]
r_hat_I = np.array(r_hat_I)
r_hat_Q = np.array(r_hat_Q)
r_hat = np.concatenate((r_hat_I, r_hat_Q), axis = 0)
r_hat = np.reshape(r_hat, (2, len(r_hat_I)))
outputVecI = np.array(outputVecI)
outputVecQ = np.array(outputVecQ)
outputVec = np.concatenate((outputVecI, outputVecQ), axis = 0)
outputVec = np.reshape(outputVec, (2, len(outputVecI)))
symbols = len(r_hat[0])
# symbols_out = np.zeros((1, symbols))
# for i in range(symbols-1):
#     symbols_out[i] = findClosest(r_hat[:,i], outputVec)
symbols_out = findClosestMary(r_hat, outputVec)

# Draw signal space constellation diagram
plt.figure()
plt.plot(r_hat[0,:], r_hat[1,:], linestyle = 'dotted')
plt.xlabel('x_0')
plt.ylabel('x_1')
plt.grid()
plt.show()

# Draw signal space constellation diagram in 10 frames
# Skipped in Python version -- similar diagram

###########################################
# Binary Conversion
# INPUT: Symbol values
# OUTPUT: Bit values
data_out = mary2binary(symbols_out,4)

###########################################
# Translate to ascii text
# INPUT: Bits
# OUTPUT: Character vector, message_out
message_out = binvector2str(data_out)
print('Message Recieved: ')
print(message_out)
###########################################
