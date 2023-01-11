import numpy as np
def Mod_TX():
    M = 2 # bits per symbol (i.e. 2 in QPSK modulation)
    Information_to_transmit = "Mobile Wireless Week 2023"
    binary = ''.join(format(ord(i), '08b') for i in Information_to_transmit)
    data_bits = np.zeros((len(binary),))
    for i in range(len(binary)):
        data_bits[i] = binary[i]
    
    # Add synch_word to
    sync_word = np.asarray([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    print('sync_word:',sync_word)
    bit_sequence = np.hstack([sync_word, data_bits])
    
    # Add Preamble
    preamle_code = np.asarray([1,1,0,0])
    preamble_swap = preamle_code
    for i in range(16):
        if i ==0:
             preamble_swap = preamle_code
        else:    
            preamble = np.hstack([preamble_swap, preamle_code])
            preamble_swap = preamble
    
    print('preamble:',preamble)
    QPSK_frame = np.hstack([preamble, bit_sequence])
    
    # Convert serial data to parallel
    def Serial_to_Parallel(x):
        return x.reshape((len(x)//M, M))
    parallel_bits = Serial_to_Parallel(QPSK_frame)

    ## maps data_bits into complex value IQ samples
    mapping_table = {
        (0,0) : 1.4142 + 1.4142j,
        (0,1) : -1.4142 + 1.4142j,
        (1,0) : 1.4142 - 1.4142j,
        (1,1) : -1.4142 - 1.4142j
    }

    # mapping
    def Mapping(x):
        return np.array([mapping_table[tuple(b)] for b in x])
    IQ_samples = Mapping(parallel_bits)

    # Adding synchronization bits
    # [1 1 1 0 1 0 1 1 1 0 0 1 0 0 0 0]
    #sync_word = np.arrary([1 1 1 0 1 0 1 1 1 0 0 1 0 0 0 0])
    
    return IQ_samples

QPSK_samples = Mod_TX()
QPSK_samples

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
write_complex_binary(QPSK_samples, 'QPSK_signal.iq')

def get_samps_from_file(filename): 
    '''
    load samples from the binary file
    '''
    # File should be in GNURadio's format, i.e., interleaved I/Q samples as float32
    samples = np.fromfile(filename, dtype=np.float32)
    samps = (samples[::2] + 1j*samples[1::2]).astype((np.complex64)) # convert to IQIQIQ  
    return samps

get_samps_from_file('QPSK_signal.iq')