import uhd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", default="", type=str)

    parser.add_argument("-f", "--freq", type=float, required=True)
    parser.add_argument("-r", "--rate", default=2e6, type=float)
    parser.add_argument("-d", "--duration", default=5.0, type=float)
    parser.add_argument("-c", "--channels", default= 0, nargs="+", type=int)
    parser.add_argument("-g", "--gain", type=int, default=20)
    return parser.parse_args()

def Mod_TX():
    M = 2 # bits per symbol (i.e. 2 in QPSK modulation)
    Information_to_transmit = "Mobile Wireless Week 2023"
    binary = ''.join(format(ord(i), '08b') for i in Information_to_transmit)
    data_bits = np.zeros((len(binary),))
    for i in range(len(binary)):
        data_bits[i] = binary[i]
    
    # Convert serial data to parallel
    def Serial_to_Parallel(x):
        return x.reshape((len(x)//M, M))
    parallel_bits = Serial_to_Parallel(data_bits)

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
    QPSK_signal = Mapping(parallel_bits)

    return QPSK_signal
def main():

    signal = Mod_TX()
    
    args = parse_args()
    usrp = uhd.usrp.MultiUSRP(args.args)
    if not isinstance(args.channels, list):
        args.channels = [args.channels]
        
    usrp.send_waveform(signal, args.duration, args.freq, args.rate,
                       args.channels, args.gain)
    """
    tb = usrp.send_waveform(signal, args.duration, args.freq, args.rate,
                       args.channels, args.gain)
    
    try:
        tb.run()
    except KeyboardInterrupt:
        pass
    """
if __name__ == "__main__":
    main()
