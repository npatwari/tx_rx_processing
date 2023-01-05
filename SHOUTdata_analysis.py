#! /usr/bin python

import json
import numpy as np
import pandas as pd
from scipy import signal, stats
import scipy
import time
import h5py
import datetime
import math
import matplotlib.pyplot as plt
from utm import from_latlon
from digicomm import *

GPS_LOCATIONS = {
    "cbrssdr1-browning-comp": (40.76627,-111.84774),
    "cellsdr1-browning-comp": (40.76627,-111.84774),
    "cbrssdr1-hospital-comp": (40.77105,-111.83712),
    "cellsdr1-hospital-comp": (40.77105,-111.83712),
    "cbrssdr1-smt-comp": (40.76740,-111.83118),
    "cellsdr1-smt-comp": (40.76740,-111.83118),
    "cbrssdr1-ustar-comp": (40.76895,-111.84167),
    "cellsdr1-ustar-comp": (40.76895,-111.84167),
    "cbrssdr1-bes-comp": (40.76134,-111.84629),
    "cellsdr1-bes-comp": (40.76134,-111.84629),
    "cbrssdr1-honors-comp": (40.76440,-111.83699),
    "cellsdr1-honors-comp": (40.76440,-111.83699),
    "cbrssdr1-fm-comp": (40.75807,-111.85325),
    "cellsdr1-fm-comp": (40.75807,-111.85325),
    "moran-nuc1-b210": (40.77006,-111.83784),
    "moran-nuc2-b210": (40.77006,-111.83784),
    "ebc-nuc1-b210": (40.76770,-111.83816),
    "ebc-nuc2-b210": (40.76770,-111.83816),
    "guesthouse-nuc1-b210": (40.76627,-111.83632),
    "guesthouse-nuc2-b210": (40.76627,-111.83632),
    "humanities-nuc1-b210": (40.76486,-111.84319),
    "humanities-nuc2-b210": (40.76486,-111.84319),
    "web-nuc1-b210": (40.76791,-111.84561),
    "web-nuc2-b210": (40.76791,-111.84561),
    "bookstore-nuc1-b210": (40.76414,-111.84759),
    "bookstore-nuc2-b210": (40.76414,-111.84759),
    "sagepoint-nuc1-b210": (40.76278,-111.83061),
    "sagepoint-nuc2-b210": (40.76278,-111.83061),
    "law73-nuc1-b210": (40.76160,-111.85185),
    "law73-nuc2-b210": (40.76160,-111.85185),
    "garage-nuc1-b210": (40.76148,-111.84201),
    "garage-nuc2-b210": (40.76148,-111.84201),
    "madsen-nuc1-b210": (40.75786,-111.83634),
    "madsen-nuc2-b210": (40.75786,-111.83634),
    "cnode-wasatch-dd-b210": (40.77107659230161, -111.84315777334905),
    "cnode-mario-dd-b210": (40.77281166195038, -111.8409002868012),
    "cnode-moran-dd-b210": (40.77006,-111.83872),
    "cnode-guesthouse-dd-b210": (40.76769,-111.83609),
    "cnode-ustar-dd-b210": (40.76851381022761, -111.8404502186636), 
    "cnode-ebc-dd-b210": (40.76720,-111.83810),
}

gold_codes_dict = {
    31:{0:np.array([1,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1])},
    63:{0:np.array([0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,0,0,0,0,1,0])},
    127:{0:np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,1,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1])},
    }

def calcDistLatLong(coord1, coord2):
    '''
    Calculate distance between two nodes using lat/lon coordinates
    INPUT: coordinates, coord1, coord2
    OUTPUT: distance between the coordinates
    '''
    R = 6373000.0 # approximate radius of earth in meters

    lat1 = math.radians(coord1[0])
    lon1 = math.radians(coord1[1])
    lat2 = math.radians(coord2[0])
    lon2 = math.radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a    = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return dist

def get_time_string(timestamp):
    '''
    Helper function to get data and time from timestamp
    INPUT: timestamp
    OUTPUT: data and time. Example: 01-04-2023, 19:50:27
    '''
    date_time = datetime.datetime.fromtimestamp(int(timestamp))
    return date_time.strftime("%m-%d-%Y, %H:%M:%S")

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
                                txrxloc.setdefault(tx, []).append(rx)
                                rxsamples = dataset[cmd][cmd_time][tx][tx_gain][rx_gain][rx]['rxsamples'][:repeat, :]
                                data.setdefault(tx, []).append(np.array(rxsamples))

        else:                       
            print('Unsupported command: ', cmd)

    return data, noise, txrxloc

def get_gold_code(nbits, code_id):
    return gold_codes_dict[nbits][code_id]

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
    gold_id = config_dict['gold_id']
    wotxrepeat = config_dict['wotxrepeat']
    gold_length = config_dict['gold_length']
    if 'samps_per_chip' not in config_dict:
        samps_per_chip = 2
    else:
        samps_per_chip = config_dict['samps_per_chip']
    gold_code = get_gold_code(gold_length, gold_id) * 2 - 1
    long_gold_code = np.repeat(gold_code, samps_per_chip)
    rxrepeat = config_dict['rxrepeat']

    return samps_per_chip, wotxrepeat, rxrate, long_gold_code

def main(folder):
    '''
    main function
    '''
    # load parameters from the json script
    jsonfile = 'save_iq_w_tx_gold.json'
    samps_per_chip, Wotxrepeat, samp_rate, goldcode = JsonLoad(folder, jsonfile)

    # load data
    rx_data, _, txrxloc = traverse_dataset(folder)

    for txloc in rx_data:
        # check one specific transmitter
        if txloc != 'cbrssdr1-honors-comp':continue
        print('\n\n\nTX: {}'.format(txloc))
        rx_data[txloc] = np.vstack(rx_data[txloc])
        print('[Debug] data shape', np.shape(rx_data[txloc]))
        
        # measurements vs. distance
        plt.figure()
        for j, rxloc in enumerate(txrxloc[txloc]):
            # check one specific receiver
            # if rxloc != 'cbrssdr1-ustar-comp':continue
            print('RX: {}'.format(rxloc))

            namelist = (rxloc.split('-'))
            print('namelist',namelist)
            dis = (calcDistLatLong(GPS_LOCATIONS[txloc], GPS_LOCATIONS[rxloc]))
            
            # Example: DSSS sequence correlation
            # matched filtering
            pulse = SRRC(0.5, samps_per_chip, 5)
            MFsamp = signal.lfilter(pulse, 1, rx_data[txloc][j,:].real) + 1j* \
                     signal.lfilter(pulse, 1, rx_data[txloc][j,:].imag)
            # Despread with gold code
            correlation = signal.correlate(MFsamp, goldcode)[len(goldcode)//2-1:-(len(goldcode)//2)]

            # Symbol Samples 
            corr_abs = abs(correlation)
            height = (np.mean(corr_abs)+corr_abs.max())/2
            idx = signal.find_peaks(corr_abs, distance=len(goldcode)-4, height=height)[0] 
            yhat = correlation[idx]

            # custom processing functions implemented here

            power = np.mean(10.0 * np.log10(np.abs(yhat)**2))
            arrRX = (power)

            plt.plot(dis, arrRX,'o',label='RX: '+namelist[1])
        plt.xlabel('Distance (m)',fontsize=15)
        plt.ylabel('RSSI (dB)',fontsize=15)
        plt.legend(ncol=2)
        plt.title('TX: {}'.format(txloc),fontsize=15)
        plt.show()

if __name__=="__main__":
    temp_folder = "BS/Shout_meas_12-30-2022_00-19-23" 
    main(temp_folder)