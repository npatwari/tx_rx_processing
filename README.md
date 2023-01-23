# Hands-on Session: Over-the-air Narrowband QPSK Modulation and Demodulation

## Measurement collection via SHOUT
We use the SHOUT measurement framework to automate TX/RX functions across multiple POWDER nodes. The measurements of different communication links will later be assigned to several groups for QPSK demodulation.

#### Instantiate an Experiment
1. Log onto [POWDER](https://powderwireless.net/) 
2. Select the [shout-long-measurement](https://www.powderwireless.net/show-profile.php?profile=2a6f2d5e-7319-11ec-b318-e4434b2381fc) profile. If you do not have access to the profile, one can be created via:
    **`Experiments` &rarr; `Create Experiment Profile` &rarr; `Git Repo` &rarr; add [repo link](https://gitlab.flux.utah.edu/frost/proj-radio-meas) &rarr; select the profile**
3. Start an experiment by specifying all necessary parameters [compute node type, radio types and frequency range] and finish. If the radios are not available, please create a reservation ahead of time.


#### SSH into the orchestrator and clients
Once the experiemnt is ready, go to `List View` for the node hostname.
1. Use the following commands to start ssh and tmux sessions for the orchestor:
    ```
    ssh -Y -p 22 -t <username>@<orch_node_hostname> 'cd /local/repository/bin && tmux new-session -A -s shout1 &&  exec $SHELL'
    ssh -Y -p 22 -t <username>@<orch_node_hostname> 'cd /local/repository/bin && tmux new-session -A -s shout2 &&  exec $SHELL'
    ```

2. Use the following command to start a ssh and tmux session for each of the clients:
    ```
    ssh -Y -p 22 -t <username>@<radio_hostname> 'cd /local/repository/bin && tmux new-session -A -s shout &&  exec $SHELL'
    ```
Note: `tmux` allows multiple remote sessions to remain active even when the SSH connection gets disconncted.

#### Transmission and reception 
1. Files to modify before running an experiment:

    (1) **./3.run_cmd.sh**: make sure that the CMD in line 16 is `save_iq_w_tx_file`.
    
    (2) **save_iq_w_tx_file.json**: 
    
    Path: 
        `/local/repository/etc/cmdfiles/save_iq_w_tx_file.json`.
        
    Parameters:
    * `txamps`: which file to transmit samples from.

    * `txrate` and `rxrate`: sampling rate at TX and RX.

    * `txgain` and `rxgain`: TX and RX gain.

    * `txfreq` and `rxfreq`: TX and RX carrier/center frequency.

    * `txclients` and `rxclients`: nodes for transmission and reception.

    * `rxrepeat`: number of repeated sample collection runs.

    * `sync`: whether to enable sync between TX and RX.

    * `nsamps`: number of samples to be collected.

    * `wotxrepeat`: number of repeated sample collection runs without TX.

2. Orchestrator-client connection

    (1) In one of your `orch` SSH sessions, run:
        ```
        ./1.start_orch.sh
        ```
        
    (2) In the SSH session for each of the clients, run:
        ```
        ./2.start_client.sh
        ```
        
3. Measurement collection
    In your other `orch` SSH session, run:

    ```
    ./3.run_cmd.sh
    ```
## QPSK Demodulation
#### SDR node pairs used for this session
| Group Number   | Link |
| --- | --- |
| 1 | cbrssdr1-ustar-comp <--> cbrssdr1-hospital-comp |
| 2 | cbrssdr1-ustar-comp <--> cbrssdr1-browning-comp | 
| 3 | cbrssdr1-smt-comp <--> cbrssdr1-honors-comp | 
| 4 | cbrssdr1-honors-comp <--> cbrssdr1-hospital-comp |
| 5 | cbrssdr1-smt-comp <--> cbrssdr1-ustar-comp | 
| 6 | cbrssdr1-fm-comp <--> cbrssdr1-bes-comp | 
| 7 | cbrssdr1-bes-comp <--> cbrssdr1-browning-comp | 
| 8 | cbrssdr1-honors-comp <--> cbrssdr1-ustar-comp | 

#### Measurement analysis

For the demonstration, we will analyze the received signals on Google Colab as our python notebook.  (You can also certainly run the python notebook locally on your own Jupyter Notebook if you have one installed on your computer.)   

Compress the collected measuremnt folder using zip in order to upload all its files to the Colab notebook.  Or if you want a previously collected dataset, there are some in this repo.

You will then load [our python notebook on Google Colab](https://colab.research.google.com/drive/1g2f8LmdU5wFYMR0MdZjbAmKMLLIxUWLe?usp=sharing).  You'll follow all of the instructions on this notebook.  That includes making a copy of the notebook (the linked file is read only); uploading the zipped measuremnt file; and picking the `txloc` and `rxloc` and `repNum` for the measurement you will analyze.  

In the demo, we will attempt to organize so that each person / group looks at a different link.
