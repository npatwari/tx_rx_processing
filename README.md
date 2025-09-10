## Over-the-air Narrowband QPSK Modulation and Demodulation

This tutorial goes step-by-step through the process of transmitting and receiving a narrowband digitally modulated packet over the air in [POWDER](https://powderwireless.net/).

First, a big picture view of the software we use to run this tutorial.  We create a modulated packet signal in Python, and save it as a file. We start a shout-long-measurement experiment on POWDER with two (or potentially more) SDR nodes, and specify a quiet and available frequency band to operate in.  We use the Shout framework to automate TX/RX operations across multiple POWDER SDR nodes, specifying the particular parameters for our experiment in a JSON file.  The shout framework allows us to transmit from one node, and receive at all of the other nodes; and then to iteratively switch the transmitting node (and receiving nodes). After Shout runs, we collect a file with the receivers' complex baseband sampled signals. Finally we use Python to run the receiver, observe the received signal, and demodulate the data.

This tutorial assumes you have an account on [POWDER](https://powderwireless.net/) and that you're logged in.

### Reserve Resources

Typically, the POWDER over-the-air resources are in high demand. If you want to make sure you're going to be able to run an experiment on any particular day, it's a good idea to reserve the resources you need. In my example, I went to [Reserve Resources](https://www.powderwireless.net/resgroup.php) and reserved: 

- Nodes: 
  1. cbrssdr1-bes
  2. cbrssdr1-fm
- Frequency range: 3390-3400 MHz

My choice of nodes and frequency band was based on the [current availability](https://www.powderwireless.net/resinfo.php). I also checked the [POWDER Radio Info Page](https://www.powderwireless.net/radioinfo.php) to find the received power graph for potential nodes to check for current interference in the band of interest. We don't want to compete with other strong signals in the band. Beyond trying to be nice to other wireless systems in the area, it is just harder to demodulate our signal if it is on top of some other wireless signal.  I do this by clicking on the three vertical bars icon in the "Monitor" column, for the row with node "cbrssdr1-fm" (since that is one I want). I type in 3300 and 3500 into the "Min MHz" and "Max MHz" fields to zoom in. Low levels of interference will be ok (2-4 dB above the average noise floor), but more than that indicates that I should pick a different node or frequency band for my experiment.

Your availability will vary. We only need about 0.5 MHz for the experiment I run, but I want to 
- make sure my sidelobes don't exceed my reservation limits, and
- make sure I have some space in case there is some narrowband interference at the time I run my experiment to move within my reserved band.

 
### Instantiate an Experiment
When it is time to run your experiment, use "Experiments: Start Experiment" on the Powder website to begin.

2. *Select a Profile*. Use the "Change Profile" button and the search field to search for "shout-long-measurement". the [shout-long-measurement](https://gitlab.flux.utah.edu/frost/proj-radio-meas) profile.  Click "Next" to advance to the next step.
 - If you do not have access to the profile, one can be created via the "Experiments: Create Experiment Profile". In this case, fork and modify the [existing profile repository](https://gitlab.flux.utah.edu/npatwari/proj-radio-meas). Use your new git repo as the "Source". Pick a name for your profile, and the project group (the group you are sharing it with), and click "Create".
    
3. *Parameterize*. Specifying all necessary parameters.
 - Leave the Orchestrator and Compute Node Types to the default.
 - On the “Dataset to connect” select “None”.
 - Under "CBAND X310 radios." Add the Radio Sites. I added those listed above in the Reserve Resources section above. (Most are self-explanatory, but there are two exceptions: "bes" = "Behavioral" and "fm" = "Friendship Manor").
 - Under "Frequency ranges for CBAND operation" I put in my min and max frequencies from my reservation, the number only, in MHz.
Click "Next" when finished.

4. *Finalize*. The name must be 16 charaters or less, so I put in my initials and some minimal abbreviations for what I'm doing, and a digit or two. You want it to be unique so you can share it with someone else if they want to look at / debug it. Don't repeat the exact name you used in prior experiments.  Select your project and Emulab cluster. Click "Next".
  
6. *Schedule*. Leave the start time blank to start it immediately. Put in the end time so that someone else can use the resources after you plan to be done. Click "Finish".

### SSH into the orchestrator and clients
Once the experiemnt is ready, go to `List View` for the node hostnames.  Each host is listed with an SSH command for you. Mine says `ssh npatwari@pc11-fort.emulab.net` in the row labelled `orch`.  This means that my username is `npatwari` and the orchestrator node is `pc11-fort.emulab.net`. Each rooftop X310 is listed, and each has a `-comp` compute node, with a hostname.  

Hope you have some screen space! We're going to need two terminals connected to the orchestrator, and one terminal for each radio hostname. I 

1. Use the following commands to start ssh and tmux sessions for the orchestor. Run the two lines separately, one per terminal.

    ```
    ssh -Y -p 22 -t <username>@<orch_node_hostname> 'cd /local/repository/bin && tmux new-session -A -s shout1 &&  exec $SHELL'
    ssh -Y -p 22 -t <username>@<orch_node_hostname> 'cd /local/repository/bin && tmux new-session -A -s shout2 &&  exec $SHELL'
    ```

3. Use the following command to start a ssh and tmux session for each of the clients:

    ```
    ssh -Y -p 22 -t <username>@<radio_hostname> 'cd /local/repository/bin && tmux new-session -A -s shout &&  exec $SHELL'
    ```
Note: `tmux` allows multiple remote sessions to remain active even when the SSH connection gets disconncted.

### Check on each SDR

The compute node is connected to a USRP software defined radio, and we need to check that it is running, and that it is running the FPGA image we expect. On each compute node, run

   ```
   uhd_usrp_probe
   ```

Ignore the warnings. But if the output complains about a firmware mismatch, do the following three steps: 
  1. Run:
   ```
   ./update_x310.sh
   ```
  2. After the firmware update finishes, find the corresponding **X310 radio device** in the "List View" on the POWDER web UI. Click the checkbox for each device row where a firmware update was executed, then click the "gear" icon at the top of the column and select "Power Cycle". Confirm to complete the operation and wait about 15 seconds for the devices to come back online. 
  3. Double check that the firmware has updated by running uhd_usrp_probe again.


### Editing the JSON File and Creating the IQ File

We typically need to change the 1) experiment parameters `save_iq_w_tx_file.json` and 2) transmitted signal file <TX signal.iq> after instantiating the experiment. While the contents of the repository are copied, if you want to edit the parameters file on your local machine after instatiation, you need to copy the edited files to the orchestrator and each radio host.

I have the `save_iq_w_tx_file.json` and transmitted signal file on my local computer. I use my python code, `QPSK_rx_sync_meles.ipynb` to generate the .iq file, and I use a text editor on my local computer to edit the `save_iq_w_tx_file.json` file. 

#### JSON File

Some tips on editing your JSON file. These commands assume the JSON file name is `save_iq_w_tx_file.json`. The parameters you're most likely to need to change are in bold below.
        
Parameters:
    * **`txsamps`**: File to transmit samples from. Note that the file should contain complex samples as float32 IQIQ...
    * `txrate` and `rxrate`: sampling rate at TX and RX.
    * `txgain` and `rxgain`: TX and RX gain, which are device dependent. Different SDRs have different ranges of gain integers.
    * **`txfreq` and `rxfreq`**: TX and RX carrier/center frequency. Make sure both match!
    * **`txclients` and `rxclients`:** nodes for transmission and reception.
    * `rxrepeat`: number of repeated sample collection runs.
    * `sync`: whether to enable sync between TX and RX. "true" enables the use of the external (white rabbit) synchronization.
    * `nsamps`: number of samples to be collected.
    * `wotxrepeat`: number of repeated sample collection runs without any TX.

### Copying the files to the POWDER compute nodes

I use the following commands to do this for my username `npatwari`, my orchestrator node `pc11-fort.emulab.net`, and my compute nodes `pc07-fort.emulab.net` and `pc05-fort.emulab.net`. My IQ file was named `QPSK_signal_IH3_2025.iq`. On my local machine, I changed directory to the folder containing my .json and .iq files, and then ran:
```
scp save_iq_w_tx_file.json npatwari@pc11-fort.emulab.net:/local/repository/etc/cmdfiles/save_iq_w_tx_file.json 
scp save_iq_w_tx_file.json npatwari@pc07-fort.emulab.net:/local/repository/etc/cmdfiles/save_iq_w_tx_file.json 
scp save_iq_w_tx_file.json npatwari@pc05-fort.emulab.net:/local/repository/etc/cmdfiles/save_iq_w_tx_file.json 

scp QPSK_signal_IH3_2025.iq npatwari@pc11-fort.emulab.net:/local/repository/shout/QPSK_signal_IH3_2025.iq
scp QPSK_signal_IH3_2025.iq npatwari@pc07-fort.emulab.net:/local/repository/shout/QPSK_signal_IH3_2025.iq 
scp QPSK_signal_IH3_2025.iq npatwari@pc05-fort.emulab.net:/local/repository/shout/QPSK_signal_IH3_2025.iq
```
You could copy these commands to a text editor, and do a search and replace to change the username, node names, and the iq file name.


### Run it

2. Orchestrator-client connection

    (1) In one of your `orch` SSH sessions, run:
        ```
        ./1.start_orch.sh
        ```
        
    (2) In the SSH session for *each* of the clients, run:
        ```
        ./2.start_client.sh
        ```
        
3. Measurement collection
    It is a good idea to check `./3.run_cmd.sh` to make sure that the CMD in line 16 is `save_iq_w_tx_file`.    

    In your other `orch` SSH session, run:

    ```
    ./3.run_cmd.sh
    ```

5. Wait for Shout to complete.
   
7. Transfer the measurement file back to the local host. Check the directory name that has your experiment data saved by, on the orchestrator node, using
   ```
   ls /local/data/
   ```
   Then from your local terminal window:
   ```
   scp -r <username>@<orch_node_hostname>:/local/data/Shout_meas_<datestr>_<timestr> /<local_dir>
   ```
   
### Analyze the Received Signals

For the demonstration, we will analyze the received signals on Google Colab as our python notebook.  (You can also certainly run the python notebook locally on your own Jupyter Notebook if you have one installed on your computer.)   

Compress the collected measuremnt folder using zip in order to upload all its files to the Colab notebook.  Or if you want a previously collected dataset, there are some in this repo.

You will then load [our python notebook on Google Colab](https://colab.research.google.com/drive/1OP3-o0ORI5ho-nJp_lZEhWDgkex6B15j?usp=sharing).  You'll follow all of the instructions on this notebook.  That includes making a copy of the notebook (the linked file is read only); uploading the zipped measurement file; and picking the `txloc` and `rxloc` and `repNum` for the measurement you will analyze.  
