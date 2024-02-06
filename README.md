## Over-the-air Narrowband QPSK Modulation and Demodulation

This tutorial goes step-by-step through the process of transmitting and receiving a narrowband digitally modulated packet over the air in [POWDER](https://powderwireless.net/).

First, a big picture view of the software we use to run this tutorial.  We create a modulated packet signal in Python, and save it as a file. We start an experiment on POWDER with a few nodes, and specify a quiet and available frequency band to operate in.  We use the SHOUT measurement framework to automate TX/RX operations across multiple POWDER nodes, specifying the particular parameters for our experiment in a JSON file.  The shout framework allows us to transmit from one node, and receive at all of the other nodes; and then to iteratively switch the transmitting node (and receiving nodes). After Shout runs, we collect a file with the receivers' complex baseband sampled signals; we use Python to run the receiver.

This tutorial assumes you have an account on [POWDER](https://powderwireless.net/) and that you're logged in.

### Reserve Resources

Typically, the POWDER over-the-air resources are in high demand. If you want to make sure you're going to be able to run an experiment on any particular day, it's a good idea to reserve the resources you need. In my example, I went to [Reserve Resources](https://www.powderwireless.net/resgroup.php) and reserved: 

- Time: 9am - 6pm Tuesday, Feb 6.
- Nodes: 
  1. cbrssdr1-bes
  2. cbrssdr1-browning
  3. cbrssdr1-honors
  4. cbrssdr1-hospital
  5. cbrssdr1-meb
- Frequency range: 3395-3405 MHz

My choice of nodes and frequency band was based on the [current availability](https://www.powderwireless.net/resinfo.php). Your availability will vary. We only need about 0.5 MHz for the experiment I run, but I want to 
- make sure my sidelobes don't exceed my reservation limits, and
- make sure I have some space in case there is some narrowband interference at the time I run my experiment to move within my reserved band.

In addition what POWDER says is an available frequency, one should see if there is usually interference in that band. We don't want to compete with other strong signals in the band. Beyond trying to be nice to other wireless systems in the area, it is just harder to demodulate our signal if it is on top of some other wireless signal. To check, go to the [Powder Radio Info](https://www.powderwireless.net/radioinfo.php) page, and look in the "RF Scans" column. There is an icon link if this is a radio that is regularly measured. For example, look at the [Honors rooftop node scan](https://www.powderwireless.net/frequency-graph.php?baseline=1&endpoint=Emulab&node_id=cbrssdr1-honors&iface=rf0). In this plot you need to ignore the regular spikes (these are due to LO feedthrough). The title, which for me is "Spectrum Monitoring Graph for Emulab cbrssdr1-honors:rf0 02/06/2024 8:04:29 AM", says the cluster (Emulab), the node name (cbrssdr1-honors), the name of the antenna port (rf0), and the date and time when the scan was taken.  Look atleast at the nodes you intend to reserve, and find at least a few MHz that does not look occupied at any of your nodes.
 
### Instantiate an Experiment
When it is time to run your experiment, use "Experiments: Start Experiment" on the Powder website to begin.

2. *Select a Profile*. Use the "Change Profile" button and the search field to search for "shout-long-2024". the [shout-long-measurement](https://www.powderwireless.net/show-profile.php?profile=2a6f2d5e-7319-11ec-b318-e4434b2381fc) profile.  Click "Next" to advance to the next step.
 - If you do not have access to the profile, one can be created via the "Experiments: Create Experiment Profile". In this case, fork and modify the [existing profile repository](https://gitlab.flux.utah.edu/npatwari/proj-radio-meas). Use your new git repo as the "Source". Pick a name for your profile, and the project group (the group you are sharing it with), and click "Create".
    
3. *Parameterize*. Specifying all necessary parameters.
 - Leave the Orchestrator and Compute Node Types to the default.
 - On the “Dataset to connect” select “None”.
 - Under "CBAND X310 radios." I create five Radio sites in this category as I listed in the Reserve Resources section above. (Most are self-explanatory, but "bes" = "Behavioral" and "fm" = "Friendship Manor").
 - Under "Frequency ranges for CBAND operation" I put in my min and max frequencies, the number only, in MHz.
Click "Next" when finished.

4. *Finalize*. The name must be 16 charaters or less, so I put in my initials and some minimal abbreviations for what I'm doing. You want it to be unique so you can share it with someone else if they want to look at / debug it.  Select your project and Emulab cluster. Click "Next".
  
5. *Schedule*. Put in the end time (at the least) so that someone else can use the resources after you plan to be done. Click "Finish".

#### SSH into the orchestrator and clients
Once the experiemnt is ready, go to `List View` for the node hostnames.  Each host is listed with an SSH command for you. Mine says `ssh npatwari@pc11-fort.emulab.net` in the row labelled `orch`.  This means that my username is `npatwari` and the orchestrator node is `pc11-fort.emulab.net`. There are five other `-comp` nodes, each with a hostname.  Those are my "radio hostnames".

Hope you have some screen space! We're going to need two terminals connected to the orchestrator, and one terminal for each radio hostname.

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

### Editing and Uploading Files

We often need to change the 1) experiment parameters `save_iq_w_tx_file.json` and 2) transmitted signal file <TX signal.iq> after instantiating the experiment. While the contents of the repository are copied, if you want to edit the parameters file on your local machine after instatiation, you need to copy the edited files to the orchestrator and each radio host.

I use the following commands to do this:

    ```
    scp <path to my local repo>/etc/cmdfiles/save_iq_w_tx_file.json <username>@<orch_node_hostname>:/local/repository/etc/cmdfiles/save_iq_w_tx_file.json 
    scp <path to my local repo>/shout/QPSK_signal_2024-02-06.iq <username>@<orch_node_hostname>:/local/repository/shout/<TX signal.iq>
    ```

#### Transmission and reception 
1. Files to modify before running an experiment:

   - `./3.run_cmd.sh`: make sure that the CMD in line 16 is `save_iq_w_tx_file`.    
   - `save_iq_w_tx_file.json`: 
    
    Path: 
        `/local/repository/etc/cmdfiles/save_iq_w_tx_file.json`.
        
    Parameters:
    * `txsamps`: which file to transmit samples from. Note that the file should contain complex samples as float32 IQIQ...

    * `txrate` and `rxrate`: sampling rate at TX and RX.

    * `txgain` and `rxgain`: TX and RX gain.

    * `txfreq` and `rxfreq`: TX and RX carrier/center frequency. Make sure both match!

    * `txclients` and `rxclients`: nodes for transmission and reception.

    * `rxrepeat`: number of repeated sample collection runs.

    * `sync`: whether to enable sync between TX and RX. "true" enables the use of the external (white rabbit) synchronization.

    * `nsamps`: number of samples to be collected.

    * `wotxrepeat`: number of repeated sample collection runs without TX.

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
    In your other `orch` SSH session, run:

    ```
    ./3.run_cmd.sh
    ```

4. Wait for Shout to complete. 
5. Transfer the measurement file back to the local host. From your local terminal window:
   ```
   scp -r <username>@<orch_node_hostname>:/local/data/Shout_meas_<datestr>_<timestr> /<local_dir>
   ```
   
#### Measurement analysis

For the demonstration, we will analyze the received signals on Google Colab as our python notebook.  (You can also certainly run the python notebook locally on your own Jupyter Notebook if you have one installed on your computer.)   

Compress the collected measuremnt folder using zip in order to upload all its files to the Colab notebook.  Or if you want a previously collected dataset, there are some in this repo.

You will then load [our python notebook on Google Colab](https://colab.research.google.com/drive/1g2f8LmdU5wFYMR0MdZjbAmKMLLIxUWLe?usp=sharing).  You'll follow all of the instructions on this notebook.  That includes making a copy of the notebook (the linked file is read only); uploading the zipped measuremnt file; and picking the `txloc` and `rxloc` and `repNum` for the measurement you will analyze.  
