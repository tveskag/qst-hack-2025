This folder contains the data and scripts needed to build a classifier for photon number detection.
The "8-pixel x photon.csv" files define single and double photon detections, which can be used as reference to ideal photon detection.
"psnspd.py" defines a class for generating signals with noise from "noise.py" and randomised shifts for each photon. 
"PSNSPD_data.py" calls this class, and parsing an integer to the script returns the desired number of generated signals, as a .csv file, as well as the number of photons detected in the signal. 

Start by installing the required packages in the directory. For Shell: "python3 -m venv venv && pip3 install -r requirements.txt"
To generate x signals write in the terminal: "python3 PSNSPD_data.py --signal_no x"
if no input is given, the default is 100 signals.
