This folder contains the data and scripts needed to build a classifier for photon-number resolving detection.

- **"PSNSPD_data.py" is the main script needed for generating training data.**

- **"QST-Hack-2025 - DTU Physics - Case - Photon Counting.pdf" contains the problem description incluing the target performance of the classifier**

The other files are there to make this script work. Understanding them is not necessary for building the classifier.
- The "8-pixel x photon.csv" files define single and double photon detections, which can be used as reference to ideal photon detection.
- "psnspd.py" defines a class for generating signals with noise from "noise.py" and randomised shifts for each photon. 

"PSNSPD_data.py" works by calling the class defined in "psnspd.py" and parsing an integer to the script returns the desired number of simulated detection signals, as a .csv file, as well as the number of photons corresponding to each signal. 

To use the script, start by installing the required packages in the directory.\
For Shell (on Linux): "python3 -m venv venv && pip3 install -r requirements.txt".

**To generate N signals, run the script like this: "python3 PSNSPD_data.py --signal_no N".\
If no input is given, the default is 100 signals.**
