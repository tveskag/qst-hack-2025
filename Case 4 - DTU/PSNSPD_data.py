# Generate N PSNSPD data sets.

import numpy as np
import matplotlib.pyplot as plt
from psnspd import Psnspd
import pandas as pd
import argparse

# number of generated signal data sets.

# Set up argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Generate and save signals from P-SNSPD.")
parser.add_argument('--signal_no', type=int, default=100, help="Number of signals to generate (default: 100)")
args = parser.parse_args()
signal_no = args.signal_no

# A Psnspd object can create a signal from a 
# Parallel Superconducting Nanowire Single Photon Detector (PSNSPD). 
# Each photon detection is set to start after 30 ns.
signaller = Psnspd()


# Initialize a list to store each signal's data for the CSV
data_collection = []

# Generate the signals and structure the data for CSV output
print("Start generating signals...")
for i in range(signal_no):
    signal, photons, sample_time = signaller.signal()
    # Combine the signal array with the photons and sample_time into a single row
    row_data = [photons, sample_time] + list(signal)
    data_collection.append(row_data)
print("Finished generating signals.")
"""
The signal method has the following definition:
    Generate P-SNSPD output signal with zero or more clicks. Vector is 100 ns long using 20 ps/sample (5000 points).
    1st click (if any) is always at 30 ns, followed by zero or more clicks with an exponential distribution
    with tau = 4 ns

    Parameters:
        shifts: list of times of photon clicks. Time 0 ns is placed 30 ns into the returned signal.
        rms_noise: Vrms of white noise added to signal, default 10 mVrms
        decimate:

    Returns:
        signal: 100 ns / P-SNSPD waveform with 0 or more clicks. Signal has at least a 30 ns preamble
                with noise only.
        clicks: number of clicks contained in the signal (0-4)
        sample_time: sample-time of the returned signal


Each item contained in data_collection contains:
    1.) The number of photons detected in the collection.
    2.) The time per bin of collection
    3.) a list of length 5000 with generated data from a PSNSPD in mV.
"""

# Create a pandas DataFrame to structure the data
# The first row has labels for the number of photons, the sample time per bin, and the sample number, from 1 to 5000.
# Each subsequent row will contain the items structered as in the 1.), 2.), 3.) order. 
# The first two columns will be for the number of photons and the sample time.
print("Converting to Pandas dataframe...")
columns = ['Photons', 'Sample Time (ns)'] + [f'Sample {i+1}' for i in range(len(signal))]
df = pd.DataFrame(data_collection, columns=columns)
print("Writing the dataframe to a .csv file for further processing.")
# Write the DataFrame to a CSV file
df.to_csv('generated_signals.csv', index=False)
print("Data successfully written to 'generated_signals.csv'.")