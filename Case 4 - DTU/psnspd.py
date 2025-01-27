# Model of IDQ P-SNSPD detector

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scipy.signal as signal


class Psnspd:
    def __init__(self, decimate=None):
        """
        Initialize the Psnspd class and load the waveform data from CSV files.

        Parameters:
            decimate: default decimation setting of the original 50 GSPS signal. Some functions
                      have a decimation parameter, which will override this value.
        """
        if decimate is None:
            self.decimate = 1
        else:
            self.decimate = decimate

        package_root = os.getcwd()

        self.sample_time = 0.02  # 'ns'

        # Construct paths relative to the package root directory
        file1 = os.path.join(package_root, "8-pixel 1 photon.csv")
        file2 = os.path.join(package_root, "8-pixel 2 photon.csv")
        file3 = os.path.join(package_root, "noise.csv")

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)

        self.s1 = df1['CH1'].to_numpy() * 1e-3  # Scale factor
        self.s2 = df2['CH1'].to_numpy()
        self.s3 = df3['CH1'].to_numpy()

        ph1 = self.s1[748:1416]  # Define the photon signals
        ph2 = self.s2[744:1385]
        # Max length is 35 ns
        self.photon1 = self.s1[748:]
        self.photon2 = np.concatenate((ph2, np.zeros(750 - len(ph2))))

    def low_pass_filter(self, input_signal, bandwidth=1e9):
        """
        Apply a low-pass filter to the given signal

        Parameters:
            input_signal: The signal to which the low-pass filter will be applied.
            bandwidth: The cutoff bandwidth for the low-pass filter (default is 1 GHz).

        Returns:
            band_limited_signal: The band-limited signal after applying the low-pass filter.
        """
        # Parameters
        dt = self.sample_time * 1e-9  # 's', time step (20 ps/sample)
        fs = 1 / dt  # Sampling frequency in Hz

        # Design a low-pass filter to limit bandwidth
        nyquist = 0.5 * fs
        normal_cutoff = bandwidth / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)

        # Pad the signal before filtering (symmetric padding)
        pad_width = 100  # Number of samples to pad at each end
        padded_signal = np.concatenate((np.zeros(pad_width), input_signal, np.zeros(pad_width)))

        # Apply the low-pass filter to the padded signal
        filtered_padded_signal = signal.lfilter(b, a, padded_signal)

        # Remove the padding
        band_limited_signal = filtered_padded_signal[pad_width:-pad_width]

        return band_limited_signal

    def click(self, bandwidth=None, decimate=None):
        """Return single photon click pulse 30 ns in length. Default uses 20 ps sample period.

        Parameters:
            bandwidth: low-pass filter photon click pulse with the specified bandwidth [Hz]
            decimate: subsample the 50 GSPS pulse by taking every 'decimate' sample, e.g. decimate = 50
                      corresponds to 1 GSPS pulse. If bandwidth is not specified will set it to 90% of
                      Shannons bandwidth, = 0.9 * 1/2 1/T, T = sample period"""
        if decimate is None:
            decimate = self.decimate
        if bandwidth is None and decimate == 1:
            return self.photon1, 1, self.sample_time
        if bandwidth is None:
            # set bandwidth to 90% of Shannon bandwidth
            bandwidth = 0.5/self.sample_time/decimate*1.e9*0.9
        total_length = int(30 / self.sample_time)
        pulse_full = np.concatenate((self.photon1, np.zeros(total_length - len(self.photon1))))
        pulse_signal = self.low_pass_filter(pulse_full, bandwidth=bandwidth)

        pulse_signal = pulse_signal[::decimate]

        return pulse_signal, 1, self.sample_time*decimate

    @staticmethod
    def rnd_shifts(photons=None):
        """Generate a list of random photon shifts/delays, which can be used as argument to 'sspd()'. The list
        will contain 'photons' elements, default being a random number of photons uniformly distributed 0-4
        (inclusive). The first photon (if >0 photons) is uniformly delayed 0 ns - 10 ns. Subsequent photons
        have delays following an exponential distribution with tau = 4 ns, similar to the NKT laser in
        Computinglab."""
        if photons is None:
            photons = np.random.randint(5)
        if photons == 0:
            return []
        elif photons == 1:
            delays = [np.random.uniform(0.0, 10.0)]
        else:
            delays = np.concatenate(([np.random.uniform(0.0, 10.0)], np.random.exponential(4.0, photons - 1)))
        return np.cumsum(delays)

    @staticmethod
    def sig_add(sig1, sig2, offset):
        """Add 'sig2' to 'sig1' af offset 'offset' in 'ns', assuming 20 ps/sample.

        Returns:
            waveform with same length as 'sig1'"""
        zero_pre = np.zeros(round(50 * offset))
        sig2b = np.concatenate((zero_pre, sig2))

        post_cnt = len(sig1) - len(sig2b)
        if post_cnt < 0:
            sig2b =  sig2b[:len(sig1)]
        else:
            sig2b = np.concatenate((sig2b, np.zeros(post_cnt)))
        return sig1 + sig2b

    def sspd_noise(self, n_samples=3000, rms_noise=0.01, bandwidth=1e9):
        """Generate noise signal similar to what is seen from P-SNSPD"""
        pad_width = 100
        white_noise = np.random.randn(n_samples+2*pad_width)
        band_limited_noise = self.low_pass_filter(white_noise, bandwidth)
        band_limited_noise = band_limited_noise[pad_width:-pad_width]

        # Adjust the RMS value of the noise to 0.01
        current_rms = np.sqrt(np.mean(band_limited_noise**2))
        scaling_factor = rms_noise / current_rms
        band_limited_noise = band_limited_noise * scaling_factor

        return band_limited_noise

    def signal(self, shifts=None, rms_noise=0.01, decimate=None):
        """
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
        """
        total_time = 100  # 'ns', total time of returned signal
        offset = 30       # 'ns', offset where 1st click is placed
        if shifts is None:
            shifts = self.rnd_shifts()
        elif type(shifts) is int:
            shifts = self.rnd_shifts(shifts)
        sig_base = np.zeros(int(total_time / self.sample_time))
        for shift_val in shifts:
            sig_base = self.sig_add(sig_base, self.click(decimate=1)[0], offset + shift_val)
        sig_base = sig_base + self.sspd_noise(n_samples=len(sig_base), rms_noise=rms_noise)

        if decimate is None:
            decimate = self.decimate
        if decimate != 1:
            sig_base = self.low_pass_filter(sig_base, bandwidth=0.5/self.sample_time/decimate*1.e9*0.9)
            sig_base = sig_base[::decimate]

        return sig_base, len(shifts), self.sample_time*decimate

    @staticmethod
    def plot(spd, **plotargs):
        """Plot 'signal()' output with a nicely formatted x-axis in 'ns'."""

        # Determine the signal and label based on the input
        if isinstance(spd, tuple) and len(spd) == 3:  # If spd is a tuple (signal, label)
            sig, label, sample_time = spd
            label = plotargs.pop('label', label)
        else:  # If spd is a single signal without a label
            sig = spd
            sample_time = 0.02  # 'ns'
            label = plotargs.pop('label', ' ')

        # Generate x-axis values based on the sample time
        x_axis = np.arange(len(sig)) * sample_time

        # Plot the signal
        plt.plot(x_axis, sig, label=label, **plotargs)

        # Set axis labels and other plot attributes
        plt.ylabel('[V]')
        plt.xlabel('[ns]')
        plt.grid(True)
        plt.autoscale(axis='y')
        plt.legend()
